// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// http-proxy-fix.js — transport wrapper resolving proxy mismatches between
// NODE_USE_ENV_PROXY=1 (Node.js 22+) and HTTP libraries that independently
// read HTTPS_PROXY (axios, follow-redirects, proxy-from-env). See
// NemoClaw#2109 and NemoClaw#4730.
//
// Problem:
//   Node.js 22 with NODE_USE_ENV_PROXY=1 (baked into the OpenShell base
//   image) intercepts https.request() calls and handles proxying via a
//   CONNECT tunnel. HTTP libraries also read HTTPS_PROXY and configure
//   HTTP FORWARD mode, so the request is processed twice and the L7 proxy
//   rejects it with "FORWARD rejected: HTTPS requires CONNECT".
//
// Fix:
//   Wrap http.request() — the lowest common denominator many HTTP clients
//   bottoms out at. Detect FORWARD-mode requests (hostname = proxy IP,
//   path = full https:// URL) and rewrite them as https.request() against
//   the real target host, letting NODE_USE_ENV_PROXY handle the CONNECT
//   tunnel correctly.
//
//   Also wrap fetch() only for https://inference.local/*, which OpenClaw cron
//   provider preflight can reach through undici/fetch instead of http.request.
//   The wrapper converts that fetch into the same FORWARD-mode shape handled
//   above, preserving NemoClaw's managed inference.local route while avoiding
//   a raw DNS lookup for the sandbox-only host.
//
// Earlier PR #2110 tried a Module._load hook intercepting require('axios').
// That could not catch follow-redirects + proxy-from-env bundled as ESM in
// OpenClaw's dist/ — there are no require() calls to intercept. The
// http.request wrapper sits below all libraries and catches every path.
//
// This file is the canonical source for review and tests. The Dockerfile
// copies it into /usr/local/lib/nemoclaw/preloads/, then at sandbox boot
// nemoclaw-start.sh writes an identical copy to /tmp/nemoclaw-http-proxy-fix.js
// and loads it via NODE_OPTIONS=--require.

(function () {
  'use strict';
  if (process.env.NODE_USE_ENV_PROXY !== '1') return;

  var http = require('http');
  var origRequest = http.request;

  var proxyUrl =
    process.env.HTTPS_PROXY ||
    process.env.https_proxy ||
    process.env.HTTP_PROXY ||
    process.env.http_proxy ||
    '';
  var proxyHost = '';
  var proxyPort = '';
  var proxyProtocol = '';
  try {
    var parsedProxy = new URL(proxyUrl);
    proxyHost = parsedProxy.hostname;
    proxyPort = parsedProxy.port || '80';
    proxyProtocol = parsedProxy.protocol;
  } catch (_e) {
    /* no usable proxy configured */
  }
  if (!proxyHost) return;

  // Strip headers that were meaningful for the proxy hop only. Once we
  // re-issue against the target via https.request, the original Host
  // points at the proxy and the hop-by-hop headers (RFC 7230 §6.1) leak
  // upstream — they describe the connection between the caller and the
  // proxy, not the rewritten connection to the target.
  //
  // RFC 7230 §6.1 hop-by-hop set (request direction):
  //   Connection, Keep-Alive, Proxy-Authorization, TE, Trailer,
  //   Transfer-Encoding, Upgrade.
  // Also stripped: Host (points at the proxy); Proxy-Connection (de
  // facto deprecated header still emitted by some clients); and
  // Proxy-Authenticate (response-only per RFC 7235 §4.3, included
  // belt-and-suspenders for clients that echo response headers into
  // retry-request options). Plus: per RFC 7230 §6.1, any token named in
  // the Connection header is itself hop-by-hop and must be stripped.
  var STATIC_HOP_BY_HOP = [
    'host',
    'connection',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'proxy-connection',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
  ];

  function sanitizeHeaders(headers) {
    if (!headers || typeof headers !== 'object') return undefined;
    // Collect tokens named in the Connection header — those become
    // hop-by-hop transitively per RFC 7230 §6.1.
    var dynamic = new Set();
    for (var k in headers) {
      if (
        !Object.prototype.hasOwnProperty.call(headers, k) ||
        String(k).toLowerCase() !== 'connection'
      ) {
        continue;
      }
      var raw = headers[k];
      var listed = Array.isArray(raw) ? raw.join(',') : raw;
      if (typeof listed === 'string') {
        listed.split(',').forEach(function (token) {
          var t = token.trim().toLowerCase();
          if (t) dynamic.add(t);
        });
      }
    }
    var staticSet = new Set(STATIC_HOP_BY_HOP);
    var out = {};
    for (var key in headers) {
      if (!Object.prototype.hasOwnProperty.call(headers, key)) continue;
      var lower = String(key).toLowerCase();
      if (staticSet.has(lower) || dynamic.has(lower)) continue;
      out[key] = headers[key];
    }
    return out;
  }

  function fetchInputUrl(input) {
    if (typeof input === 'string') return input;
    if (input && typeof input.url === 'string') return input.url;
    if (input && typeof input.href === 'string') return input.href;
    return '';
  }

  function inferenceLocalFetchUrl(input) {
    var raw = fetchInputUrl(input);
    if (!raw) return null;
    try {
      var target = new URL(raw);
      if (target.protocol !== 'https:' || target.hostname !== 'inference.local') {
        return null;
      }
      return target;
    } catch (_e) {
      return null;
    }
  }

  function requestHeaders(request) {
    var out = {};
    request.headers.forEach(function (value, key) {
      out[key] = value;
    });
    return out;
  }

  function responseHeaders(headers) {
    var out = [];
    Object.keys(headers || {}).forEach(function (key) {
      var value = headers[key];
      if (Array.isArray(value)) {
        value.forEach(function (entry) {
          if (entry != null) out.push([key, String(entry)]);
        });
      } else if (value != null) {
        out.push([key, String(value)]);
      }
    });
    return out;
  }

  function responseBody(method, statusCode, res) {
    if (method === 'HEAD' || statusCode === 204 || statusCode === 304) {
      return null;
    }
    var stream = require('stream');
    if (stream.Readable && typeof stream.Readable.toWeb === 'function') {
      return stream.Readable.toWeb(res);
    }
    return res;
  }

  async function fetchViaForwardProxy(input, init, originalFetch, thisArg) {
    if (proxyProtocol !== 'http:' || typeof Request === 'undefined') {
      return originalFetch.call(thisArg, input, init);
    }

    var request;
    try {
      request = new Request(input, init);
    } catch (_e) {
      return originalFetch.call(thisArg, input, init);
    }

    var target = inferenceLocalFetchUrl(request);
    if (!target) return originalFetch.call(thisArg, input, init);

    var method = request.method || 'GET';
    var headers = requestHeaders(request);
    var body = null;
    if (method !== 'GET' && method !== 'HEAD') {
      body = Buffer.from(await request.clone().arrayBuffer());
      if (
        body.length > 0 &&
        !Object.prototype.hasOwnProperty.call(headers, 'content-length')
      ) {
        headers['content-length'] = String(body.length);
      }
    }

    return new Promise(function (resolve, reject) {
      var req = http.request(
        {
          hostname: proxyHost,
          port: proxyPort,
          path: target.href,
          method: method,
          headers: headers,
          signal: request.signal,
        },
        function (res) {
          var status = res.statusCode || 200;
          resolve(
            new Response(responseBody(method, status, res), {
              status: status,
              statusText: res.statusMessage || '',
              headers: responseHeaders(res.headers),
            })
          );
        }
      );
      req.on('error', reject);
      if (body && body.length > 0) req.write(body);
      req.end();
    });
  }

  /**
   * NemoClaw#4730: OpenClaw 2026.5.27 cron provider preflight reaches the
   * managed provider base URL through fetch()/undici instead of http.request().
   * Native fetch bypasses the FORWARD-mode rewrite above, so it can attempt
   * raw DNS for the sandbox-only inference.local host and skip cron agentTurn
   * runs before normal model calls get a chance to use the proxy-aware path.
   *
   * This shim is intentionally narrow: only https://inference.local/* is
   * converted into the existing proxy path. Other fetches keep their original
   * transport, and the shim does not hardcode Ollama or modify NO_PROXY.
   */
  function wrapFetchForInferenceLocal() {
    if (typeof globalThis.fetch !== 'function') return;
    if (globalThis.__nemoclawFetchPatched) return;
    globalThis.__nemoclawFetchPatched = true;

    var _originalFetch = globalThis.fetch.bind(globalThis);
    var wrappedFetch = async function (input, init) {
      if (!inferenceLocalFetchUrl(input)) {
        return _originalFetch(input, init);
      }
      return fetchViaForwardProxy(input, init, _originalFetch, globalThis);
    };
    wrappedFetch.__nemoclawInferenceLocalProxyFix = true;
    globalThis.fetch = wrappedFetch;
  }

  http.request = function (options, callback) {
    if (typeof options === 'string' || !options) {
      return origRequest.apply(http, arguments);
    }
    if (
      options.hostname === proxyHost &&
      options.path &&
      options.path.startsWith('https://')
    ) {
      var target;
      try {
        target = new URL(options.path);
      } catch (_e) {
        return origRequest.apply(http, arguments);
      }
      var https = require('https');
      // Clone caller's options and overwrite proxy-specific routing
      // fields. Strip fields that were set up for the proxy hop and
      // would misbehave on the rewritten https.request to the target:
      //   - agent: a forward-proxy http.Agent cannot speak TLS. Leaving
      //     it attached caused upstreams like deepinfra to surface as
      //     "LLM request failed: network connection error" while other
      //     upstreams that don't end up on this code path still worked.
      //     On Node 22 https.request throws a synchronous TypeError; on
      //     Node 18/20 it falls through and the TLS handshake fails.
      //   - auth: basic-auth meant for the proxy hop. Leaving it on
      //     would Basic-auth the target server with proxy credentials.
      //   - servername / checkServerIdentity: TLS SNI + cert validation
      //     pre-computed for the proxy hop. Wrong cert chain and wrong
      //     SNI must not survive into the rewrite — drop them so Node
      //     re-derives from the new `hostname`.
      //   - socketPath: Unix-socket proxies exist (e.g. cntlm-style
      //     local proxies). Routing TLS bytes into the proxy's Unix
      //     socket would defeat the entire rewrite.
      //   - localAddress / lookup / family / hints: source-binding and
      //     DNS hints picked for reachability to the proxy. The
      //     rewritten target may not be reachable from the same NIC or
      //     DNS family.
      //   - Host / hop-by-hop headers (RFC 7230 §6.1): stripped via
      //     sanitizeHeaders so Node regenerates Host from `host`/`port`
      //     to point at the real target.
      // Signal (AbortController) and TLS material (ca/cert/key/
      // rejectUnauthorized), timeout, body, and target-intent headers
      // (Authorization, Content-Type, …) are preserved.
      var rewritten = Object.assign({}, options, {
        method: options.method || 'GET',
        hostname: target.hostname,
        host: target.hostname,
        port: target.port || 443,
        path: target.pathname + target.search,
        protocol: 'https:',
        headers: sanitizeHeaders(options.headers),
      });
      delete rewritten.agent;
      delete rewritten.auth;
      delete rewritten.servername;
      delete rewritten.checkServerIdentity;
      delete rewritten.socketPath;
      delete rewritten.localAddress;
      delete rewritten.lookup;
      delete rewritten.family;
      delete rewritten.hints;
      return https.request(rewritten, callback);
    }
    return origRequest.apply(http, arguments);
  };

  wrapFetchForInferenceLocal();
})();
