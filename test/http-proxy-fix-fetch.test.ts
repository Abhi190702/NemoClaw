// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Regression coverage for NemoClaw#4730.
//
// OpenClaw cron provider preflight can use Node fetch/undici directly. Native
// fetch does not pass through the existing http.request FORWARD-mode rewrite,
// so it attempted raw DNS for https://inference.local/v1 and skipped cron runs.
// The preload now routes only inference.local fetches through the same
// http.request shape that the existing rewrite handles.

import { EventEmitter } from "node:events";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import { Readable } from "node:stream";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const FIX_PATH = path.resolve(
  import.meta.dirname,
  "..",
  "nemoclaw-blueprint",
  "scripts",
  "http-proxy-fix.js",
);

const PROXY_URL = "http://10.200.0.1:3128";

type RewrittenOptions = http.RequestOptions & {
  protocol?: string;
};

type FakeClientRequest = EventEmitter & {
  write: (
    chunk: unknown,
    encoding?: BufferEncoding | ((err?: Error | null) => void),
    cb?: (err?: Error | null) => void,
  ) => boolean;
  end: (
    chunk?: unknown,
    encoding?: BufferEncoding | (() => void),
    cb?: () => void,
  ) => FakeClientRequest;
  destroy: (err?: Error) => FakeClientRequest;
  setTimeout: () => FakeClientRequest;
};

function loadWrapper() {
  delete require.cache[FIX_PATH];
  require(FIX_PATH);
}

function readableResponse(body: string): http.IncomingMessage {
  const res = new Readable({
    read() {
      this.push(body);
      this.push(null);
    },
  }) as http.IncomingMessage;
  res.statusCode = 200;
  res.statusMessage = "OK";
  res.headers = { "content-type": "application/json" };
  return res;
}

function addChunk(chunks: Buffer[], chunk: unknown, encoding?: BufferEncoding | (() => void)) {
  if (chunk == null || typeof chunk === "function") return;
  if (typeof chunk === "string") {
    chunks.push(Buffer.from(chunk, typeof encoding === "string" ? encoding : undefined));
    return;
  }
  if (chunk instanceof ArrayBuffer) {
    chunks.push(Buffer.from(chunk));
    return;
  }
  if (ArrayBuffer.isView(chunk)) {
    chunks.push(Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength));
    return;
  }
  chunks.push(Buffer.from(String(chunk)));
}

describe("http-proxy-fix fetch routing for inference.local (#4730)", () => {
  let origHttpRequest: typeof http.request;
  let origFetch: typeof globalThis.fetch;
  let originalFetchSpy: ReturnType<typeof vi.fn>;
  let httpsSpy: ReturnType<typeof vi.spyOn>;
  let captured: RewrittenOptions | null;
  let capturedBody: string;

  beforeEach(() => {
    origHttpRequest = http.request;
    origFetch = globalThis.fetch;
    originalFetchSpy = vi.fn(async () => new Response("passthrough", { status: 202 }));
    globalThis.fetch = originalFetchSpy as typeof globalThis.fetch;
    delete (globalThis as { __nemoclawFetchPatched?: boolean }).__nemoclawFetchPatched;

    captured = null;
    capturedBody = "";
    vi.stubEnv("NODE_USE_ENV_PROXY", "1");
    vi.stubEnv("HTTPS_PROXY", PROXY_URL);

    loadWrapper();

    httpsSpy = vi.spyOn(https, "request").mockImplementation(
      // @ts-expect-error stubbed request shape is enough for the wrapper.
      (options: RewrittenOptions, callback?: (res: http.IncomingMessage) => void) => {
        captured = options;
        const chunks: Buffer[] = [];
        const req = new EventEmitter() as FakeClientRequest;
        req.write = (chunk, encoding, cb) => {
          addChunk(chunks, chunk, encoding);
          const done = typeof encoding === "function" ? encoding : cb;
          if (done) done();
          return true;
        };
        req.end = (chunk, encoding, cb) => {
          addChunk(chunks, chunk, encoding);
          capturedBody = Buffer.concat(chunks).toString("utf-8");
          const done =
            typeof chunk === "function" ? chunk : typeof encoding === "function" ? encoding : cb;
          if (done) done();
          process.nextTick(() => callback?.(readableResponse('{"ok":true}')));
          return req;
        };
        req.destroy = () => req;
        req.setTimeout = () => req;
        return req;
      },
    );
  });

  afterEach(() => {
    httpsSpy.mockRestore();
    http.request = origHttpRequest;
    globalThis.fetch = origFetch;
    delete (globalThis as { __nemoclawFetchPatched?: boolean }).__nemoclawFetchPatched;
    delete require.cache[FIX_PATH];
    vi.unstubAllEnvs();
  });

  it("routes inference.local fetches through the existing FORWARD-mode rewrite path", async () => {
    const requestBody = JSON.stringify({ model: "inference/gemma4:26b" });

    const response = await fetch("https://inference.local/v1/models", {
      method: "POST",
      headers: {
        Authorization: "Bearer target-token",
        "Content-Type": "application/json",
      },
      body: requestBody,
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ok: true });
    expect(originalFetchSpy).not.toHaveBeenCalled();
    expect(captured).not.toBeNull();
    expect(captured?.hostname).toBe("inference.local");
    expect(captured?.host).toBe("inference.local");
    expect(captured?.port).toBe(443);
    expect(captured?.path).toBe("/v1/models");
    expect(captured?.protocol).toBe("https:");
    expect(captured?.method).toBe("POST");
    expect((captured?.headers as Record<string, string>)?.authorization).toBe(
      "Bearer target-token",
    );
    expect((captured?.headers as Record<string, string>)?.["content-type"]).toBe(
      "application/json",
    );
    expect(capturedBody).toBe(requestBody);
  });

  it("does not intercept non-inference.local fetches", async () => {
    const response = await fetch("https://example.com/v1/models");

    expect(response.status).toBe(202);
    expect(await response.text()).toBe("passthrough");
    expect(originalFetchSpy).toHaveBeenCalledTimes(1);
    expect(httpsSpy).not.toHaveBeenCalled();
  });

  it("is idempotent if the preload is required more than once", async () => {
    const wrapped = globalThis.fetch;
    loadWrapper();
    expect((globalThis as { __nemoclawFetchPatched?: boolean }).__nemoclawFetchPatched).toBe(true);
    expect(globalThis.fetch).toBe(wrapped);

    await fetch("https://inference.local/v1/models");

    expect(httpsSpy).toHaveBeenCalledTimes(1);
  });
});

describe("http-proxy-fix fetch routing without native fetch", () => {
  let origHttpRequest: typeof http.request;
  let origFetch: typeof globalThis.fetch;

  beforeEach(() => {
    origHttpRequest = http.request;
    origFetch = globalThis.fetch;
    delete (globalThis as { fetch?: typeof globalThis.fetch }).fetch;
    delete (globalThis as { __nemoclawFetchPatched?: boolean }).__nemoclawFetchPatched;
    vi.stubEnv("NODE_USE_ENV_PROXY", "1");
    vi.stubEnv("HTTPS_PROXY", PROXY_URL);
  });

  afterEach(() => {
    http.request = origHttpRequest;
    globalThis.fetch = origFetch;
    delete (globalThis as { __nemoclawFetchPatched?: boolean }).__nemoclawFetchPatched;
    delete require.cache[FIX_PATH];
    vi.unstubAllEnvs();
  });

  it("is a no-op when globalThis.fetch is undefined", () => {
    expect(() => loadWrapper()).not.toThrow();
    expect(globalThis.fetch).toBeUndefined();
    expect((globalThis as { __nemoclawFetchPatched?: boolean }).__nemoclawFetchPatched).toBe(
      undefined,
    );
  });
});
