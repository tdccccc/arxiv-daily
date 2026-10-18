import {
  HttpTransportError,
  clearTimer,
  isHttpTransportError,
  setTimer,
  throwIfCancelled,
  type HttpClient,
  type HttpRequest,
  type HttpResponse,
  type HttpStreamResponse,
} from "@arxiv-daily/core";

export type FetchLike = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

export class NodeHttpClient implements HttpClient {
  private fetchImpl: FetchLike;

  constructor(fetchImpl?: FetchLike) {
    this.fetchImpl = fetchImpl ?? requireFetch();
  }

  request(req: HttpRequest): Promise<HttpResponse> {
    throwIfCancelled(req.signal);
    const controller = new AbortController();
    const operation = this.performRequest(req, controller.signal);
    return settleRequest(operation, req, controller);
  }

  /**
   * Streaming variant: resolves at response headers, then hands the body out
   * chunk by chunk. `timeoutMs` bounds only the wait for headers. Aborting
   * `req.signal` (or calling `return()` on the body iterator) cancels the body
   * reader, which closes the connection.
   */
  async requestStream(req: HttpRequest): Promise<HttpStreamResponse> {
    throwIfCancelled(req.signal);
    const controller = new AbortController();
    const onAbort = () => controller.abort(req.signal?.reason);
    const detach = () => req.signal?.removeEventListener("abort", onAbort);
    req.signal?.addEventListener("abort", onAbort, { once: true });
    let res: Response;
    try {
      res = await settleRequest(this.fetchResponse(req, controller.signal), req, controller);
    } catch (error) {
      detach();
      throw error;
    }
    const headers: Record<string, string> = {};
    res.headers.forEach((value, key) => {
      headers[key] = value;
    });
    const body = streamBody(res, req, detach);
    if (req.signal?.aborted) {
      await body[Symbol.asyncIterator]().return?.();
      throwIfCancelled(req.signal);
    }
    return { status: res.status, headers, body };
  }

  private async fetchResponse(req: HttpRequest, signal: AbortSignal): Promise<Response> {
    try {
      return await this.fetchImpl(req.url, {
        method: req.method ?? "GET",
        headers: req.headers,
        body: req.body as BodyInit | undefined,
        signal,
      });
    } catch (error) {
      throw toTransportError(error, req);
    }
  }

  private async performRequest(
    req: HttpRequest,
    signal: AbortSignal,
  ): Promise<HttpResponse> {
    try {
      const res = await this.fetchImpl(req.url, {
        method: req.method ?? "GET",
        headers: req.headers,
        body: req.body as BodyInit | undefined,
        signal,
      });
      const headers: Record<string, string> = {};
      res.headers.forEach((value, key) => {
        headers[key] = value;
      });
      if (req.responseType === "arrayBuffer") {
        return {
          status: res.status,
          headers,
          bodyText: "",
          bodyBuffer: await res.arrayBuffer(),
        };
      }
      return {
        status: res.status,
        headers,
        bodyText: await res.text(),
      };
    } catch (error) {
      throw toTransportError(error, req);
    }
  }
}

/** Maps a fetch/body failure to the core contract; cancellation wins over everything. */
function toTransportError(error: unknown, req: HttpRequest): Error {
  try {
    throwIfCancelled(req.signal);
  } catch (cancelled) {
    return cancelled as Error;
  }
  if (isHttpTransportError(error)) return error;
  return new HttpTransportError(
    "network",
    `HTTP network failure: ${req.url}`,
    // Only a failure that provably happened before any request bytes were
    // sent may be repeated blindly; anything else (reset, hang up) might
    // have reached the server.
    { cause: error, retryableAttempt: failedBeforeRequestSent(error) },
  );
}

/**
 * Exposes a fetch body as an async iterator of decoded text. `return()` cancels
 * the reader immediately, even while a read is pending, so a consumer that
 * stops early (or hits a limit) closes the connection instead of draining it.
 */
function streamBody(
  res: Response,
  req: HttpRequest,
  onDone: () => void,
): AsyncIterable<string> {
  const reader = res.body?.getReader();
  const decoder = new TextDecoder();
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    onDone();
  };
  const iterator: AsyncIterator<string> = {
    async next() {
      if (!reader || finished) {
        finish();
        return { done: true, value: undefined };
      }
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (finished) return { done: true, value: undefined };
          if (done) {
            const tail = decoder.decode();
            finish();
            return tail ? { done: false, value: tail } : { done: true, value: undefined };
          }
          const text = decoder.decode(value, { stream: true });
          if (text) return { done: false, value: text };
        }
      } catch (error) {
        finish();
        throw toTransportError(error, req);
      }
    },
    async return() {
      const wasFinished = finished;
      finish();
      if (reader && !wasFinished) await reader.cancel().catch(() => undefined);
      return { done: true, value: undefined };
    },
  };
  return { [Symbol.asyncIterator]: () => iterator };
}

const PRE_SEND_ERROR_CODES: ReadonlySet<string> = new Set([
  // Name resolution and connection establishment.
  "ENOTFOUND", "EAI_AGAIN", "ECONNREFUSED", "EHOSTUNREACH", "ENETUNREACH", "ENETDOWN",
  "UND_ERR_CONNECT_TIMEOUT",
  // TLS handshake / certificate failures (the request is sent after the handshake).
  "EPROTO", "CERT_HAS_EXPIRED", "DEPTH_ZERO_SELF_SIGNED_CERT", "SELF_SIGNED_CERT_IN_CHAIN",
  "UNABLE_TO_VERIFY_LEAF_SIGNATURE", "UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
]);

/** True when `error` (or a cause in its chain) is a connect-phase failure: nothing was sent. */
export function failedBeforeRequestSent(error: unknown): boolean {
  const seen = new Set<unknown>();
  const pending: unknown[] = [error];
  while (pending.length > 0 && seen.size < 16) {
    const current = pending.pop();
    if (!current || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const { code, cause, errors } = current as { code?: unknown; cause?: unknown; errors?: unknown };
    if (typeof code === "string" && (PRE_SEND_ERROR_CODES.has(code) || /^ERR_(?:TLS|SSL)_/.test(code))) {
      return true;
    }
    pending.push(cause);
    if (Array.isArray(errors)) pending.push(...errors);
  }
  return false;
}

function settleRequest<T>(
  operation: Promise<T>,
  req: HttpRequest,
  controller: AbortController,
): Promise<T> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timeout: ReturnType<typeof setTimer> | undefined;

    const cleanup = () => {
      if (timeout) clearTimer(timeout);
      req.signal?.removeEventListener("abort", onAbort);
    };
    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      cleanup();
      fn();
    };
    const onAbort = () => {
      controller.abort(req.signal?.reason);
      settle(() => {
        try {
          throwIfCancelled(req.signal);
        } catch (error) {
          reject(error instanceof Error ? error : new Error(String(error), { cause: error }));
        }
      });
    };

    req.signal?.addEventListener("abort", onAbort, { once: true });
    if (req.signal?.aborted) {
      onAbort();
    } else if (req.timeoutMs && req.timeoutMs > 0) {
      timeout = setTimer(() => {
        const error = new HttpTransportError(
          "timeout",
          `HTTP timeout after ${req.timeoutMs}ms: ${req.url}`,
          { retryableAttempt: false },
        );
        controller.abort(error);
        settle(() => reject(error));
      }, req.timeoutMs);
    }

    operation.then(
      (response) => settle(() => resolve(response)),
      (error: unknown) => settle(() => reject(error instanceof Error ? error : new Error(String(error), { cause: error }))),
    );
  });
}

function requireFetch(): FetchLike {
  if (typeof fetch !== "function") {
    throw new Error("NodeHttpClient requires global fetch");
  }
  return fetch;
}
