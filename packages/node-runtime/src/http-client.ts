import {
  HttpTransportError,
  clearTimer,
  isHttpTransportError,
  setTimer,
  throwIfCancelled,
  type HttpClient,
  type HttpRequest,
  type HttpResponse,
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
      throwIfCancelled(req.signal);
      if (isHttpTransportError(error)) throw error;
      throw new HttpTransportError(
        "network",
        `HTTP network failure: ${req.url}`,
        // Only a failure that provably happened before any request bytes were
        // sent may be repeated blindly; anything else (reset, hang up) might
        // have reached the server.
        { cause: error, retryableAttempt: failedBeforeRequestSent(error) },
      );
    }
  }
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

function settleRequest(
  operation: Promise<HttpResponse>,
  req: HttpRequest,
  controller: AbortController,
): Promise<HttpResponse> {
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
