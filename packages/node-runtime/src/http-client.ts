import {
  HttpTransportError,
  isHttpTransportError,
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
        { cause: error, retryableAttempt: true },
      );
    }
  }
}

function settleRequest(
  operation: Promise<HttpResponse>,
  req: HttpRequest,
  controller: AbortController,
): Promise<HttpResponse> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;

    const cleanup = () => {
      if (timeout) clearTimeout(timeout);
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
          reject(error);
        }
      });
    };

    req.signal?.addEventListener("abort", onAbort, { once: true });
    if (req.signal?.aborted) {
      onAbort();
    } else if (req.timeoutMs && req.timeoutMs > 0) {
      timeout = setTimeout(() => {
        const error = new HttpTransportError(
          "timeout",
          `HTTP timeout after ${req.timeoutMs}ms: ${req.url}`,
          { retryableAttempt: true },
        );
        controller.abort(error);
        settle(() => reject(error));
      }, req.timeoutMs);
    }

    operation.then(
      (response) => settle(() => resolve(response)),
      (error) => settle(() => reject(error)),
    );
  });
}

function requireFetch(): FetchLike {
  if (typeof globalThis.fetch !== "function") {
    throw new Error("NodeHttpClient requires global fetch");
  }
  return globalThis.fetch.bind(globalThis);
}
