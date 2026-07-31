import { requestUrl } from "obsidian";
import {
  HttpTransportError,
  isHttpTransportError,
  throwIfCancelled,
  type HttpClient,
  type HttpRequest,
  type HttpResponse,
} from "@arxiv-daily/core";

export interface ObsidianRequestResponse {
  status: number;
  headers?: Record<string, string>;
  text: string;
  arrayBuffer?: ArrayBuffer;
}

export type ObsidianRequestImpl = (
  request: Parameters<typeof requestUrl>[0],
) => Promise<ObsidianRequestResponse>;

export class ObsidianHttpClient implements HttpClient {
  constructor(
    private readonly requestImpl: ObsidianRequestImpl = requestUrl,
  ) {}

  request(req: HttpRequest): Promise<HttpResponse> {
    throwIfCancelled(req.signal);
    const operation = this.performRequest(req);
    return settleRequest(operation, req);
  }

  private async performRequest(req: HttpRequest): Promise<HttpResponse> {
    try {
      const res = await this.requestImpl({
        url: req.url,
        method: req.method ?? "GET",
        headers: req.headers,
        body: req.body,
        throw: false,
      });
      return {
        status: res.status,
        headers: res.headers ?? {},
        bodyText: res.text,
        bodyBuffer:
          req.responseType === "arrayBuffer" ? res.arrayBuffer : undefined,
      };
    } catch (error) {
      if (req.signal?.aborted) throwCancelled(req.signal);
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
): Promise<HttpResponse> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timeout: number | undefined;

    const cleanup = () => {
      if (timeout) window.clearTimeout(timeout);
      req.signal?.removeEventListener("abort", onAbort);
    };
    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      cleanup();
      fn();
    };
    const onAbort = () => settle(() => reject(cancellationError(req.signal)));

    req.signal?.addEventListener("abort", onAbort, { once: true });
    if (req.signal?.aborted) {
      onAbort();
    } else if (req.timeoutMs && req.timeoutMs > 0) {
      timeout = window.setTimeout(
          () => settle(() => reject(new HttpTransportError(
          "timeout",
          `HTTP timeout after ${req.timeoutMs}ms: ${req.url}`,
          { retryableAttempt: false },
        ))),
        req.timeoutMs,
      );
    }

    operation.then(
      (response) => settle(() => resolve(response)),
      (error) => settle(() => reject(
        error instanceof Error ? error : new Error(String(error)),
      )),
    );
  });
}

function cancellationError(signal?: AbortSignal): Error {
  try {
    throwIfCancelled(signal);
  } catch (error) {
    return error as Error;
  }
  return new Error("cancelled by user");
}

function throwCancelled(signal: AbortSignal): never {
  throw cancellationError(signal);
}
