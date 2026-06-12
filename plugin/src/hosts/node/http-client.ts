import type {
  HttpClient,
  HttpRequest,
  HttpResponse,
} from "../../core/adapters";

export type FetchLike = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

export class NodeHttpClient implements HttpClient {
  private fetchImpl: FetchLike;

  constructor(fetchImpl?: FetchLike) {
    this.fetchImpl = fetchImpl ?? requireFetch();
  }

  async request(req: HttpRequest): Promise<HttpResponse> {
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const abortFromSignal = () => controller.abort(req.signal?.reason);

    if (req.signal?.aborted) {
      abortFromSignal();
    } else {
      req.signal?.addEventListener("abort", abortFromSignal, { once: true });
    }

    if (req.timeoutMs && req.timeoutMs > 0) {
      timeout = setTimeout(
        () => controller.abort(new Error(`HTTP timeout after ${req.timeoutMs}ms`)),
        req.timeoutMs,
      );
    }

    try {
      const res = await this.fetchImpl(req.url, {
        method: req.method ?? "GET",
        headers: req.headers,
        body: req.body as BodyInit | undefined,
        signal: controller.signal,
      });
      const headers: Record<string, string> = {};
      res.headers.forEach((value, key) => {
        headers[key] = value;
      });
      return {
        status: res.status,
        headers,
        bodyText: await res.text(),
      };
    } finally {
      if (timeout) clearTimeout(timeout);
      req.signal?.removeEventListener("abort", abortFromSignal);
    }
  }
}

function requireFetch(): FetchLike {
  if (typeof globalThis.fetch !== "function") {
    throw new Error("NodeHttpClient requires global fetch");
  }
  return globalThis.fetch.bind(globalThis);
}
