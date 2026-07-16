import { requestUrl } from "obsidian";
import {
  throwIfCancelled,
  type HttpClient,
  type HttpRequest,
  type HttpResponse,
} from "@arxiv-daily/core";

export class ObsidianHttpClient implements HttpClient {
  async request(req: HttpRequest): Promise<HttpResponse> {
    throwIfCancelled(req.signal);
    const res = await requestUrl({
      url: req.url,
      method: req.method ?? "GET",
      headers: req.headers,
      body: req.body,
      throw: false,
    });
    throwIfCancelled(req.signal);
    return {
      status: res.status,
      headers: (res.headers ?? {}) as Record<string, string>,
      bodyText: res.text,
      bodyBuffer:
        req.responseType === "arrayBuffer"
          ? (res as { arrayBuffer?: ArrayBuffer }).arrayBuffer
          : undefined,
    };
  }
}
