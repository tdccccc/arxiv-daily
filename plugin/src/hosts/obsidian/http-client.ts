import { requestUrl } from "obsidian";
import type {
  HttpClient,
  HttpRequest,
  HttpResponse,
} from "../../core/adapters";

export class ObsidianHttpClient implements HttpClient {
  async request(req: HttpRequest): Promise<HttpResponse> {
    const res = await requestUrl({
      url: req.url,
      method: req.method ?? "GET",
      headers: req.headers,
      body: req.body,
      throw: false,
    });
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
