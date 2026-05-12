import { requestUrl } from "obsidian";
import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import { parseAtomAbstracts } from "./atom-parser";

export interface ArxivFetcherOptions {
  category: string;
  logger: Logger;
  requestDelayMs: number;
}

export class ArxivFetcher {
  private lastRequestAt = 0;

  constructor(private opts: ArxivFetcherOptions) {}

  /** Fetch the /list/<cat>/recent page with show=2000 to capture all 5 days in one shot. */
  async fetchRecent(): Promise<string> {
    const url = `https://arxiv.org/list/${this.opts.category}/recent?skip=0&show=2000`;
    return this.fetchHtml(url, { allow404: false });
  }

  /**
   * Bulk-fetch abstracts via arXiv's Atom API.
   *
   * Returns a Map keyed by base arXiv id (e.g. "2605.08080" with version stripped).
   * Papers not found in the response are omitted from the map; callers should
   * fall back to an empty abstract for those.
   *
   * arXiv recommends batches of <=300; we conservatively cap at 200.
   */
  async fetchAbstractsByIds(ids: string[]): Promise<Map<string, string>> {
    const out = new Map<string, string>();
    const BATCH = 200;
    for (let i = 0; i < ids.length; i += BATCH) {
      const batch = ids.slice(i, i + BATCH);
      if (batch.length === 0) continue;
      const url = `https://export.arxiv.org/api/query?id_list=${batch.join(",")}&max_results=${batch.length}`;
      const xml = await this.fetchHtml(url, { allow404: false });
      for (const [k, v] of parseAtomAbstracts(xml)) out.set(k, v);
    }
    return out;
  }

  /** Fetch /html/<id> for full paper rendering. Returns ok:false on 404. */
  async fetchPaperHtml(
    arxivId: string,
  ): Promise<{ ok: true; body: string } | { ok: false; status: number }> {
    const url = `https://arxiv.org/html/${arxivId}`;
    try {
      const body = await this.fetchHtml(url, { allow404: true });
      return { ok: true, body };
    } catch (err: any) {
      if (err?.status === 404) return { ok: false, status: 404 };
      throw err;
    }
  }

  async fetchPaperAbsPage(arxivId: string): Promise<string> {
    const url = `https://arxiv.org/abs/${arxivId}`;
    return this.fetchHtml(url, { allow404: false });
  }

  /** Fetch the raw Atom XML for a single id (for manual lookup with full metadata). */
  async fetchAtomEntry(arxivId: string): Promise<string> {
    const url = `https://export.arxiv.org/api/query?id_list=${arxivId}&max_results=1`;
    return this.fetchHtml(url, { allow404: false });
  }

  private async fetchHtml(url: string, opts: { allow404: boolean }): Promise<string> {
    await this.respectDelay();
    return retry(
      async () => {
        const res = await requestUrl({
          url,
          method: "GET",
          headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
          throw: false,
        });
        if (res.status >= 200 && res.status < 300) return res.text;
        if (opts.allow404 && res.status === 404) {
          const e: any = new Error(`HTTP 404: ${url}`);
          e.status = 404;
          throw e;
        }
        throw new Error(`HTTP ${res.status}: ${url}`);
      },
      {
        maxAttempts: 3,
        baseDelayMs: 2000,
        shouldRetry: (err: any) => err?.status !== 404,
        onRetry: (err, attempt, wait) =>
          this.opts.logger.warn(
            `fetch retry #${attempt} after ${wait}ms: ${url}: ${(err as Error).message}`,
          ),
      },
    );
  }

  private async respectDelay() {
    const elapsed = Date.now() - this.lastRequestAt;
    const wait = this.opts.requestDelayMs - elapsed;
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    this.lastRequestAt = Date.now();
  }
}
