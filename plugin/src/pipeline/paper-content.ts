import type { ArxivFetcher } from "./arxiv-fetcher";
import type { HtmlCache } from "./html-cache";
import type { Logger } from "../services/logger";
import {
  extractAbstractConclusion,
  extractSections,
  type ExtractSectionsOpts,
} from "./section-extractor";

export interface PaperContent {
  abstractConclusion: string;
  fullSections: string | null;
}

export interface PaperContentOpts {
  isDetail: boolean;
  sectionCharLimit: number;
  paperCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
}

export class PaperContentFetcher {
  constructor(
    private fetcher: ArxivFetcher,
    private cache: HtmlCache,
    private logger: Logger,
  ) {}

  async fetch(arxivId: string, opts: PaperContentOpts): Promise<PaperContent> {
    // 1. Try the rendered HTML version (cached on hit)
    const htmlKey = `html/${arxivId}`;
    let html = await this.cache.get(htmlKey, "html");
    if (!html) {
      const res = await this.fetcher.fetchPaperHtml(arxivId);
      if (res.ok) {
        html = res.body;
        await this.cache.set(htmlKey, "html", html);
      }
    }

    if (html) {
      const ac = extractAbstractConclusion(html, {
        sectionCharLimit: opts.sectionCharLimit,
      });
      const sectionsOpts: ExtractSectionsOpts = {
        sectionCharLimit: opts.sectionCharLimit,
        paperCharLimit: opts.paperCharLimit,
        skipSections: opts.skipSections,
        prioritySections: opts.prioritySections,
      };
      const fs = opts.isDetail ? extractSections(html, sectionsOpts) : null;
      if (ac) return { abstractConclusion: ac, fullSections: fs };
      // Fallback: strip tags from full HTML if section extraction missed
      const plain = html
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .slice(0, opts.paperCharLimit);
      return { abstractConclusion: plain, fullSections: fs };
    }

    // 2. Fallback to /abs page (cached separately)
    const absKey = `abs/${arxivId}`;
    let abs = await this.cache.get(absKey, "abs");
    if (!abs) {
      try {
        abs = await this.fetcher.fetchPaperAbsPage(arxivId);
        await this.cache.set(absKey, "abs", abs);
      } catch (e) {
        this.logger.error(`paper-content: abs fetch failed ${arxivId}`, e);
        return {
          abstractConclusion: `[获取失败] arXiv ID: ${arxivId}`,
          fullSections: null,
        };
      }
    }
    const doc = new DOMParser().parseFromString(abs, "text/html");
    const bq = doc.querySelector("blockquote.abstract");
    const text =
      (bq?.textContent ?? "").replace(/^\s*Abstract:?\s*/, "").trim() || "N/A";
    return { abstractConclusion: `## Abstract\n${text}`, fullSections: null };
  }
}
