import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import { normalizeArxivId } from "./manual-fetch";
import type { PaperIndexStore } from "./paper-index";
import type { Logger } from "./logger";

export type BibtexResult =
  | {
      kind: "done";
      arxivId: string;
      bibtex: string;
      citationKey: string;
      entryUpdated: boolean;
    }
  | { kind: "invalid_id"; reason: string }
  | { kind: "fetch_error"; reason: string }
  | { kind: "invalid_bibtex"; reason: string };

export interface BibtexServiceDeps {
  fetcher: Pick<ArxivFetcher, "fetchBibtex">;
  paperIndex?: PaperIndexStore;
  logger: Logger;
}

export class BibtexService {
  constructor(private deps: BibtexServiceDeps) {}

  async fetchAndStore(rawId: string): Promise<BibtexResult> {
    const arxivId = normalizeArxivId(rawId);
    if (!arxivId) {
      return { kind: "invalid_id", reason: `invalid arXiv id: ${rawId}` };
    }

    let bibtex: string;
    try {
      bibtex = await this.deps.fetcher.fetchBibtex(arxivId);
    } catch (e) {
      this.deps.logger.error(`bibtex: fetch failed for ${arxivId}`, e);
      return {
        kind: "fetch_error",
        reason: (e as Error).message,
      };
    }

    if (!bibtex.trim()) {
      return { kind: "invalid_bibtex", reason: "arXiv returned empty BibTeX" };
    }

    const citationKey = parseBibtexKey(bibtex);
    if (!citationKey) {
      return {
        kind: "invalid_bibtex",
        reason: "could not parse BibTeX citation key",
      };
    }

    let entryUpdated = false;
    if (this.deps.paperIndex) {
      try {
        const entry = await this.deps.paperIndex.setCitationKey(
          arxivId,
          citationKey,
        );
        entryUpdated = Boolean(entry);
      } catch (e) {
        this.deps.logger.warn(
          `bibtex: fetched ${arxivId} but failed to update citationKey: ${(e as Error).message}`,
        );
      }
    }

    return {
      kind: "done",
      arxivId,
      bibtex: bibtex.trimEnd(),
      citationKey,
      entryUpdated,
    };
  }
}

export function parseBibtexKey(bibtex: string): string | null {
  const match = /@\w+\s*\{\s*([^,\s]+)\s*,/m.exec(bibtex);
  return match?.[1]?.trim() || null;
}

export function extractArxivIdFromMarkdown(markdown: string): string | null {
  const frontmatter = /^---\s*\n([\s\S]*?)\n---/.exec(markdown)?.[1] ?? "";
  const frontmatterId = /^(?:arxiv_id|arxiv):\s*["']?([^"'\n]+)["']?\s*$/im.exec(
    frontmatter,
  )?.[1];
  if (frontmatterId) {
    const normalized = normalizeArxivId(frontmatterId);
    if (normalized) return normalized;
  }

  const linkedId =
    /(?:https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf|html)\/|arXiv:\s*)(\d{4}\.\d{4,5})(?:v\d+)?/i.exec(
      markdown,
    )?.[1] ?? "";
  if (linkedId) return normalizeArxivId(linkedId);

  const bareId = /\b(\d{4}\.\d{4,5})(?:v\d+)?\b/.exec(markdown)?.[1] ?? "";
  return bareId ? normalizeArxivId(bareId) : null;
}
