import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import { normalizeArxivId } from "./manual-fetch";
import {
  derivePaperInboxPaths,
  type PaperIndexEntry,
  type PaperIndexStore,
} from "./paper-index";
import type { Logger } from "./logger";
import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";

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

export type CitationSnippetFormat = "latex" | "pandoc" | "typst";

export type CitationSnippetResult =
  | {
      kind: "done";
      arxivId: string;
      citationKey: string;
      snippet: string;
      format: CitationSnippetFormat;
      entryUpdated: boolean;
      source: "index" | "bibtex";
    }
  | { kind: "invalid_id"; reason: string }
  | { kind: "fetch_error"; reason: string }
  | { kind: "invalid_bibtex"; reason: string };

export interface BibtexServiceDeps {
  fetcher: Pick<ArxivFetcher, "fetchBibtex">;
  paperIndex?: PaperIndexStore;
  storage?: StorageAdapter;
  output?: OutputSettings;
  logger: Logger;
  now?: () => Date;
}

export interface BibtexExportItem {
  arxivId: string;
  citationKey: string;
  sourceCitationKey: string;
  bibtex: string;
  entryUpdated: boolean;
  keyRenamed: boolean;
}

export interface BibtexExportFailure {
  arxivId: string;
  reason: string;
}

export type BibtexExportResult =
  | {
      kind: "done";
      path: string;
      requested: number;
      exported: number;
      items: BibtexExportItem[];
      failures: BibtexExportFailure[];
      keysRenamed: number;
    }
  | {
      kind: "empty";
      requested: number;
      failures: BibtexExportFailure[];
    };

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

  async citationSnippetForId(
    rawId: string,
    format: CitationSnippetFormat,
  ): Promise<CitationSnippetResult> {
    const arxivId = normalizeArxivId(rawId);
    if (!arxivId) {
      return { kind: "invalid_id", reason: `invalid arXiv id: ${rawId}` };
    }

    if (this.deps.paperIndex) {
      const entry = await this.deps.paperIndex.get(arxivId);
      const citationKey = entry?.citationKey.trim();
      if (citationKey) {
        return {
          kind: "done",
          arxivId,
          citationKey,
          snippet: formatCitationSnippet(citationKey, format),
          format,
          entryUpdated: false,
          source: "index",
        };
      }
    }

    const result = await this.fetchAndStore(arxivId);
    if (result.kind !== "done") return result;
    return {
      kind: "done",
      arxivId: result.arxivId,
      citationKey: result.citationKey,
      snippet: formatCitationSnippet(result.citationKey, format),
      format,
      entryUpdated: result.entryUpdated,
      source: "bibtex",
    };
  }

  async exportManyToFile(
    entries: Array<Pick<PaperIndexEntry, "arxivId">>,
    opts: { path?: string } = {},
  ): Promise<BibtexExportResult> {
    if (!this.deps.storage || !this.deps.output) {
      throw new Error("BibTeX export requires storage and output settings");
    }

    const ids = uniqueArxivIds(entries.map((entry) => entry.arxivId));
    const rawItems: Array<{
      arxivId: string;
      sourceCitationKey: string;
      bibtex: string;
      entryUpdated: boolean;
    }> = [];
    const failures: BibtexExportFailure[] = [];

    for (const arxivId of ids) {
      const result = await this.fetchAndStore(arxivId);
      if (result.kind === "done") {
        rawItems.push({
          arxivId: result.arxivId,
          sourceCitationKey: result.citationKey,
          bibtex: result.bibtex,
          entryUpdated: result.entryUpdated,
        });
      } else {
        failures.push({
          arxivId,
          reason: result.reason,
        });
      }
    }

    if (rawItems.length === 0) {
      return { kind: "empty", requested: ids.length, failures };
    }

    const items = await this.dedupeExportKeys(rawItems);
    const path = this.deps.storage.normalizePath(
      opts.path ?? defaultBibtexExportPath(this.deps.output, this.now()),
    );
    await ensureDirDeep(this.deps.storage, parentDir(path));
    await this.deps.storage.writeText(path, renderBibtexExport(items, failures));

    return {
      kind: "done",
      path,
      requested: ids.length,
      exported: items.length,
      items,
      failures,
      keysRenamed: items.filter((item) => item.keyRenamed).length,
    };
  }

  private async dedupeExportKeys(
    rawItems: Array<{
      arxivId: string;
      sourceCitationKey: string;
      bibtex: string;
      entryUpdated: boolean;
    }>,
  ): Promise<BibtexExportItem[]> {
    const used = new Set<string>();
    const items: BibtexExportItem[] = [];
    for (const item of rawItems) {
      const citationKey = uniqueBibtexKey(
        item.sourceCitationKey,
        item.arxivId,
        used,
      );
      used.add(citationKey);
      const keyRenamed = citationKey !== item.sourceCitationKey;
      let entryUpdated = item.entryUpdated;
      if (keyRenamed && this.deps.paperIndex) {
        try {
          entryUpdated = Boolean(
            await this.deps.paperIndex.setCitationKey(item.arxivId, citationKey),
          );
        } catch (e) {
          this.deps.logger.warn(
            `bibtex: exported ${item.arxivId} but failed to update deduped citationKey: ${(e as Error).message}`,
          );
        }
      }
      items.push({
        arxivId: item.arxivId,
        citationKey,
        sourceCitationKey: item.sourceCitationKey,
        bibtex: rewriteBibtexKey(item.bibtex, citationKey),
        entryUpdated,
        keyRenamed,
      });
    }
    return items;
  }

  private now(): Date {
    return this.deps.now?.() ?? new Date();
  }
}

export function parseBibtexKey(bibtex: string): string | null {
  const match = /@\w+\s*\{\s*([^,\s]+)\s*,/m.exec(bibtex);
  return match?.[1]?.trim() || null;
}

export function formatCitationSnippet(
  citationKeys: string | string[],
  format: CitationSnippetFormat,
): string {
  const keys = (Array.isArray(citationKeys) ? citationKeys : [citationKeys])
    .map((key) => key.trim())
    .filter(Boolean);
  if (keys.length === 0) return "";

  switch (format) {
    case "latex":
      return `\\cite{${keys.join(",")}}`;
    case "pandoc":
      return `[${keys.map((key) => `@${key}`).join("; ")}]`;
    case "typst":
      return keys.map((key) => `@${key}`).join(" ");
  }
}

export function rewriteBibtexKey(bibtex: string, citationKey: string): string {
  return bibtex.replace(
    /(@\w+\s*\{\s*)([^,\s]+)(\s*,)/m,
    `$1${citationKey}$3`,
  );
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

function uniqueArxivIds(values: string[]): string[] {
  const out: string[] = [];
  for (const value of values) {
    const id = normalizeArxivId(value);
    if (id && !out.includes(id)) out.push(id);
  }
  return out;
}

function uniqueBibtexKey(
  sourceCitationKey: string,
  arxivId: string,
  used: Set<string>,
): string {
  if (!used.has(sourceCitationKey)) return sourceCitationKey;
  const suffix = arxivId.replace(/[^A-Za-z0-9]+/g, "");
  let candidate = `${sourceCitationKey}_${suffix}`;
  let i = 2;
  while (used.has(candidate)) {
    candidate = `${sourceCitationKey}_${suffix}_${i}`;
    i += 1;
  }
  return candidate;
}

function renderBibtexExport(
  items: BibtexExportItem[],
  failures: BibtexExportFailure[],
): string {
  const header = [
    "% Generated by arXiv Daily",
    `% Entries: ${items.length}`,
    failures.length > 0 ? `% Failed: ${failures.length}` : "",
    ...failures.map((failure) => `% Failed ${failure.arxivId}: ${failure.reason}`),
  ].filter(Boolean);
  return `${header.join("\n")}\n\n${items
    .map((item) => item.bibtex.trimEnd())
    .join("\n\n")}\n`;
}

function defaultBibtexExportPath(output: OutputSettings, now: Date): string {
  const root = derivePaperInboxPaths(output).rootDir || "arxiv-daily";
  return `${root}/exports/arxiv-daily-${formatExportTimestamp(now)}.bib`;
}

function formatExportTimestamp(date: Date): string {
  return date
    .toISOString()
    .replace(/\.\d{3}Z$/, "Z")
    .replace(/:/g, "")
    .replace("T", "-");
}

function parentDir(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length <= 1 ? "" : parts.slice(0, -1).join("/");
}

async function ensureDirDeep(
  storage: StorageAdapter,
  dir: string,
): Promise<void> {
  if (!dir) return;
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur = cur ? `${cur}/${part}` : part;
    if (!(await storage.exists(cur))) {
      await storage.mkdir(cur);
    }
  }
}
