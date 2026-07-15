import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import {
  derivePaperInboxPaths,
  type PaperIndexEntry,
  type PaperIndexStore,
} from "./paper-index";
import { normalizeArxivId } from "./manual-fetch";
import type { Logger } from "./logger";

export type PdfDownloadResult =
  | {
      kind: "done";
      arxivId: string;
      path: string;
      bytes: number;
      entryUpdated: boolean;
    }
  | { kind: "invalid_id"; reason: string }
  | { kind: "missing_storage"; reason: string }
  | { kind: "fetch_error"; reason: string };

export interface PdfServiceDeps {
  fetcher: Pick<ArxivFetcher, "fetchPdf">;
  storage: StorageAdapter;
  paperIndex?: PaperIndexStore;
  output: OutputSettings;
  logger: Logger;
}

export class PdfService {
  constructor(private deps: PdfServiceDeps) {}

  async downloadForEntry(
    entry: Pick<PaperIndexEntry, "arxivId">,
  ): Promise<PdfDownloadResult> {
    return this.downloadById(entry.arxivId);
  }

  async downloadById(rawId: string): Promise<PdfDownloadResult> {
    const arxivId = normalizeArxivId(rawId);
    if (!arxivId) {
      return { kind: "invalid_id", reason: `invalid arXiv id: ${rawId}` };
    }
    if (!this.deps.storage.writeBinary) {
      return {
        kind: "missing_storage",
        reason: "storage adapter does not support binary writes",
      };
    }

    let pdf: ArrayBuffer;
    try {
      pdf = await this.deps.fetcher.fetchPdf(arxivId);
    } catch (e) {
      this.deps.logger.error(`pdf: fetch failed for ${arxivId}`, e);
      return { kind: "fetch_error", reason: (e as Error).message };
    }

    const path = this.deps.storage.normalizePath(
      `${derivePaperInboxPaths(this.deps.output).rootDir || "arxiv-daily"}/pdfs/${arxivId}.pdf`,
    );
    await ensureDirDeep(this.deps.storage, parentDir(path));
    await this.deps.storage.writeBinary(path, pdf);

    let entryUpdated = false;
    if (this.deps.paperIndex) {
      const updated = await this.deps.paperIndex.setPdfPath(arxivId, path);
      entryUpdated = Boolean(updated);
    }
    return {
      kind: "done",
      arxivId,
      path,
      bytes: pdf.byteLength,
      entryUpdated,
    };
  }
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
    if (!(await storage.exists(cur))) await storage.mkdir(cur);
  }
}
