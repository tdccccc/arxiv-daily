import { normalizePath, type Vault } from "obsidian";
import type { OutputSettings } from "../settings/types";

export const PAPER_INBOX_SCHEMA_VERSION = 1;

export type PaperStatus =
  | "inbox"
  | "to_read"
  | "reading"
  | "read"
  | "saved"
  | "ignored";

export type PaperPriority = "low" | "normal" | "high";

export interface PaperInbox {
  schemaVersion: 1;
  updatedAt: string;
  papers: Record<string, PaperIndexEntry>;
}

export interface PaperIndexEntry {
  arxivId: string;
  source: "arxiv";
  title: string;
  authors: string[];
  published: string;
  updated: string;
  category: string;
  topics: string[];
  primaryTopic: string;
  detail: boolean;
  status: PaperStatus;
  priority: PaperPriority;
  seenDates: string[];
  dailyReports: string[];
  paperPath: string | null;
  arxivUrl: string;
  pdfUrl: string;
  pdfPath: string;
  zoteroKey: string;
  citationKey: string;
  projects: string[];
}

export interface PaperInboxPaths {
  rootDir: string;
  indexDir: string;
  papersJsonPath: string;
  legacyIndexDir: string;
  legacyPapersJsonPath: string;
}

export interface PaperIndexUpsert {
  arxivId: string;
  title: string;
  authors: string | string[];
  date: string;
  arxivCategory: string;
  primaryTopic: string;
  detail: boolean;
  dailyReport?: string;
  paperPath?: string | null;
}

export class PaperIndexError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "PaperIndexError";
  }
}

export function derivePaperInboxPaths(output: OutputSettings): PaperInboxPaths {
  const dailyParent = parentDir(output.dailyDir);
  const papersParent = parentDir(output.papersDir);
  const root =
    dailyParent && dailyParent === papersParent
      ? dailyParent
      : dailyParent || papersParent || "arxiv-daily";
  const rootDir = normalizePath(root);
  const indexDir = normalizePath(`${rootDir}/.index`);
  const legacyIndexDir = normalizePath(`${rootDir}/index`);
  return {
    rootDir,
    indexDir,
    papersJsonPath: normalizePath(`${indexDir}/papers.json`),
    legacyIndexDir,
    legacyPapersJsonPath: normalizePath(`${legacyIndexDir}/papers.json`),
  };
}

export class PaperIndexStore {
  readonly paths: PaperInboxPaths;

  constructor(
    private vault: Vault,
    output: OutputSettings,
    private now: () => Date = () => new Date(),
  ) {
    this.paths = derivePaperInboxPaths(output);
  }

  async load(): Promise<PaperInbox> {
    const path = await this.readableIndexPath();
    if (!path) {
      return emptyInbox(this.now());
    }

    let raw: string;
    try {
      raw = await this.vault.adapter.read(path);
    } catch (e) {
      throw new PaperIndexError(
        `failed to read paper index: ${path}`,
        e,
      );
    }

    try {
      return normalizeInbox(JSON.parse(raw), this.now());
    } catch (e) {
      throw new PaperIndexError(
        `failed to parse paper index: ${path}`,
        e,
      );
    }
  }

  async save(inbox: PaperInbox): Promise<void> {
    const next: PaperInbox = {
      ...inbox,
      updatedAt: this.now().toISOString(),
      papers: { ...inbox.papers },
    };
    await this.ensureDirDeep(this.paths.indexDir);
    await this.writeAtomic(
      this.paths.papersJsonPath,
      `${JSON.stringify(next, null, 2)}\n`,
    );
    await this.removeLegacyIndexFile();
  }

  async upsertFromDailyPaper(input: PaperIndexUpsert): Promise<{
    entry: PaperIndexEntry;
    wasNew: boolean;
  }> {
    const inbox = await this.load();
    const { entry, wasNew } = upsertEntry(inbox, input);
    await this.save(inbox);
    return { entry, wasNew };
  }

  async upsertManyFromDailyPapers(
    inputs: PaperIndexUpsert[],
  ): Promise<Array<{ entry: PaperIndexEntry; wasNew: boolean }>> {
    const inbox = await this.load();
    const results = inputs.map((input) => upsertEntry(inbox, input));
    await this.save(inbox);
    return results;
  }

  async addDailyReports(arxivIds: string[], dailyReport: string): Promise<void> {
    const inbox = await this.load();
    for (const arxivId of arxivIds) {
      const entry = inbox.papers[arxivId];
      if (!entry) continue;
      entry.dailyReports = appendUnique(entry.dailyReports, dailyReport);
    }
    await this.save(inbox);
  }

  async setStatus(arxivId: string, status: PaperStatus): Promise<PaperIndexEntry | null> {
    const inbox = await this.load();
    const entry = inbox.papers[arxivId];
    if (!entry) return null;
    entry.status = status;
    await this.save(inbox);
    return entry;
  }

  async setPriority(
    arxivId: string,
    priority: PaperPriority,
  ): Promise<PaperIndexEntry | null> {
    const inbox = await this.load();
    const entry = inbox.papers[arxivId];
    if (!entry) return null;
    entry.priority = priority;
    await this.save(inbox);
    return entry;
  }

  async setCitationKey(
    arxivId: string,
    citationKey: string,
  ): Promise<PaperIndexEntry | null> {
    const inbox = await this.load();
    const entry = inbox.papers[arxivId];
    if (!entry) return null;
    entry.citationKey = citationKey;
    await this.save(inbox);
    return entry;
  }

  async setPaperPath(arxivId: string, paperPath: string): Promise<PaperIndexEntry | null> {
    const inbox = await this.load();
    const entry = inbox.papers[arxivId];
    if (!entry) return null;
    entry.paperPath = normalizePath(paperPath);
    await this.save(inbox);
    return entry;
  }

  async get(arxivId: string): Promise<PaperIndexEntry | null> {
    const inbox = await this.load();
    return inbox.papers[arxivId] ?? null;
  }

  async listByStatus(status: PaperStatus): Promise<PaperIndexEntry[]> {
    const inbox = await this.load();
    return Object.values(inbox.papers)
      .filter((entry) => entry.status === status)
      .sort(compareEntries);
  }

  private async ensureDirDeep(dir: string): Promise<void> {
    const parts = normalizePath(dir).split("/").filter(Boolean);
    let cur = "";
    for (const part of parts) {
      cur = cur ? `${cur}/${part}` : part;
      if (!(await this.vault.adapter.exists(cur))) {
        await this.vault.adapter.mkdir(cur);
      }
    }
  }

  private async readableIndexPath(): Promise<string | null> {
    if (await this.vault.adapter.exists(this.paths.papersJsonPath)) {
      return this.paths.papersJsonPath;
    }
    if (await this.vault.adapter.exists(this.paths.legacyPapersJsonPath)) {
      return this.paths.legacyPapersJsonPath;
    }
    return null;
  }

  private async removeLegacyIndexFile(): Promise<void> {
    if (
      this.paths.legacyPapersJsonPath === this.paths.papersJsonPath ||
      !(await this.vault.adapter.exists(this.paths.legacyPapersJsonPath))
    ) {
      return;
    }
    try {
      await this.vault.adapter.remove(this.paths.legacyPapersJsonPath);
    } catch {
      // The hidden index has already been written; a stale legacy file should
      // not make the main save operation fail.
    }
  }

  private async writeAtomic(path: string, content: string): Promise<void> {
    const tmp = `${path}.tmp`;
    const bak = `${path}.bak`;
    await this.vault.adapter.write(tmp, content);
    if (!(await this.vault.adapter.exists(path))) {
      await this.vault.adapter.rename(tmp, path);
      return;
    }

    if (await this.vault.adapter.exists(bak)) {
      await this.vault.adapter.remove(bak);
    }
    await this.vault.adapter.rename(path, bak);
    try {
      await this.vault.adapter.rename(tmp, path);
      await this.vault.adapter.remove(bak);
    } catch (e) {
      if (await this.vault.adapter.exists(bak)) {
        await this.vault.adapter.rename(bak, path);
      }
      throw new PaperIndexError(`failed to save paper index: ${path}`, e);
    }
  }
}

function emptyInbox(now: Date): PaperInbox {
  return {
    schemaVersion: PAPER_INBOX_SCHEMA_VERSION,
    updatedAt: now.toISOString(),
    papers: {},
  };
}

function upsertEntry(
  inbox: PaperInbox,
  input: PaperIndexUpsert,
): { entry: PaperIndexEntry; wasNew: boolean } {
  const arxivId = input.arxivId.trim();
  const existing = inbox.papers[arxivId];
  const wasNew = !existing;
  const authors = normalizeAuthors(input.authors);
  const topic = input.primaryTopic.trim();
  const paperPath =
    input.paperPath === undefined
      ? existing?.paperPath ?? null
      : input.paperPath
      ? normalizePath(input.paperPath)
      : null;

  const entry: PaperIndexEntry = {
    arxivId,
    source: "arxiv",
    title: input.title.trim() || existing?.title || arxivId,
    authors: authors.length ? authors : existing?.authors ?? [],
    published: existing?.published || input.date,
    updated: input.date,
    category: input.arxivCategory || existing?.category || "",
    topics: appendUnique(existing?.topics ?? [], topic),
    primaryTopic: topic || existing?.primaryTopic || "",
    detail: Boolean(existing?.detail || input.detail),
    status: existing?.status ?? "inbox",
    priority: existing?.priority ?? "normal",
    seenDates: appendUnique(existing?.seenDates ?? [], input.date),
    dailyReports: input.dailyReport
      ? appendUnique(existing?.dailyReports ?? [], input.dailyReport)
      : existing?.dailyReports ?? [],
    paperPath,
    arxivUrl: `https://arxiv.org/abs/${arxivId}`,
    pdfUrl: `https://arxiv.org/pdf/${arxivId}`,
    pdfPath: existing?.pdfPath ?? "",
    zoteroKey: existing?.zoteroKey ?? "",
    citationKey: existing?.citationKey ?? "",
    projects: existing?.projects ?? [],
  };
  inbox.papers[arxivId] = entry;
  return { entry, wasNew };
}

function normalizeInbox(raw: unknown, now: Date): PaperInbox {
  if (!raw || typeof raw !== "object") return emptyInbox(now);
  const obj = raw as any;
  if (obj.schemaVersion !== PAPER_INBOX_SCHEMA_VERSION) {
    throw new Error(`unsupported schemaVersion: ${obj.schemaVersion}`);
  }
  const papers: Record<string, PaperIndexEntry> = {};
  for (const [id, value] of Object.entries(obj.papers ?? {})) {
    papers[id] = normalizeEntry(id, value);
  }
  return {
    schemaVersion: PAPER_INBOX_SCHEMA_VERSION,
    updatedAt:
      typeof obj.updatedAt === "string" && obj.updatedAt
        ? obj.updatedAt
        : now.toISOString(),
    papers,
  };
}

function normalizeEntry(id: string, raw: unknown): PaperIndexEntry {
  const obj = (raw && typeof raw === "object" ? raw : {}) as any;
  const arxivId = stringOr(obj.arxivId, id);
  const status = isPaperStatus(obj.status) ? obj.status : "inbox";
  const priority = isPaperPriority(obj.priority) ? obj.priority : "normal";
  return {
    arxivId,
    source: "arxiv",
    title: stringOr(obj.title, arxivId),
    authors: normalizeAuthors(obj.authors),
    published: stringOr(obj.published, ""),
    updated: stringOr(obj.updated, ""),
    category: stringOr(obj.category, ""),
    topics: stringArray(obj.topics),
    primaryTopic: stringOr(obj.primaryTopic, ""),
    detail: Boolean(obj.detail),
    status,
    priority,
    seenDates: stringArray(obj.seenDates),
    dailyReports: stringArray(obj.dailyReports),
    paperPath: obj.paperPath ? normalizePath(String(obj.paperPath)) : null,
    arxivUrl: stringOr(obj.arxivUrl, `https://arxiv.org/abs/${arxivId}`),
    pdfUrl: stringOr(obj.pdfUrl, `https://arxiv.org/pdf/${arxivId}`),
    pdfPath: stringOr(obj.pdfPath, ""),
    zoteroKey: stringOr(obj.zoteroKey, ""),
    citationKey: stringOr(obj.citationKey, ""),
    projects: stringArray(obj.projects),
  };
}

function parentDir(path: string): string {
  const parts = normalizePath(path).split("/").filter(Boolean);
  if (parts.length <= 1) return "";
  return parts.slice(0, -1).join("/");
}

function appendUnique(items: string[], next: string): string[] {
  const out = [...items];
  const trimmed = next.trim();
  if (trimmed && !out.includes(trimmed)) out.push(trimmed);
  return out;
}

function normalizeAuthors(value: string | string[] | unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((v) => String(v).trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/\s*,\s*/)
      .map((v) => v.trim())
      .filter(Boolean);
  }
  return [];
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => String(v).trim()).filter(Boolean);
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

export function isPaperStatus(value: unknown): value is PaperStatus {
  return (
    value === "inbox" ||
    value === "to_read" ||
    value === "reading" ||
    value === "read" ||
    value === "saved" ||
    value === "ignored"
  );
}

export function isPaperPriority(value: unknown): value is PaperPriority {
  return value === "low" || value === "normal" || value === "high";
}

function compareEntries(a: PaperIndexEntry, b: PaperIndexEntry): number {
  if (a.primaryTopic !== b.primaryTopic) {
    return a.primaryTopic.localeCompare(b.primaryTopic);
  }
  const priorityOrder: Record<PaperPriority, number> = {
    high: 0,
    normal: 1,
    low: 2,
  };
  if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
    return priorityOrder[a.priority] - priorityOrder[b.priority];
  }
  return b.published.localeCompare(a.published);
}
