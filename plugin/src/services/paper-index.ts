import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";

export const PAPER_INBOX_SCHEMA_VERSION = 2;

export type PaperStatus =
  | "inbox"
  | "to_read"
  | "reading"
  | "read"
  | "saved"
  | "ignored";

export type PaperPriority = "low" | "normal" | "high";

export interface PaperInbox {
  schemaVersion: number;
  updatedAt: string;
  papers: Record<string, PaperIndexEntry>;
}

export interface PaperSummary {
  sourceSections?: string;
  coreProblem?: string;
  keyMethod?: string;
  mainResult?: string;
  whyRelevant?: string;
  limitations?: string;
}

export interface PaperIndexEntry {
  arxivId: string;
  source: "arxiv";
  title: string;
  authors: string[];
  published: string;
  updated: string;
  category: string;
  categories?: string[];
  summary?: PaperSummary;
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
  zoteroUri: string;
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
  arxivCategories?: string[];
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

export function derivePaperInboxPaths(
  output: OutputSettings,
  normalizePath: (path: string) => string = normalizeStoragePath,
): PaperInboxPaths {
  const dailyParent = parentDir(output.dailyDir, normalizePath);
  const papersParent = parentDir(output.papersDir, normalizePath);
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
    private storage: StorageAdapter,
    output: OutputSettings,
    private now: () => Date = () => new Date(),
  ) {
    this.paths = derivePaperInboxPaths(output, (path) =>
      this.storage.normalizePath(path),
    );
  }

  async load(): Promise<PaperInbox> {
    const path = await this.readableIndexPath();
    if (!path) {
      return emptyInbox(this.now());
    }

    let raw: string;
    try {
      raw = await this.storage.readText(path);
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

  async setZoteroFields(
    arxivId: string,
    fields: { zoteroKey?: string; zoteroUri?: string },
  ): Promise<PaperIndexEntry | null> {
    const validation = validateZoteroFields(fields);
    if (!validation.ok) {
      throw new PaperIndexError(
        `invalid Zotero fields: ${validation.reasons.join("; ")}`,
      );
    }
    const inbox = await this.load();
    const entry = inbox.papers[arxivId];
    if (!entry) return null;
    if (fields.zoteroKey !== undefined) entry.zoteroKey = fields.zoteroKey.trim();
    if (fields.zoteroUri !== undefined) entry.zoteroUri = fields.zoteroUri.trim();
    await this.save(inbox);
    return entry;
  }

  async setSummaries(
    summaries: Record<string, PaperSummary>,
  ): Promise<number> {
    const inbox = await this.load();
    let changed = 0;
    for (const [arxivId, summary] of Object.entries(summaries)) {
      const entry = inbox.papers[arxivId];
      if (!entry) continue;
      entry.summary = normalizeSummary(summary);
      changed += 1;
    }
    if (changed > 0) await this.save(inbox);
    return changed;
  }

  async setPaperPath(arxivId: string, paperPath: string): Promise<PaperIndexEntry | null> {
    const inbox = await this.load();
    const entry = inbox.papers[arxivId];
    if (!entry) return null;
    entry.paperPath = this.storage.normalizePath(paperPath);
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
    const parts = this.storage.normalizePath(dir).split("/").filter(Boolean);
    let cur = "";
    for (const part of parts) {
      cur = cur ? `${cur}/${part}` : part;
      if (!(await this.storage.exists(cur))) {
        await this.storage.mkdir(cur);
      }
    }
  }

  private async readableIndexPath(): Promise<string | null> {
    if (await this.storage.exists(this.paths.papersJsonPath)) {
      return this.paths.papersJsonPath;
    }
    if (await this.storage.exists(this.paths.legacyPapersJsonPath)) {
      return this.paths.legacyPapersJsonPath;
    }
    return null;
  }

  private async removeLegacyIndexFile(): Promise<void> {
    if (
      this.paths.legacyPapersJsonPath === this.paths.papersJsonPath ||
      !(await this.storage.exists(this.paths.legacyPapersJsonPath))
    ) {
      return;
    }
    try {
      await this.storage.remove(this.paths.legacyPapersJsonPath);
    } catch {
      // The hidden index has already been written; a stale legacy file should
      // not make the main save operation fail.
    }
  }

  private async writeAtomic(path: string, content: string): Promise<void> {
    const tmp = `${path}.tmp`;
    const bak = `${path}.bak`;
    await this.storage.writeText(tmp, content);
    if (!(await this.storage.exists(path))) {
      await this.storage.rename(tmp, path);
      return;
    }

    if (await this.storage.exists(bak)) {
      await this.storage.remove(bak);
    }
    await this.storage.rename(path, bak);
    try {
      await this.storage.rename(tmp, path);
      await this.storage.remove(bak);
    } catch (e) {
      if (await this.storage.exists(bak)) {
        await this.storage.rename(bak, path);
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
  const inputCategories = normalizeCategories([
    ...(input.arxivCategories ?? []),
    input.arxivCategory,
  ]);
  const existingCategories = normalizeCategories([
    ...(existing?.categories ?? []),
    existing?.category ?? "",
  ]);
  const categories = appendUniqueMany(existingCategories, inputCategories);
  const paperPath =
    input.paperPath === undefined
      ? existing?.paperPath ?? null
      : input.paperPath
        ? normalizeStoragePath(input.paperPath)
        : null;

  const entry: PaperIndexEntry = {
    arxivId,
    source: "arxiv",
    title: input.title.trim() || existing?.title || arxivId,
    authors: authors.length ? authors : existing?.authors ?? [],
    published: existing?.published || input.date,
    updated: input.date,
    category: inputCategories[0] || existing?.category || categories[0] || "",
    categories,
    summary: existing?.summary,
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
    zoteroUri: existing?.zoteroUri ?? "",
    citationKey: existing?.citationKey ?? "",
    projects: existing?.projects ?? [],
  };
  inbox.papers[arxivId] = entry;
  return { entry, wasNew };
}

function normalizeInbox(raw: unknown, now: Date): PaperInbox {
  if (!raw || typeof raw !== "object") return emptyInbox(now);
  const obj = raw as any;
  if (obj.schemaVersion !== 1 && obj.schemaVersion !== 2) {
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
    categories: normalizeCategories([
      ...stringArray(obj.categories),
      stringOr(obj.category, ""),
    ]),
    summary: normalizeSummary(obj.summary),
    topics: stringArray(obj.topics),
    primaryTopic: stringOr(obj.primaryTopic, ""),
    detail: Boolean(obj.detail),
    status,
    priority,
    seenDates: stringArray(obj.seenDates),
    dailyReports: stringArray(obj.dailyReports),
    paperPath: obj.paperPath ? normalizeStoragePath(String(obj.paperPath)) : null,
    arxivUrl: stringOr(obj.arxivUrl, `https://arxiv.org/abs/${arxivId}`),
    pdfUrl: stringOr(obj.pdfUrl, `https://arxiv.org/pdf/${arxivId}`),
    pdfPath: stringOr(obj.pdfPath, ""),
    zoteroKey: stringOr(obj.zoteroKey, ""),
    zoteroUri: stringOr(obj.zoteroUri, ""),
    citationKey: stringOr(obj.citationKey, ""),
    projects: stringArray(obj.projects),
  };
}

export function validateZoteroFields(fields: {
  zoteroKey?: string;
  zoteroUri?: string;
}): { ok: true; reasons: [] } | { ok: false; reasons: string[] } {
  const reasons: string[] = [];
  const zoteroKey = fields.zoteroKey?.trim() ?? "";
  const zoteroUri = fields.zoteroUri?.trim() ?? "";
  if (zoteroKey && /\s/.test(zoteroKey)) {
    reasons.push("zoteroKey must not contain whitespace");
  }
  if (zoteroUri && !isValidZoteroUri(zoteroUri)) {
    reasons.push("zoteroUri must be a zotero:// or http(s) URL");
  }
  return reasons.length === 0 ? { ok: true, reasons: [] } : { ok: false, reasons };
}

function isValidZoteroUri(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === "zotero:" ||
      url.protocol === "http:" ||
      url.protocol === "https:"
    );
  } catch {
    return false;
  }
}

function parentDir(
  path: string,
  normalizePath: (path: string) => string = normalizeStoragePath,
): string {
  const parts = normalizePath(path).split("/").filter(Boolean);
  if (parts.length <= 1) return "";
  return parts.slice(0, -1).join("/");
}

function normalizeStoragePath(path: string): string {
  return path
    .replace(/\\/g, "/")
    .replace(/\/+/g, "/")
    .replace(/^\/+|\/+$/g, "");
}

function appendUnique(items: string[], next: string): string[] {
  const out = [...items];
  const trimmed = next.trim();
  if (trimmed && !out.includes(trimmed)) out.push(trimmed);
  return out;
}

function appendUniqueMany(items: string[], values: string[]): string[] {
  let out = items;
  for (const value of values) out = appendUnique(out, value);
  return out;
}

function normalizeCategories(values: string[]): string[] {
  const out: string[] = [];
  for (const value of values) {
    const category = value.trim();
    if (category && !out.includes(category)) out.push(category);
  }
  return out;
}

function normalizeSummary(value: unknown): PaperSummary | undefined {
  if (!value || typeof value !== "object") return undefined;
  const obj = value as Record<string, unknown>;
  const summary: PaperSummary = {
    sourceSections: stringOr(obj.sourceSections, ""),
    coreProblem: stringOr(obj.coreProblem, ""),
    keyMethod: stringOr(obj.keyMethod, ""),
    mainResult: stringOr(obj.mainResult, ""),
    whyRelevant: stringOr(obj.whyRelevant, ""),
    limitations: stringOr(obj.limitations, ""),
  };
  for (const key of Object.keys(summary) as Array<keyof PaperSummary>) {
    if (!summary[key]) delete summary[key];
  }
  return Object.keys(summary).length > 0 ? summary : undefined;
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
