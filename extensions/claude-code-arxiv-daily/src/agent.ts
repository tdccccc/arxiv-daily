import * as fs from "node:fs/promises";
import * as path from "node:path";
import { createHash } from "node:crypto";
import { parse as parseYaml } from "yaml";
import {
  ARXIV_CATEGORIES, ArxivFetcher, HtmlCache, Logger, PaperContentFetcher,
  modernArxivResources, parseRecent, type HttpClient,
} from "@arxiv-daily/core";
import { LinkedomMarkupParser, NodeHttpClient, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { openScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";

const OUTPUT = "arxiv-daily-agent";
const CONFIG = ".workspace.json";
const MAX_RECORD_BYTES = 256 * 1024;
const KINDS = ["direction", "paper", "reading", "daily"] as const;
type Kind = typeof KINDS[number];
type Input = Record<string, unknown>;
interface Config {
  version: 1;
  library: string;
  rootIdentity: string;
  includeMarkdown: boolean;
}
interface RecordMeta {
  arxiv_daily_agent: 1;
  kind: Kind;
  slug: string;
  title: string;
  status: "draft" | "confirmed" | "saved";
  sources: string[];
}
export interface AgentOptions { http?: HttpClient; signal?: AbortSignal }

/** Structured, host-independent command boundary; the CLI only handles JSON I/O. */
export async function executeAgentCommand(
  command: string,
  workspace: string,
  input: Input = {},
  options: AgentOptions = {},
): Promise<Record<string, unknown>> {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Input must be a JSON object");
  const workspacePath = path.resolve(workspace);
  const root = path.join(workspacePath, OUTPUT);
  if (command === "init") return initialize(root, input);
  const config = await loadConfig(root);
  if (command === "status") return status(root, config);
  if (command === "library") return library(config, input, options.signal);
  if (command === "read") return readRecord(root, kindOf(input.kind), slugOf(input.slug));
  if (command === "save") return saveRecord(root, input);
  if (command === "confirm-direction") return confirmDirection(root, input);
  if (command === "recent") return recentPapers(input, options);
  if (command === "paper") return paperContent(root, input, options);
  throw new Error(`Unknown command: ${command}`);
}

function arxivServices(options: AgentOptions, category = "cs.AI") {
  const logger = new Logger("warn");
  const markupParser = new LinkedomMarkupParser();
  const fetcher = new ArxivFetcher({
    category, http: options.http ?? new NodeHttpClient(), markupParser, logger,
    requestDelayMs: 3000, textTimeoutMs: 30_000,
  });
  return { logger, markupParser, fetcher };
}

async function recentPapers(input: Input, options: AgentOptions): Promise<Record<string, unknown>> {
  const category = requiredString(input.category, "category", 64);
  if (!ARXIV_CATEGORIES.some(group => group.categories.some(item => item.id === category))) throw new Error("Unknown arXiv category");
  const requestedDate = optionalString(input.date, "date", 10);
  if (requestedDate && (!/^\d{4}-\d{2}-\d{2}$/.test(requestedDate)
    || !Number.isFinite(Date.parse(requestedDate)) || new Date(requestedDate).toISOString().slice(0, 10) !== requestedDate)) {
    throw new Error("Invalid date: expected a real YYYY-MM-DD date");
  }
  const { offset, limit } = pagination(input);
  const { fetcher, markupParser } = arxivServices(options, category);
  const buckets = parseRecent(await fetcher.fetchRecent(category, options.signal), markupParser);
  const date = requestedDate || buckets[0]?.announceDate || null;
  const selected = buckets.find(bucket => bucket.announceDate === date);
  const papers = selected?.papers ?? [];
  return {
    category, date, availableDates: buckets.map(bucket => bucket.announceDate),
    state: selected ? "ready" : requestedDate ? "date-unavailable" : "listing-unavailable",
    total: papers.length,
    items: papers.slice(offset, offset + limit).map(paper => ({
      ...paper, paperKey: `arxiv:${modernArxivResources(paper.id)!.id}`,
      evidenceDepth: paper.abstract ? "metadata-and-abstract" : "listing-metadata",
    })),
    nextOffset: offset + limit < papers.length ? offset + limit : null,
  };
}

async function paperContent(root: string, input: Input, options: AgentOptions): Promise<Record<string, unknown>> {
  const resource = modernArxivResources(requiredString(input.id, "arxiv id", 2048));
  if (!resource) throw new Error("Invalid arXiv ID or URL");
  const { fetcher, markupParser, logger } = arxivServices(options);
  const metadata = (await fetcher.fetchMetadataByIds([resource.id], options.signal)).get(resource.id);
  if (!metadata) throw new Error(`arXiv returned no metadata for ${resource.id}`);
  let fullText: string | null = null;
  let fullTextSource: string | null = null;
  let fullTextFailure: string | null = null;
  if (input.fullText === true) {
    await ensureDirectory(path.join(root, ".cache"), true);
    const cache = new HtmlCache({ storage: storageFor(root), rootDir: ".cache", expiryDays: 7 });
    const content = await new PaperContentFetcher(fetcher, cache, logger, markupParser).fetch(resource.id, {
      isDetail: true, sectionCharLimit: 12_000, paperCharLimit: 60_000,
    }, options.signal);
    fullText = content.fullSections;
    fullTextSource = content.fullTextSource ?? null;
    fullTextFailure = fullText ? null : content.fullTextFailure ?? "Usable full-text sections are unavailable; use the abstract or local PDF";
  }
  return {
    paperKey: `arxiv:${resource.id}`, metadata, ...resource,
    evidenceDepth: fullText ? "extracted-sections" : "metadata-and-abstract",
    fullText, fullTextSource, fullTextFailure,
    extractionLimits: input.fullText === true ? { sectionCharacters: 12_000, paperCharacters: 60_000 } : null,
  };
}

async function initialize(root: string, input: Input): Promise<Record<string, unknown>> {
  const selected = requiredString(input.library, "library", 8192);
  if (!path.isAbsolute(selected)) throw new Error("library must be an absolute directory path");
  const source = await openScopedLibrarySource(selected);
  if (contains(source.canonicalRoot, root) || contains(root, source.canonicalRoot)) {
    throw new Error("Research output and the source library must not overlap; select a separate workspace");
  }
  await ensureDirectory(root, true);
  const storage = storageFor(root);
  const lock = await storage.acquireLock("agent-records", { wait: true });
  if (!lock) throw new Error("Research workspace is busy");
  try {
    const existing = await readOptional(root, CONFIG);
    const candidate: Config = {
      version: 1, library: source.canonicalRoot, rootIdentity: source.rootIdentity,
      includeMarkdown: input.includeMarkdown === true,
    };
    if (existing !== null) {
      const previous = decodeConfig(JSON.parse(existing));
      if (JSON.stringify(previous) !== JSON.stringify(candidate)) {
        throw new Error("Workspace is already connected; use a new workspace to change library or file scope");
      }
    } else {
      await storage.writeTextAtomic(CONFIG, `${JSON.stringify(candidate, null, 2)}\n`, 0o600);
    }
    for (const kind of KINDS) await ensureDirectory(path.join(root, folder(kind)), true);
    return { workspace: path.dirname(root), output: root, library: candidate.library, includeMarkdown: candidate.includeMarkdown };
  } finally { await lock.release(); }
}

async function loadConfig(root: string): Promise<Config> {
  const raw = await readOptional(root, CONFIG);
  if (raw === null) throw new Error("Workspace is not connected. Run init with an absolute library directory first");
  const config = decodeConfig(JSON.parse(raw));
  if (contains(config.library, root) || contains(root, config.library)) throw new Error("Stored source library overlaps research output");
  return config;
}

function decodeConfig(value: unknown): Config {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid workspace config");
  const v = value as Input;
  if (v.version !== 1 || typeof v.library !== "string" || !path.isAbsolute(v.library)
    || typeof v.rootIdentity !== "string" || !v.rootIdentity || typeof v.includeMarkdown !== "boolean") {
    throw new Error("Invalid workspace config; preserve the file and reconnect in a new workspace");
  }
  return { version: 1, library: v.library, rootIdentity: v.rootIdentity, includeMarkdown: v.includeMarkdown };
}

async function library(config: Config, input: Input, signal?: AbortSignal): Promise<Record<string, unknown>> {
  const source = await openScopedLibrarySource(config.library);
  if (source.rootIdentity !== config.rootIdentity || source.canonicalRoot !== config.library) throw new Error("Source library root changed");
  const inventory = await source.inventory({ maxEntries: 10_000, signal });
  const query = optionalString(input.query, "query", 1024).toLocaleLowerCase();
  const { offset, limit } = pagination(input);
  const eligible = inventory.entries.filter(entry => entry.type === "file"
    && (/\.pdf$/i.test(entry.path) || (config.includeMarkdown && /\.md$/i.test(entry.path))));
  const matched = eligible.filter(entry => entry.path.toLocaleLowerCase().includes(query))
    .sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  return {
    library: config.library, searchScope: "filename-only", total: matched.length,
    inventoryTruncated: inventory.truncated,
    ignored: inventory.entries.filter(entry => entry.type !== "folder").length - eligible.length,
    items: matched.slice(offset, offset + limit).map(entry => ({
      relativePath: entry.path, absolutePath: path.join(config.library, entry.path),
      bytes: entry.size, paperKey: keyFromFilename(entry.path),
      evidenceDepth: "filename-only",
    })),
    nextOffset: offset + limit < matched.length ? offset + limit : null,
  };
}

async function status(root: string, config: Config): Promise<Record<string, unknown>> {
  const records: Record<Kind, Input[]> = { direction: [], paper: [], reading: [], daily: [] };
  const unreadable: string[] = [];
  const truncatedKinds: Kind[] = [];
  for (const kind of KINDS) {
    const dir = path.join(root, folder(kind));
    await ensureDirectory(dir, false);
    const names = (await fs.readdir(dir)).filter(n => n.endsWith(".md")).sort();
    if (names.length > 500) truncatedKinds.push(kind);
    for (const name of names.slice(0, 500)) {
      try {
        const record = await readRecord(root, kind, slugOf(name.slice(0, -3)));
        const { markdown: _markdown, ...summary } = record;
        records[kind].push(summary);
      } catch { unreadable.push(`${folder(kind)}/${name}`); }
    }
  }
  return {
    workspace: path.dirname(root), output: root, library: config.library,
    includeMarkdown: config.includeMarkdown, records, unreadable, truncatedKinds,
    confirmedDirections: records.direction.filter(record => record.status === "confirmed"),
  };
}

async function readRecord(root: string, kind: Kind, slug: string): Promise<Record<string, unknown>> {
  const relative = recordPath(kind, slug);
  const markdown = await readOptional(root, relative);
  if (markdown === null) throw new Error(`Record not found: ${relative}`);
  const meta = parseRecord(markdown, kind, slug);
  return { ...meta, path: path.join(root, relative), sha256: sha256(markdown), markdown };
}

async function saveRecord(root: string, input: Input): Promise<Record<string, unknown>> {
  const kind = kindOf(input.kind);
  const slug = slugOf(input.slug);
  const title = requiredString(input.title, "title", 500);
  const body = requiredString(input.body, "body", 200_000);
  if (!Array.isArray(input.sources) || input.sources.length > 100) throw new Error("sources must be an array of at most 100 source identifiers or paths");
  const sources = input.sources.map(value => requiredString(value, "source", 8192));
  const meta: RecordMeta = { arxiv_daily_agent: 1, kind, slug, title, status: kind === "direction" ? "draft" : "saved", sources };
  const markdown = renderRecord(meta, body);
  if (Buffer.byteLength(markdown) > MAX_RECORD_BYTES) throw new Error("Research record exceeds the size limit");
  return mutateRecord(root, kind, slug, input.expectedSha256, (current) => {
    if (current !== null) {
      const oldMeta = parseRecord(current, kind, slug);
      // A no-op save must preserve a prior direction confirmation.
      if (current === renderRecord({ ...meta, status: oldMeta.status }, body)) return current;
    }
    return markdown;
  });
}

async function confirmDirection(root: string, input: Input): Promise<Record<string, unknown>> {
  const slug = slugOf(input.slug);
  requiredString(input.expectedSha256, "expectedSha256", 64);
  return mutateRecord(root, "direction", slug, input.expectedSha256, (current) => {
    if (current === null) throw new Error("Direction draft does not exist");
    const meta = parseRecord(current, "direction", slug);
    return renderRecord({ ...meta, status: "confirmed" }, recordBody(current));
  });
}

async function mutateRecord(
  root: string, kind: Kind, slug: string, expected: unknown,
  update: (current: string | null) => string,
): Promise<Record<string, unknown>> {
  const storage = storageFor(root);
  const lock = await storage.acquireLock("agent-records", { wait: true });
  if (!lock) throw new Error("Research workspace is busy");
  try {
    const relative = recordPath(kind, slug);
    await ensureDirectory(path.dirname(path.join(root, relative)), false);
    const current = await readOptional(root, relative);
    const next = update(current);
    if (next !== current) {
      if ((current !== null && expected !== sha256(current)) || (current === null && expected !== undefined)) {
        throw new Error("Record changed (revision conflict). Read it again and preserve existing researcher edits");
      }
      await storage.writeTextAtomic(relative, next, 0o600);
    }
    return await readRecord(root, kind, slug);
  } finally { await lock.release(); }
}

function renderRecord(meta: RecordMeta, body: string): string {
  return `---\n${Object.entries(meta).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join("\n")}\n---\n\n${body.trim()}\n`;
}

function parseRecord(markdown: string, kind: Kind, slug: string): RecordMeta {
  const match = /^---\n([\s\S]*?)\n---\n/.exec(markdown);
  if (!match) throw new Error("Record has no supported frontmatter; preserve the file");
  const v = parseYaml(match[1]!, { maxAliasCount: 0 }) as Partial<RecordMeta> | null;
  if (!v || v.arxiv_daily_agent !== 1 || v.kind !== kind || v.slug !== slug
    || typeof v.title !== "string" || !Array.isArray(v.sources) || !v.sources.every(s => typeof s === "string")
    || !(kind === "direction" ? ["draft", "confirmed"].includes(v.status ?? "") : v.status === "saved")) {
    throw new Error("Invalid research record; preserve the file");
  }
  return v as RecordMeta;
}

function recordBody(markdown: string): string { return markdown.replace(/^---\n[\s\S]*?\n---\n/, "").trim(); }
function folder(kind: Kind): string { return ({ direction: "directions", paper: "papers", reading: "reading", daily: "daily" })[kind]; }
function recordPath(kind: Kind, slug: string): string { return `${folder(kind)}/${slug}.md`; }
function kindOf(value: unknown): Kind {
  if (typeof value !== "string" || !KINDS.includes(value as Kind)) throw new Error("kind must be direction, paper, reading, or daily");
  return value as Kind;
}
function slugOf(value: unknown): string {
  const slug = requiredString(value, "slug", 120);
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(slug) || slug.includes("..") || /[. ]$/.test(slug)
    || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(slug)) throw new Error("Invalid record slug");
  return slug;
}
function requiredString(value: unknown, name: string, max: number): string {
  if (typeof value !== "string" || !value.trim() || value.length > max || value.includes("\0")) throw new Error(`Invalid ${name}`);
  return value.trim();
}
function optionalString(value: unknown, name: string, max: number): string {
  return value === undefined || value === "" ? "" : requiredString(value, name, max);
}
function pagination(input: Input) {
  const offset = input.offset ?? 0;
  const limit = input.limit ?? 30;
  if (!Number.isSafeInteger(offset) || Number(offset) < 0 || !Number.isSafeInteger(limit) || Number(limit) < 1 || Number(limit) > 100) throw new Error("Invalid pagination: offset >= 0 and limit 1..100");
  return { offset: Number(offset), limit: Number(limit) };
}
function keyFromFilename(name: string): string | null {
  const match = /(?:^|[^\d])(\d{4}\.\d{4,5}(?:v\d+)?)(?=[^\d]|$)/.exec(path.basename(name));
  const resource = match ? modernArxivResources(match[1]!) : null;
  return resource ? `arxiv:${resource.id}` : null;
}
function sha256(text: string): string { return createHash("sha256").update(text).digest("hex"); }
function contains(parent: string, child: string): boolean {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}
function storageFor(root: string): NodeStorageAdapter {
  return new NodeStorageAdapter(root, { lockRoot: path.join(root, ".locks") });
}
async function ensureDirectory(directory: string, create: boolean): Promise<void> {
  const absolute = path.resolve(directory);
  let current = path.parse(absolute).root;
  for (const segment of absolute.slice(current.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, segment);
    if (create) await fs.mkdir(current, { mode: 0o700 }).catch(error => { if (error.code !== "EEXIST") throw error; });
    const info = await fs.lstat(current);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Unsafe directory: symbolic links are not supported");
  }
}
async function readOptional(root: string, relative: string): Promise<string | null> {
  const target = path.join(root, relative);
  if (!contains(root, target)) throw new Error("Unsafe record path");
  try {
    await ensureDirectory(path.dirname(target), false);
    const info = await fs.lstat(target);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Unsafe file: symbolic links are not supported");
    if (info.size > MAX_RECORD_BYTES) throw new Error("Research record exceeds the size limit");
    return fs.readFile(target, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}
