import type { PaperInbox, PaperIndexStore, PaperIndexUpsert } from "../services/paper-index";
import type { OutputSettings, Topic } from "../settings/types";
import { looksLikeDetailSummary } from "./detail-summary";

export interface DashboardMarkdownFile {
  path: string;
}

export interface DashboardVaultLike {
  getMarkdownFiles(): DashboardMarkdownFile[];
  adapter: {
    read(path: string): Promise<string>;
  };
}

export interface DashboardHistorySyncDeps {
  vault: DashboardVaultLike;
  store: PaperIndexStore;
  output: OutputSettings;
  topics: Topic[];
  logger?: {
    info(message: string, ...rest: unknown[]): void;
    warn(message: string, ...rest: unknown[]): void;
  };
}

interface PaperCandidate {
  arxivId: string;
  title: string;
  authors: string;
  date: string;
  topic: string;
  path?: string;
  detail: boolean;
}

interface DailyCandidate extends PaperCandidate {
  dailyReport: string;
}

export async function syncDashboardHistory(
  deps: DashboardHistorySyncDeps,
): Promise<PaperInbox> {
  const current = await deps.store.load();
  const paperCandidates = await collectPaperCandidates(deps);
  const dailyCandidates = await collectDailyCandidates(deps, paperCandidates);
  const inputs = buildSyncInputs(
    current,
    [
      ...dailyCandidates,
      ...[...paperCandidates.values()].filter((candidate) => candidate.detail),
    ],
  );

  let index = current;
  if (inputs.length > 0) {
    await deps.store.upsertManyFromDailyPapers(inputs);
    deps.logger?.info(`dashboard: synced ${inputs.length} historical papers`);
    index = await deps.store.load();
  }

  const stale = staleDetailActions(index, paperCandidates, deps.output);
  if (stale.clearIds.length > 0) {
    const changed = await deps.store.clearPaperDetails(stale.clearIds);
    deps.logger?.info(`dashboard: cleared ${changed} stale detail summaries`);
  }
  if (stale.removeIds.length > 0) {
    const changed = await deps.store.removePapers(stale.removeIds);
    deps.logger?.info(`dashboard: removed ${changed} orphan detail summaries`);
  }
  return stale.clearIds.length > 0 || stale.removeIds.length > 0
    ? deps.store.load()
    : index;
}

async function collectPaperCandidates(
  deps: DashboardHistorySyncDeps,
): Promise<Map<string, PaperCandidate>> {
  const papersDir = normalizeVaultPath(deps.output.papersDir);
  const out = new Map<string, PaperCandidate>();
  for (const file of deps.vault.getMarkdownFiles()) {
    const path = normalizeVaultPath(file.path);
    if (!isDirectChildMarkdown(path, papersDir)) continue;
    try {
      const markdown = await deps.vault.adapter.read(path);
      const frontmatter = parseFrontmatter(markdown);
      const arxivId =
        normalizeArxivId(frontmatter.arxiv_id) ||
        normalizeArxivId(frontmatter.arxiv) ||
        normalizeArxivId(basenameWithoutExtension(path));
      if (!arxivId) continue;
      const detail = looksLikeDetailSummary(markdown);
      const topic = topicFromPaper(frontmatter, deps.topics);
      out.set(arxivId, {
        arxivId,
        title: frontmatter.title || firstH1(markdown) || arxivId,
        authors: frontmatter.authors || "",
        date: frontmatter.date || "1970-01-01",
        topic,
        path,
        detail,
      });
    } catch (e) {
      deps.logger?.warn(`dashboard: failed to inspect paper file ${path}`, e);
    }
  }
  return out;
}

async function collectDailyCandidates(
  deps: DashboardHistorySyncDeps,
  paperCandidates: Map<string, PaperCandidate>,
): Promise<DailyCandidate[]> {
  const dailyDir = normalizeVaultPath(deps.output.dailyDir);
  const out: DailyCandidate[] = [];
  const seen = new Set<string>();
  for (const file of deps.vault.getMarkdownFiles()) {
    const path = normalizeVaultPath(file.path);
    const date = dailyDateFromPath(path, dailyDir);
    if (!date) continue;
    try {
      const markdown = await deps.vault.adapter.read(path);
      for (const candidate of parseDailyCandidates(
        markdown,
        path,
        date,
        deps.topics,
        paperCandidates,
      )) {
        const key = `${candidate.dailyReport}:${candidate.arxivId}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(candidate);
      }
    } catch (e) {
      deps.logger?.warn(`dashboard: failed to inspect daily file ${path}`, e);
    }
  }
  return out;
}

function parseDailyCandidates(
  markdown: string,
  dailyReport: string,
  date: string,
  topics: Topic[],
  paperCandidates: Map<string, PaperCandidate>,
): DailyCandidate[] {
  const out: DailyCandidate[] = [];
  let currentTopic = "";
  let currentHeading = "";
  let currentBlock: string[] = [];

  const flush = () => {
    if (!currentHeading) return;
    const block = currentBlock.join("\n");
    const ids = extractArxivIds(`${currentHeading}\n${block}`);
    if (ids.length === 0) return;
    const topic = topicFromHeading(currentTopic, topics);
    const title = cleanDailyHeading(currentHeading);
    const authors = parseDailyAuthors(block);
    for (const arxivId of ids) {
      const paper = paperCandidates.get(arxivId);
      out.push({
        arxivId,
        title: title || paper?.title || arxivId,
        authors: authors || paper?.authors || "",
        date,
        topic: topic || paper?.topic || "arxiv",
        dailyReport,
        detail: Boolean(paper?.detail),
        path: paper?.detail ? paper.path : undefined,
      });
    }
  };

  for (const line of stripFrontmatter(markdown).split(/\r?\n/)) {
    const h2 = /^##\s+(.+?)\s*$/.exec(line);
    if (h2) {
      flush();
      currentTopic = h2[1].trim();
      currentHeading = "";
      currentBlock = [];
      continue;
    }
    const h3 = /^###\s+(.+?)\s*$/.exec(line);
    if (h3) {
      flush();
      currentHeading = h3[1].trim();
      currentBlock = [];
      continue;
    }
    if (currentHeading) currentBlock.push(line);
  }
  flush();

  return out;
}

function buildSyncInputs(
  inbox: PaperInbox,
  candidates: Array<PaperCandidate | DailyCandidate>,
): PaperIndexUpsert[] {
  const inputs: PaperIndexUpsert[] = [];
  const seenInputs = new Set<string>();
  for (const candidate of candidates) {
    const existing = inbox.papers[candidate.arxivId];
    const dailyReport =
      "dailyReport" in candidate ? candidate.dailyReport : undefined;
    const paperPath = candidate.detail ? candidate.path : undefined;
    const key = [
      candidate.arxivId,
      candidate.date,
      dailyReport ?? "",
      paperPath ?? "",
    ].join("\t");
    if (seenInputs.has(key)) continue;
    seenInputs.add(key);
    if (!needsSync(existing, candidate, dailyReport, paperPath)) continue;
    inputs.push({
      arxivId: candidate.arxivId,
      title: candidate.title,
      authors: candidate.authors,
      date: candidate.date,
      arxivCategory: candidate.topic,
      primaryTopic: candidate.topic,
      detail: candidate.detail,
      ...(dailyReport ? { dailyReport } : {}),
      ...(paperPath ? { paperPath } : {}),
    });
  }
  return inputs;
}

function needsSync(
  existing: PaperInbox["papers"][string] | undefined,
  candidate: PaperCandidate,
  dailyReport: string | undefined,
  paperPath: string | undefined,
): boolean {
  if (!existing) return true;
  if (dailyReport && !existing.dailyReports.includes(dailyReport)) return true;
  if (!existing.seenDates.includes(candidate.date)) return true;
  if (candidate.detail && !existing.detail) return true;
  if (
    paperPath &&
    normalizeVaultPath(existing.paperPath ?? "") !== normalizeVaultPath(paperPath)
  ) {
    return true;
  }
  if (candidate.topic && !existing.topics.includes(candidate.topic)) return true;
  return false;
}

function staleDetailActions(
  inbox: PaperInbox,
  paperCandidates: Map<string, PaperCandidate>,
  output: OutputSettings,
): { clearIds: string[]; removeIds: string[] } {
  const papersDir = normalizeVaultPath(output.papersDir);
  const detailIds = new Set(
    [...paperCandidates.values()]
      .filter((candidate) => candidate.detail)
      .map((candidate) => candidate.arxivId),
  );
  const clearIds: string[] = [];
  const removeIds: string[] = [];

  for (const entry of Object.values(inbox.papers)) {
    if (detailIds.has(entry.arxivId)) continue;
    const paperPath = normalizeVaultPath(entry.paperPath ?? "");
    const referencesManagedPaper =
      Boolean(paperPath) && isDirectChildMarkdown(paperPath, papersDir);
    if (!entry.detail && !referencesManagedPaper) continue;
    if (entry.dailyReports.length === 0) {
      removeIds.push(entry.arxivId);
    } else {
      clearIds.push(entry.arxivId);
    }
  }

  return { clearIds, removeIds };
}

function parseFrontmatter(markdown: string): Record<string, string> {
  const match = /^---\r?\n([\s\S]*?)\r?\n---/.exec(markdown);
  if (!match) return {};
  const out: Record<string, string> = {};
  for (const line of match[1].split(/\r?\n/)) {
    const item = /^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$/.exec(line);
    if (!item) continue;
    out[item[1]] = parseYamlScalar(item[2]);
  }
  return out;
}

function parseYamlScalar(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const quoted = /^"(.*)"$/.exec(trimmed) ?? /^'(.*)'$/.exec(trimmed);
  if (quoted) return quoted[1].replace(/\\"/g, "\"");
  const inlineArray = /^\[(.*)\]$/.exec(trimmed);
  if (inlineArray) {
    return inlineArray[1]
      .split(/\s*,\s*/)
      .map((part) => part.replace(/^["']|["']$/g, "").trim())
      .filter(Boolean)
      .join(", ");
  }
  return trimmed;
}

function topicFromPaper(frontmatter: Record<string, string>, topics: Topic[]): string {
  const direct = frontmatter.primary_topic || frontmatter.primaryTopic || "";
  if (direct) return topicFromHeading(direct, topics);
  const tags = (frontmatter.tags || "")
    .split(/\s*,\s*/)
    .map((tag) => tag.trim())
    .filter((tag) => tag && !["arxiv", "paper"].includes(tag.toLowerCase()));
  return topicFromHeading(tags[0] ?? "arxiv", topics);
}

function topicFromHeading(heading: string, topics: Topic[]): string {
  const trimmed = heading.trim();
  if (!trimmed) return "arxiv";
  const key = topicKey(trimmed);
  const topic = topics.find(
    (candidate) =>
      topicKey(candidate.tag) === key ||
      topicKey(candidate.name) === key ||
      topicKey(candidate.id) === key,
  );
  return topic?.tag || slugTopic(trimmed);
}

function parseDailyAuthors(block: string): string {
  const match = /^\s*-\s*\*\*(?:作者|Authors?)\*\*:\s*(.+?)\s*$/im.exec(block);
  return match?.[1]?.trim() ?? "";
}

function extractArxivIds(text: string): string[] {
  const ids = new Set<string>();
  const patterns = [
    /arxiv\.org\/abs\/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?/gi,
    /arxiv_id:\s*["']?([0-9]{4}\.[0-9]{4,5})(?:v\d+)?["']?/gi,
    /\[\[([0-9]{4}\.[0-9]{4,5})(?:v\d+)?(?:\|[^\]]+)?\]\]/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) {
      const id = normalizeArxivId(match[1]);
      if (id) ids.add(id);
    }
  }
  return [...ids];
}

function cleanDailyHeading(heading: string): string {
  return heading
    .replace(/\s*(?:→|->)\s*(?:\[\[[^\]]+\]\]|\[[^\]]+\]\([^)]+\))\s*$/g, "")
    .replace(/<!--.*?-->/g, "")
    .trim();
}

function firstH1(markdown: string): string {
  return /^#\s+(.+?)\s*$/m.exec(stripFrontmatter(markdown))?.[1]?.trim() ?? "";
}

function stripFrontmatter(markdown: string): string {
  return markdown.replace(/^---\r?\n[\s\S]*?\r?\n---\s*(?:\r?\n|$)/, "");
}

function normalizeArxivId(value: string | undefined): string {
  return /([0-9]{4}\.[0-9]{4,5})(?:v\d+)?/.exec(value ?? "")?.[1] ?? "";
}

function basenameWithoutExtension(path: string): string {
  return path.split("/").pop()?.replace(/\.md$/i, "") ?? "";
}

function isDirectChildMarkdown(path: string, dir: string): boolean {
  const prefix = `${dir}/`;
  if (!path.startsWith(prefix) || !/\.md$/i.test(path)) return false;
  return !path.slice(prefix.length).includes("/");
}

function dailyDateFromPath(path: string, dailyDir: string): string | null {
  const prefix = `${dailyDir}/`;
  if (!path.startsWith(prefix)) return null;
  const rest = path.slice(prefix.length);
  return /^(\d{4}-\d{2}-\d{2})\.md$/i.exec(rest)?.[1] ?? null;
}

function topicKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function slugTopic(value: string): string {
  return (
    value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "arxiv"
  );
}

function normalizeVaultPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
}
