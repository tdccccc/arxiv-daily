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
  dailyReport?: string;
}

interface DailyCandidate extends PaperCandidate {
  dailyReport: string;
}

interface DailyCandidateCollection {
  candidates: DailyCandidate[];
  paperIdsByReport: Map<string, Set<string>>;
  parsedReports: Set<string>;
}

export async function syncDashboardHistory(
  deps: DashboardHistorySyncDeps,
): Promise<PaperInbox> {
  const current = await deps.store.load();
  const dailyReportPaths = collectDailyReportPaths(deps);
  const paperCandidates = await collectPaperCandidates(deps, dailyReportPaths);
  const dailyCollection = await collectDailyCandidates(deps, paperCandidates);
  const inputs = buildSyncInputs(
    current,
    [
      ...dailyCollection.candidates,
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
  if (stale.clearIds.length > 0 || stale.removeIds.length > 0) {
    index = await deps.store.load();
  }

  const pruned = pruneStaleDailyReports(
    index,
    dailyReportPaths,
    dailyCollection.paperIdsByReport,
    dailyCollection.parsedReports,
    deps.output,
  );
  if (pruned.changed > 0 || pruned.removed > 0) {
    await deps.store.save(index);
    deps.logger?.info(
      `dashboard: pruned ${pruned.changed} stale daily references and removed ${pruned.removed} orphan daily papers`,
    );
    index = await deps.store.load();
  }
  return index;
}

async function collectPaperCandidates(
  deps: DashboardHistorySyncDeps,
  dailyReportPaths: Set<string>,
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
      const dailyReport = dailyReportPathFromLink(frontmatter.daily_report);
      const existingDailyReport =
        dailyReport && dailyReportPaths.has(dailyReport)
          ? dailyReport
          : undefined;
      out.set(arxivId, {
        arxivId,
        title: frontmatter.title || firstH1(markdown) || arxivId,
        authors: frontmatter.authors || "",
        date:
          frontmatter.date ||
          dateFromDailyReport(dailyReport) ||
          "1970-01-01",
        topic,
        path,
        detail,
        dailyReport: existingDailyReport,
      });
    } catch (e) {
      deps.logger?.warn(`dashboard: failed to inspect paper file ${path}`, e);
    }
  }
  return out;
}

function collectDailyReportPaths(deps: DashboardHistorySyncDeps): Set<string> {
  const dailyDir = normalizeVaultPath(deps.output.dailyDir);
  const out = new Set<string>();
  for (const file of deps.vault.getMarkdownFiles()) {
    const path = normalizeVaultPath(file.path);
    if (!dailyDateFromPath(path, dailyDir)) continue;
    out.add(path);
  }
  return out;
}

async function collectDailyCandidates(
  deps: DashboardHistorySyncDeps,
  paperCandidates: Map<string, PaperCandidate>,
): Promise<DailyCandidateCollection> {
  const dailyDir = normalizeVaultPath(deps.output.dailyDir);
  const candidates: DailyCandidate[] = [];
  const paperIdsByReport = new Map<string, Set<string>>();
  const parsedReports = new Set<string>();
  const seen = new Set<string>();
  for (const file of deps.vault.getMarkdownFiles()) {
    const path = normalizeVaultPath(file.path);
    const date = dailyDateFromPath(path, dailyDir);
    if (!date) continue;
    try {
      parsedReports.add(path);
      const ids = paperIdsByReport.get(path) ?? new Set<string>();
      paperIdsByReport.set(path, ids);
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
        ids.add(candidate.arxivId);
        candidates.push(candidate);
      }
    } catch (e) {
      deps.logger?.warn(`dashboard: failed to inspect daily file ${path}`, e);
    }
  }
  return { candidates, paperIdsByReport, parsedReports };
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
    const dailyReport = candidate.dailyReport;
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

function pruneStaleDailyReports(
  inbox: PaperInbox,
  existingDailyReports: Set<string>,
  paperIdsByReport: Map<string, Set<string>>,
  parsedReports: Set<string>,
  output: OutputSettings,
): { changed: number; removed: number } {
  const dailyDir = normalizeVaultPath(output.dailyDir);
  let changed = 0;
  let removed = 0;

  for (const [arxivId, entry] of Object.entries({ ...inbox.papers })) {
    const removedDates = new Set<string>();
    const dailyReports = entry.dailyReports
      .map(normalizeVaultPath)
      .filter((path) => {
        const date = dailyDateFromPath(path, dailyDir);
        if (!date) return true;
        const reportMissing = !existingDailyReports.has(path);
        const paperMissingFromParsedReport =
          parsedReports.has(path) &&
          !paperIdsByReport.get(path)?.has(entry.arxivId);
        if (reportMissing || paperMissingFromParsedReport) {
          removedDates.add(date);
          return false;
        }
        return true;
      });

    if (!sameStrings(entry.dailyReports, dailyReports)) {
      entry.dailyReports = dailyReports;
      changed += 1;
    }

    if (removedDates.size > 0) {
      const remainingDates = new Set(
        dailyReports
          .map((path) => dailyDateFromPath(path, dailyDir))
          .filter((date): date is string => Boolean(date)),
      );
      const seenDates = entry.seenDates.filter(
        (date) => !removedDates.has(date) || remainingDates.has(date),
      );
      if (!sameStrings(entry.seenDates, seenDates)) {
        entry.seenDates = seenDates;
        changed += 1;
      }
    }

    if (
      entry.dailyReports.length === 0 &&
      !entry.detail &&
      !entry.paperPath
    ) {
      delete inbox.papers[arxivId];
      removed += 1;
    }
  }

  return { changed, removed };
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

function dateFromDailyReport(value: string | undefined): string {
  if (!value) return "";
  const match = /\b(\d{4}-\d{2}-\d{2})\b/.exec(value);
  return match?.[1] ?? "";
}

function dailyReportPathFromLink(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const wiki = /^\[\[([^\]|]+)(?:\|[^\]]+)?\]\]$/.exec(value.trim());
  if (!wiki) return undefined;
  const path = normalizeVaultPath(wiki[1].trim());
  if (!path) return undefined;
  return path.endsWith(".md") ? path : `${path}.md`;
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

function sameStrings(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function normalizeVaultPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
}
