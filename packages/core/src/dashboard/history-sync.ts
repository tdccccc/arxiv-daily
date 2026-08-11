import type {
  PaperInbox,
  PaperIndexStore,
  PaperIndexUpsert,
  PaperSummary,
} from "../services/paper-index";
import { paperKeyFromArxivId } from "../services/paper-key";
import type { OutputSettings, Topic } from "../settings/types";
import {
  extractFallbackAbstracts,
  extractPaperSummaries,
} from "../pipeline/daily-summary-parser";
import { classifyPaperNote } from "./paper-note-classifier";
import { modernArxivResources } from "../utils/arxiv";

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
  markdownFiles?: DashboardMarkdownFile[];
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
  abstract?: string;
}

interface DailyCandidate extends PaperCandidate {
  dailyReport: string;
}

type PaperNoteScanKind =
  | "verified_detail"
  | "replaceable"
  | "conflict"
  | "unreadable"
  | "unidentified";

interface PaperNoteObservation {
  path: string;
  kind: PaperNoteScanKind;
  arxivId?: string;
  conflictReason?: "identity_mismatch" | "identity_invalid" | "user_content";
  replaceableForm?: "empty" | "frontmatter_only" | "generated_empty_stub";
}

interface PaperCandidateCollection {
  candidates: Map<string, PaperCandidate>;
  observationsByPath: Map<string, PaperNoteObservation>;
  duplicateIds: Set<string>;
  ambiguousIds: Set<string>;
}

interface DailyCandidateCollection {
  candidates: DailyCandidate[];
  paperIdsByReport: Map<string, Set<string>>;
  parsedReports: Set<string>;
  summaries: Record<string, PaperSummary>;
}

export async function syncDashboardHistory(
  deps: DashboardHistorySyncDeps,
): Promise<PaperInbox> {
  const baseline = await deps.store.load();
  const markdownFiles = deps.markdownFiles ?? deps.vault.getMarkdownFiles();
  const dailyReportPaths = collectDailyReportPaths(deps, markdownFiles);
  const paperScan = await collectPaperCandidates(deps, markdownFiles, baseline);
  const dailyCollection = await collectDailyCandidates(
    deps,
    paperScan.candidates,
    markdownFiles,
  );
  const candidates = [
    ...dailyCollection.candidates,
    ...[...paperScan.candidates.values()].filter((candidate) => candidate.detail),
  ];
  const final = await deps.store.mutate((index, mutation) => {
    const destructiveSnapshot = snapshotInbox(index);
    const inputs = buildSyncInputs(index, candidates, baseline, paperScan);
    if (inputs.length > 0) mutation.upsertManyFromDailyPapers(inputs);
    const summariesChanged = mutation.setSummaries(dailyCollection.summaries);

    const stale = pruneStaleDetails(
      index,
      baseline,
      destructiveSnapshot,
      paperScan,
      deps.output,
    );

    const pruned = pruneStaleDailyReports(
      index,
      baseline,
      destructiveSnapshot,
      dailyReportPaths,
      dailyCollection.paperIdsByReport,
      dailyCollection.parsedReports,
      deps.output,
    );
    const changed =
      inputs.length > 0 ||
      summariesChanged > 0 ||
      stale.clearIds.length > 0 ||
      stale.removeIds.length > 0 ||
      pruned.changed > 0 ||
      pruned.removed > 0;
    return {
      result: {
        index,
        inputs: inputs.length,
        summariesChanged,
        stale,
        pruned,
      },
      changed,
    };
  });

  if (final.inputs > 0) {
    deps.logger?.info(`dashboard: synced ${final.inputs} historical papers`);
  }
  if (final.summariesChanged > 0) {
    deps.logger?.info(
      `dashboard: backfilled summaries for ${final.summariesChanged} historical papers`,
    );
  }
  if (final.stale.clearIds.length > 0) {
    deps.logger?.info(
      `dashboard: cleared ${final.stale.clearIds.length} stale detail summaries`,
    );
  }
  if (final.stale.removeIds.length > 0) {
    deps.logger?.info(
      `dashboard: removed ${final.stale.removeIds.length} orphan detail summaries`,
    );
  }
  if (final.pruned.changed > 0 || final.pruned.removed > 0) {
    deps.logger?.info(
      `dashboard: pruned ${final.pruned.changed} stale daily references and removed ${final.pruned.removed} orphan daily papers`,
    );
  }
  return final.index;
}

async function collectPaperCandidates(
  deps: DashboardHistorySyncDeps,
  markdownFiles: DashboardMarkdownFile[],
  baseline: PaperInbox,
): Promise<PaperCandidateCollection> {
  const papersDir = normalizeVaultPath(deps.output.papersDir);
  const observationsByPath = new Map<string, PaperNoteObservation>();
  const candidatesById = new Map<string, PaperCandidate[]>();
  const observedPathsById = new Map<string, Set<string>>();
  const ambiguousIds = new Set<string>();
  const baselineIdByPath = new Map(
    Object.values(baseline.papers)
      .map((entry) => [
        normalizeVaultPath(entry.paperPath ?? ""),
        entry.arxivId,
      ] as const)
      .filter(([path]) => isDirectChildMarkdown(path, papersDir)),
  );
  for (const file of markdownFiles) {
    const path = normalizeVaultPath(file.path);
    if (!isDirectChildMarkdown(path, papersDir)) continue;
    const pathArxivId = normalizeArxivId(basenameWithoutExtension(path));
    if (pathArxivId) registerObservedPath(observedPathsById, pathArxivId, path);
    try {
      const markdown = await deps.vault.adapter.read(path);
      const parsedFrontmatter = parseFrontmatter(markdown);
      const frontmatter = parsedFrontmatter.values;
      const frontmatterArxivIds = uniqueStrings(
        parsedFrontmatter.topLevelArxivScalars.map(normalizeArxivId),
      );
      const arxivId = pathArxivId || frontmatterArxivIds[0];
      if (!arxivId) {
        observationsByPath.set(path, { path, kind: "unidentified" });
        const baselineId = baselineIdByPath.get(path);
        if (baselineId) ambiguousIds.add(baselineId);
        continue;
      }
      for (const observedId of uniqueStrings([
        pathArxivId,
        ...frontmatterArxivIds,
      ])) {
        registerObservedPath(observedPathsById, observedId, path);
      }
      const identityConflict =
        parsedFrontmatter.topLevelArxivScalars.length > 1 ||
        frontmatterArxivIds.some((id) => id !== arxivId);
      const classification = identityConflict
        ? { kind: "conflict" as const, reason: "identity_mismatch" as const }
        : classifyPaperNote(markdown, arxivId);
      observationsByPath.set(path, {
        path,
        arxivId,
        kind: classification.kind,
        ...(classification.kind === "conflict" && "reason" in classification
          ? { conflictReason: classification.reason }
          : {}),
        ...(classification.kind === "replaceable"
          ? { replaceableForm: classification.form }
          : {}),
      });
      if (classification.kind !== "verified_detail") {
        for (const observedId of uniqueStrings([
          pathArxivId,
          ...frontmatterArxivIds,
        ])) {
          ambiguousIds.add(observedId);
        }
        const baselineId = baselineIdByPath.get(path);
        if (baselineId) ambiguousIds.add(baselineId);
        continue;
      }

      const topic = topicFromPaper(frontmatter, deps.topics);
      const dailyReport = dailyReportPathFromLink(frontmatter.daily_report);
      const candidates = candidatesById.get(arxivId) ?? [];
      candidates.push({
        arxivId,
        title: frontmatter.title || firstH1(markdown) || arxivId,
        authors: frontmatter.authors || "",
        date:
          frontmatter.date ||
          dateFromDailyReport(dailyReport) ||
          "1970-01-01",
        topic,
        path,
        detail: true,
      });
      candidatesById.set(arxivId, candidates);
    } catch (e) {
      observationsByPath.set(path, { path, kind: "unreadable", arxivId: pathArxivId || undefined });
      const protectedId = pathArxivId || baselineIdByPath.get(path);
      if (protectedId) ambiguousIds.add(protectedId);
      deps.logger?.warn(`dashboard: failed to inspect paper file ${path}`, e);
    }
  }

  const candidates = new Map<string, PaperCandidate>();
  const duplicateIds = new Set(
    [...observedPathsById]
      .filter(([, paths]) => paths.size > 1)
      .map(([arxivId]) => arxivId),
  );
  for (const arxivId of duplicateIds) ambiguousIds.add(arxivId);
  for (const [arxivId, matches] of candidatesById) {
    if (matches.length !== 1 || duplicateIds.has(arxivId)) {
      duplicateIds.add(arxivId);
      continue;
    }
    const candidate = matches[0];
    if (candidate) candidates.set(arxivId, candidate);
  }
  return { candidates, observationsByPath, duplicateIds, ambiguousIds };
}

function collectDailyReportPaths(
  deps: DashboardHistorySyncDeps,
  markdownFiles: DashboardMarkdownFile[],
): Set<string> {
  const dailyDir = normalizeVaultPath(deps.output.dailyDir);
  const out = new Set<string>();
  for (const file of markdownFiles) {
    const path = normalizeVaultPath(file.path);
    if (!dailyDateFromPath(path, dailyDir)) continue;
    out.add(path);
  }
  return out;
}

async function collectDailyCandidates(
  deps: DashboardHistorySyncDeps,
  paperCandidates: Map<string, PaperCandidate>,
  markdownFiles: DashboardMarkdownFile[],
): Promise<DailyCandidateCollection> {
  const dailyDir = normalizeVaultPath(deps.output.dailyDir);
  const candidates: DailyCandidate[] = [];
  const paperIdsByReport = new Map<string, Set<string>>();
  const parsedReports = new Set<string>();
  const summaries: Record<string, PaperSummary> = {};
  const seen = new Set<string>();
  const dailyFiles = markdownFiles
    .map((file) => ({ file, path: normalizeVaultPath(file.path) }))
    .filter(({ path }) => dailyDateFromPath(path, dailyDir))
    .sort((a, b) => a.path.localeCompare(b.path));
  for (const { path } of dailyFiles) {
    const date = dailyDateFromPath(path, dailyDir);
    if (!date) continue;
    try {
      const markdown = await deps.vault.adapter.read(path);
      mergePaperSummaries(summaries, extractPaperSummaries(markdown));
      const fallbackAbstracts = extractFallbackAbstracts(markdown);
      const parsed = parseDailyCandidates(
        markdown,
        path,
        date,
        deps.topics,
        paperCandidates,
        fallbackAbstracts,
      );
      parsedReports.add(path);
      const ids = paperIdsByReport.get(path) ?? new Set<string>();
      paperIdsByReport.set(path, ids);
      for (const candidate of parsed) {
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
  return { candidates, paperIdsByReport, parsedReports, summaries };
}

function mergePaperSummaries(
  target: Record<string, PaperSummary>,
  incoming: Record<string, PaperSummary>,
): void {
  for (const [arxivId, summary] of Object.entries(incoming)) {
    target[arxivId] = { ...target[arxivId], ...summary };
  }
}

function parseDailyCandidates(
  markdown: string,
  dailyReport: string,
  date: string,
  topics: Topic[],
  paperCandidates: Map<string, PaperCandidate>,
  fallbackAbstracts: Record<string, string>,
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
        abstract: fallbackAbstracts[arxivId],
      });
    }
  };

  for (const line of stripFrontmatter(markdown).split(/\r?\n/)) {
    const h2 = /^##\s+(.+?)\s*$/.exec(line);
    if (h2) {
      flush();
      currentTopic = h2[1]?.trim() ?? "";
      currentHeading = "";
      currentBlock = [];
      continue;
    }
    const h3 = /^###\s+(.+?)\s*$/.exec(line);
    if (h3) {
      flush();
      currentHeading = h3[1]?.trim() ?? "";
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
  baseline: PaperInbox,
  paperScan: PaperCandidateCollection,
): PaperIndexUpsert[] {
  const inputs: PaperIndexUpsert[] = [];
  const seenInputs = new Set<string>();
  const resolvedAbstracts = new Map(
    Object.values(inbox.papers)
      .map((entry) => [entry.arxivId, meaningfulAbstract(entry.abstract)] as const)
      .filter((item): item is readonly [string, string] => Boolean(item[1])),
  );
  for (const candidate of candidates) {
    const paperKey = paperKeyFromArxivId(candidate.arxivId);
    const existing = inbox.papers[paperKey];
    const baselineEntry = baseline.papers[paperKey];
    const dailyReport = candidate.dailyReport;
    const candidatePaperPath = candidate.detail ? candidate.path : undefined;
    if (candidate.detail && !dailyReport && baselineEntry && !existing) {
      // A detail observed before the queued mutation may have been deliberately
      // removed meanwhile. Only independent daily-report evidence may recreate
      // a non-detail entry; the stale paper-note candidate contributes nothing.
      continue;
    }
    const mergeScannedDetail = canMergeScannedDetail(
      baselineEntry,
      existing,
      candidatePaperPath,
      paperScan.ambiguousIds.has(candidate.arxivId),
    );
    const detail = candidate.detail && mergeScannedDetail;
    const paperPath = detail ? candidatePaperPath : undefined;
    const key = [
      candidate.arxivId,
      candidate.date,
      dailyReport ?? "",
      paperPath ?? "",
    ].join("\t");
    if (seenInputs.has(key)) continue;
    seenInputs.add(key);
    // Report abstracts are normalized display fallbacks, not canonical Atom data.
    // The first confirmed fallback may repair a missing value, but no report may
    // replace an existing or already-recovered meaningful abstract.
    const fallbackAbstract = resolvedAbstracts.has(candidate.arxivId)
      ? undefined
      : meaningfulAbstract(candidate.abstract);
    if (fallbackAbstract) {
      resolvedAbstracts.set(candidate.arxivId, fallbackAbstract);
    }
    if (!needsSync(existing, candidate, detail, dailyReport, paperPath, fallbackAbstract)) {
      continue;
    }
    inputs.push({
      arxivId: candidate.arxivId,
      title: candidate.title,
      authors: candidate.authors,
      date: candidate.date,
      arxivCategory: candidate.topic,
      primaryTopic: candidate.topic,
      detail,
      ...(fallbackAbstract ? { abstract: fallbackAbstract } : {}),
      ...(dailyReport ? { dailyReport } : {}),
      ...(paperPath ? { paperPath } : {}),
    });
  }
  return inputs;
}

function canMergeScannedDetail(
  baseline: PaperInbox["papers"][string] | undefined,
  current: PaperInbox["papers"][string] | undefined,
  candidatePaperPath: string | undefined,
  ambiguous: boolean,
): boolean {
  if (!candidatePaperPath || ambiguous) return false;
  if (!current) return !baseline;
  const candidatePath = normalizeVaultPath(candidatePaperPath);
  const currentPath = normalizeVaultPath(current.paperPath ?? "");
  if (!baseline) return !currentPath || currentPath === candidatePath;
  const baselinePath = normalizeVaultPath(baseline.paperPath ?? "");
  if (
    baseline.detail !== current.detail ||
    baselinePath !== currentPath
  ) {
    return false;
  }
  // An existing managed path is changed only by a candidate verified at that
  // exact path. A different path needs ambiguity-free positive proof from a
  // future explicit reconciliation, not this scan.
  return !baselinePath || baselinePath === candidatePath;
}

function needsSync(
  existing: PaperInbox["papers"][string] | undefined,
  candidate: PaperCandidate,
  detail: boolean,
  dailyReport: string | undefined,
  paperPath: string | undefined,
  fallbackAbstract: string | undefined,
): boolean {
  if (!existing) return true;
  if (dailyReport && !existing.dailyReports.includes(dailyReport)) return true;
  if (!existing.seenDates.includes(candidate.date)) return true;
  if (detail && !existing.detail) return true;
  if (
    paperPath &&
    normalizeVaultPath(existing.paperPath ?? "") !== normalizeVaultPath(paperPath)
  ) {
    return true;
  }
  if (candidate.topic && !existing.topics.includes(candidate.topic)) return true;
  if (fallbackAbstract && !meaningfulAbstract(existing.abstract)) return true;
  return false;
}

function meaningfulAbstract(value: string | null | undefined): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function pruneStaleDetails(
  inbox: PaperInbox,
  baseline: PaperInbox,
  destructiveSnapshot: PaperInbox,
  paperScan: PaperCandidateCollection,
  output: OutputSettings,
): { clearIds: string[]; removeIds: string[] } {
  const papersDir = normalizeVaultPath(output.papersDir);
  const clearIds: string[] = [];
  const removeIds: string[] = [];

  for (const baselineEntry of Object.values(baseline.papers)) {
    const baselinePath = normalizeVaultPath(baselineEntry.paperPath ?? "");
    if (!baselinePath || !isDirectChildMarkdown(baselinePath, papersDir)) continue;
    const scanStartEntry = destructiveSnapshot.papers[baselineEntry.paperKey];
    if (!scanStartEntry || !sameDestructiveFields(baselineEntry, scanStartEntry)) {
      continue;
    }
    if (paperScan.ambiguousIds.has(baselineEntry.arxivId)) continue;

    const observation = paperScan.observationsByPath.get(baselinePath);
    const confirmedStale =
      observation === undefined ||
      (observation.kind === "replaceable" &&
        observation.arxivId === baselineEntry.arxivId);
    if (!confirmedStale) continue;

    const entry = inbox.papers[baselineEntry.paperKey];
    if (!entry) continue;
    if (
      normalizeVaultPath(entry.paperPath ?? "") !== baselinePath ||
      entry.detail !== baselineEntry.detail
    ) {
      continue;
    }

    if (entry.dailyReports.length === 0 && isSafeToDeleteWholeEntry(entry)) {
      delete inbox.papers[entry.paperKey];
      removeIds.push(entry.arxivId);
    } else {
      entry.detail = false;
      entry.paperPath = null;
      clearIds.push(entry.arxivId);
    }
  }

  return { clearIds, removeIds };
}

function pruneStaleDailyReports(
  inbox: PaperInbox,
  baseline: PaperInbox,
  destructiveSnapshot: PaperInbox,
  existingDailyReports: Set<string>,
  paperIdsByReport: Map<string, Set<string>>,
  parsedReports: Set<string>,
  output: OutputSettings,
): { changed: number; removed: number } {
  const dailyDir = normalizeVaultPath(output.dailyDir);
  let changed = 0;
  let removed = 0;

  for (const baselineEntry of Object.values(baseline.papers)) {
    const managedReports = baselineEntry.dailyReports
      .map(normalizeVaultPath)
      .filter((path) => Boolean(dailyDateFromPath(path, dailyDir)));
    if (managedReports.length === 0) continue;
    const scanStartEntry = destructiveSnapshot.papers[baselineEntry.paperKey];
    if (!scanStartEntry || !sameDestructiveFields(baselineEntry, scanStartEntry)) {
      continue;
    }

    const staleReports = new Set(
      managedReports.filter((path) => {
        const reportMissing = !existingDailyReports.has(path);
        const paperMissingFromParsedReport =
          parsedReports.has(path) &&
          !paperIdsByReport.get(path)?.has(baselineEntry.arxivId);
        return reportMissing || paperMissingFromParsedReport;
      }),
    );
    if (staleReports.size === 0) continue;

    const entry = inbox.papers[baselineEntry.paperKey];
    if (!entry) continue;
    const removedDates = new Set(
      [...staleReports]
        .map((path) => dailyDateFromPath(path, dailyDir))
        .filter((date): date is string => Boolean(date)),
    );
    const dailyReports = entry.dailyReports.filter(
      (path) => !staleReports.has(normalizeVaultPath(path)),
    );
    let removedManagedReference = false;
    if (!sameStrings(entry.dailyReports, dailyReports)) {
      entry.dailyReports = dailyReports;
      changed += 1;
      removedManagedReference = true;
    }

    if (removedManagedReference && removedDates.size > 0) {
      const remainingDates = new Set(
        dailyReports
          .map((path) => dailyDateFromPath(normalizeVaultPath(path), dailyDir))
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
      removedManagedReference &&
      entry.dailyReports.length === 0 &&
      !entry.detail &&
      !entry.paperPath &&
      isSafeToDeleteWholeEntry(entry)
    ) {
      delete inbox.papers[entry.paperKey];
      removed += 1;
    }
  }

  return { changed, removed };
}

function snapshotInbox(inbox: PaperInbox): PaperInbox {
  return {
    ...inbox,
    papers: Object.fromEntries(
      Object.entries(inbox.papers).map(([key, entry]) => [
        key,
        {
          ...entry,
          topics: [...entry.topics],
          seenDates: [...entry.seenDates],
          dailyReports: [...entry.dailyReports],
          projects: [...entry.projects],
        },
      ]),
    ),
  };
}

function sameDestructiveFields(
  baseline: PaperInbox["papers"][string],
  current: PaperInbox["papers"][string],
): boolean {
  return (
    normalizeVaultPath(baseline.paperPath ?? "") ===
      normalizeVaultPath(current.paperPath ?? "") &&
    baseline.detail === current.detail &&
    sameStrings(baseline.dailyReports, current.dailyReports) &&
    sameStrings(baseline.seenDates, current.seenDates) &&
    baseline.status === current.status &&
    baseline.priority === current.priority &&
    sameStrings(baseline.projects, current.projects) &&
    baseline.pdfPath === current.pdfPath &&
    baseline.zoteroKey === current.zoteroKey &&
    baseline.zoteroUri === current.zoteroUri &&
    baseline.citationKey === current.citationKey
  );
}

function isSafeToDeleteWholeEntry(
  entry: PaperInbox["papers"][string],
): boolean {
  return (
    entry.status === "inbox" &&
    entry.priority === "normal" &&
    entry.projects.length === 0 &&
    !entry.pdfPath &&
    !entry.zoteroKey &&
    !entry.zoteroUri &&
    !entry.citationKey
  );
}

function parseFrontmatter(markdown: string): {
  values: Record<string, string>;
  topLevelArxivScalars: string[];
} {
  const match = /^---\r?\n([\s\S]*?)\r?\n---/.exec(markdown);
  if (!match) return { values: {}, topLevelArxivScalars: [] };
  const values: Record<string, string> = {};
  const topLevelArxivScalars: string[] = [];
  for (const line of (match[1] ?? "").split(/\r?\n/)) {
    const item = /^([A-Za-z_][A-Za-z0-9_-]*):[ \t]*(.*)$/.exec(line);
    if (!item) continue;
    const key = item[1];
    if (!key) continue;
    const rawValue = item[2] ?? "";
    const value = parseYamlScalar(rawValue);
    values[key] = value;
    if (key === "arxiv_id" || key === "arxiv") {
      const scalar = parseTopLevelIdentityScalar(rawValue);
      if (scalar !== null) topLevelArxivScalars.push(scalar);
    }
  }
  return { values, topLevelArxivScalars };
}

function parseTopLevelIdentityScalar(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed || trimmed.startsWith("[") || trimmed.startsWith("{")) return null;
  const quoted = /^(?:"([^"]*)"|'([^']*)')$/.exec(trimmed);
  if (quoted) return quoted[1] ?? quoted[2] ?? "";
  return /\s/.test(trimmed) ? null : trimmed;
}

function parseYamlScalar(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const quoted = /^"(.*)"$/.exec(trimmed) ?? /^'(.*)'$/.exec(trimmed);
  if (quoted) return (quoted[1] ?? "").replace(/\\"/g, "\"");
  const inlineArray = /^\[(.*)\]$/.exec(trimmed);
  if (inlineArray) {
    return (inlineArray[1] ?? "")
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
  const path = normalizeVaultPath((wiki[1] ?? "").trim());
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
  return value ? modernArxivResources(value)?.id ?? "" : "";
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

function registerObservedPath(
  observedPathsById: Map<string, Set<string>>,
  arxivId: string,
  path: string,
): void {
  const paths = observedPathsById.get(arxivId) ?? new Set<string>();
  paths.add(path);
  observedPathsById.set(arxivId, paths);
}

function sameStrings(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function normalizeVaultPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
}
