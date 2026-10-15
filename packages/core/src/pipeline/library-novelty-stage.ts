import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { Logger } from "../services/logger";
import type { MetricsObserver } from "../metrics/generation";
import type { ProgressReporter } from "../services/progress";
import { paperKeyFromArxivId } from "../services/paper-key";
import type { LlmSettings, SummaryLanguage } from "../settings/types";
import {
  attachPersonalNoveltyBasis,
  LibraryPriorWorkUnavailableError,
  PERSONAL_NOVELTY_MAX_ABSTRACT_CODE_UNITS,
  PERSONAL_NOVELTY_MAX_PAPERS,
  PERSONAL_NOVELTY_MAX_REPRESENTATIVES,
  PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS,
  PERSONAL_NOVELTY_MAX_TITLE_CODE_UNITS,
  preparePersonalizedNoveltyInput,
  runPersonalNoveltyStage,
  type LibraryPriorWorkPort,
  type NoveltyCheckpointPort,
  type NoveltyDailyPaper,
  type NoveltyRepresentativePaper,
  type PersonalNoveltyLlmPort,
  type PersonalNoveltyStageNoNoveltyReason,
  type PersonalNoveltyWithBasis,
} from "./personalized-novelty";

/**
 * The union of the per-paper bases must fit one novelty input, so a very large
 * daily cap is cut here (highest-ranked papers first) rather than invalidating
 * the whole stage.
 */
const MAX_STAGE_PAPERS = Math.min(
  PERSONAL_NOVELTY_MAX_PAPERS,
  Math.floor(PERSONAL_NOVELTY_MAX_REPRESENTATIVES / PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS),
);

/**
 * This many consecutive lookups failing with the same error class is treated as
 * a systematic fault (a bad query, a broken index), not bad luck on one paper:
 * the stage warns once and stops looking up the remaining papers.
 */
const SYSTEMATIC_LOOKUP_FAILURES = 3;

export interface LibraryNoveltyStageOptions {
  port: LibraryPriorWorkPort;
  /** Visible digest papers in report order. */
  papers: ReadonlyArray<{ id: string; title: string; abstract: string }>;
  reportDate: string;
  llm: PersonalNoveltyLlmPort;
  llmSettings: LlmSettings;
  /** Daily summary language; the novelty explanation is written in it. */
  summaryLanguage?: SummaryLanguage;
  logger: Logger;
  progress: ProgressReporter;
  checkpointStore?: NoveltyCheckpointPort;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
}

interface RetrievedPaper {
  id: string;
  query: NoveltyDailyPaper;
  basis: NoveltyRepresentativePaper[];
}

/**
 * Compares each digest paper with the library papers most similar to it.
 *
 * Returns novelty by arXiv id for the papers that earned one. Everything else
 * degrades to "no entry" and never fails the run: no retrieval, an empty or
 * malformed basis, the model finding nothing comparable, or any non-cancel
 * error. Cancellation propagates. A port that reports itself unavailable ends
 * retrieval for the remaining papers; papers already retrieved are still
 * compared.
 */
export async function runLibraryNoveltyStage(
  options: LibraryNoveltyStageOptions,
): Promise<Map<string, PersonalNoveltyWithBasis>> {
  const { logger, signal } = options;
  const novelties = new Map<string, PersonalNoveltyWithBasis>();
  try {
    const retrieved = await retrieveBases(options);
    if (retrieved.length === 0) return novelties;
    const sorted = [...retrieved].sort((left, right) =>
      codeUnitCompare(left.query.paperKey, right.query.paperKey));
    const representatives = new Map<string, NoveltyRepresentativePaper>();
    for (const { basis } of sorted) {
      for (const paper of basis) {
        if (!representatives.has(paper.paperKey)) representatives.set(paper.paperKey, paper);
      }
    }
    const { outcomes } = await runPersonalNoveltyStage({
      input: {
        papers: sorted.map(({ query }) => query),
        representatives: [...representatives.values()]
          .sort((left, right) => codeUnitCompare(left.paperKey, right.paperKey)),
      },
      matches: {
        paperBases: sorted.map(({ query, basis }) => ({
          paperKey: query.paperKey,
          basisPaperKeys: basis.map(({ paperKey }) => paperKey),
        })),
      },
      llm: options.llm,
      llmSettings: options.llmSettings,
      reportDate: options.reportDate,
      summaryLanguage: options.summaryLanguage,
      checkpointStore: options.checkpointStore,
      signal,
      onMetrics: options.onMetrics,
      onWarning: (message, error) => logger.warn(`pipeline: ${message}`, error),
    });
    const outcomeByKey = new Map(outcomes.map((outcome) => [outcome.paperKey, outcome]));
    for (const entry of retrieved) {
      const outcome = outcomeByKey.get(entry.query.paperKey);
      if (!outcome) {
        logOutcome(logger, entry.id, "failed:no-result");
      } else if (outcome.status !== "novelty") {
        logOutcome(logger, entry.id, noNoveltyOutcome(outcome.reason));
      } else {
        try {
          novelties.set(entry.id, attachPersonalNoveltyBasis(outcome.novelty, entry.basis));
          logOutcome(logger, entry.id, "written");
        } catch (error) {
          logger.warn(`pipeline: personal novelty for ${entry.id} could not be attached`, error);
          logOutcome(logger, entry.id, "failed:attach");
        }
      }
    }
    logger.info(`pipeline: personal novelty for ${novelties.size}/${options.papers.length} papers`);
  } catch (error) {
    if (isCancellationError(error)) throw error;
    throwIfCancelled(signal);
    logger.warn("pipeline: personal novelty stage failed; continuing without it", error);
    novelties.clear();
  }
  return novelties;
}

async function retrieveBases(options: LibraryNoveltyStageOptions): Promise<RetrievedPaper[]> {
  const { port, logger, signal } = options;
  const papers = options.papers.slice(0, MAX_STAGE_PAPERS);
  for (const { id } of options.papers.slice(MAX_STAGE_PAPERS)) logOutcome(logger, id, "skipped:over stage limit");
  const retrieved: RetrievedPaper[] = [];
  // Consecutive same-class failures are held back so that a systematic fault
  // produces one warning, not one stack trace per paper.
  let failing: Array<{ id: string; error: unknown }> = [];
  const flushFailures = (): void => {
    for (const { id, error } of failing) logger.warn(`pipeline: library lookup failed for ${id}`, error);
    failing = [];
  };
  for (let index = 0; index < papers.length; index += 1) {
    throwIfCancelled(signal);
    const paper = papers[index]!;
    options.progress.setStage("personal-novelty", index + 1, papers.length);
    const prepared = digestQuery(paper);
    if (typeof prepared === "string") {
      logger.warn(`pipeline: personal novelty skipped ${paper.id}: ${prepared}`);
      logOutcome(logger, paper.id, `skipped:${prepared}`);
      continue;
    }
    const query = prepared;
    let found: NoveltyRepresentativePaper[];
    try {
      found = await port.findSimilar(query, PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS, signal);
    } catch (error) {
      if (isCancellationError(error)) throw error;
      throwIfCancelled(signal);
      if (error instanceof LibraryPriorWorkUnavailableError) {
        flushFailures();
        logger.warn("pipeline: library prior work unavailable; skipping personal novelty", error);
        for (const { id } of papers.slice(index)) logOutcome(logger, id, "skipped:library unavailable");
        break;
      }
      if (failing.length > 0 && errorClass(failing[0]!.error) !== errorClass(error)) flushFailures();
      failing.push({ id: paper.id, error });
      logOutcome(logger, paper.id, `failed:${errorName(error)}`);
      if (failing.length >= SYSTEMATIC_LOOKUP_FAILURES) {
        logger.warn(
          `pipeline: library lookup failed ${failing.length} times in a row with ${describeFailure(error)}; `
          + "skipping personal novelty for the remaining papers",
          error,
        );
        failing = [];
        for (const { id } of papers.slice(index + 1)) logOutcome(logger, id, "skipped:lookup stopped");
        break;
      }
      continue;
    }
    flushFailures();
    const basis = normalizeBasis(query, found);
    if (basis === "malformed") {
      logger.warn(`pipeline: library lookup for ${paper.id} returned a malformed basis`);
      logOutcome(logger, paper.id, "skipped:malformed basis");
      continue;
    }
    if (basis.length === 0) {
      logOutcome(logger, paper.id, "skipped:no similar library paper");
      continue;
    }
    retrieved.push({ id: paper.id, query, basis });
  }
  flushFailures();
  return retrieved;
}

/**
 * One line per digest paper saying what became of its novelty comparison:
 * written, no-comparable-prior, skipped:<reason> or failed:<class>. Plain text
 * with no error object, so expected outcomes never print a stack trace.
 */
function logOutcome(logger: Logger, id: string, outcome: string): void {
  logger.info(`pipeline: personal novelty ${id}: ${outcome}`);
}

/** The model finding nothing comparable is an answer, not a fault. */
function noNoveltyOutcome(reason: PersonalNoveltyStageNoNoveltyReason): string {
  switch (reason) {
    case "no-comparable-prior": return "no-comparable-prior";
    case "plan-too-large":
    case "input-invalid": return `skipped:${reason}`;
    default: return `failed:${reason}`;
  }
}

function errorName(error: unknown): string {
  return error instanceof Error ? error.constructor.name : typeof error;
}

function errorClass(error: unknown): unknown {
  return error instanceof Error ? error.constructor : typeof error;
}

function describeFailure(error: unknown): string {
  return error instanceof Error ? `${error.constructor.name}: ${error.message}` : String(error);
}

/**
 * Canonical key plus bounded title/abstract, or the plain reason the paper
 * cannot be a novelty input.
 */
function digestQuery(paper: { id: string; title: string; abstract: string }): NoveltyDailyPaper | string {
  if (typeof paper.title !== "string" || paper.title.trim() === "") return "missing title";
  if (typeof paper.abstract !== "string" || paper.abstract.trim() === "") return "missing abstract";
  if (paper.title.length > PERSONAL_NOVELTY_MAX_TITLE_CODE_UNITS) return "title too long to compare";
  if (paper.abstract.length > PERSONAL_NOVELTY_MAX_ABSTRACT_CODE_UNITS) return "abstract too long to compare";
  try {
    const query = { paperKey: paperKeyFromArxivId(paper.id), title: paper.title, abstract: paper.abstract };
    preparePersonalizedNoveltyInput({ papers: [query], representatives: [] });
    return query;
  } catch {
    return "paper id, title or abstract is not valid comparison input";
  }
}

/**
 * Drops the query paper itself and repeats, keeps the port's rank order up to
 * the bound, then code-unit sorts the result as the novelty contract requires.
 */
function normalizeBasis(
  query: NoveltyDailyPaper,
  found: readonly NoveltyRepresentativePaper[],
): NoveltyRepresentativePaper[] | "malformed" {
  const seen = new Set<string>([query.paperKey]);
  const basis: NoveltyRepresentativePaper[] = [];
  for (const paper of found) {
    if (basis.length === PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS) break;
    const key = (paper as { paperKey?: unknown } | null)?.paperKey;
    if (typeof key !== "string" || seen.has(key)) continue;
    seen.add(key);
    basis.push(paper);
  }
  basis.sort((left, right) => codeUnitCompare(left.paperKey, right.paperKey));
  try {
    preparePersonalizedNoveltyInput({ papers: [query], representatives: basis });
  } catch {
    return "malformed";
  }
  return basis;
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
