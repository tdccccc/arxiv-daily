export { buildLegend, parseExportArgs, runExport, sampleCandidates, SHEET_COLUMNS } from "./export";
export { encodeCsv, parseCsv } from "./csv";
export { parseEvaluateArgs, runEvaluate } from "./evaluate";
export { parseSheet, SheetError } from "./sheet";
// Re-exported so script tests can build fixture vaults and fake arXiv traffic with the bundled core.
export {
  ArxivRequestCoordinator, canonicalizeScientificMarkdownMath, DAILY_FILTER_PROMPT_CONTRACT_VERSION, DAILY_FILTER_RESULT_CONTRACT_VERSION,
  DEFAULT_SETTINGS, FilterDecisionRecordStore, Logger,
} from "@arxiv-daily/core";
export { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
export { DEFAULT_DAILY_DIR, DEFAULT_PAPERS_DIR, loadSummaryCorpus } from "./summary-corpus";
export type { CorpusOptions, DetailNote, FallbackEntry, NoteSection, SummaryCorpus, SummaryEntry } from "./summary-corpus";
export {
  BANNED_GENERIC_PHRASES, checkEntry, checkSummaries, CLAIM_PHRASE, MAIN_RESULT_LENGTH_RANGE, mainResultRangeApplies, measureFieldLength,
  monthlyMainResult, NOT_STATED_PHRASE, SUMMARY_FIELDS, unescapeRenderedField,
} from "./summary-checks";
export type {
  ChecksAggregate, EntryChecks, LengthDistribution, MonthlyMainResult, Rate, SummaryCheckKind, SummaryChecksResult, SummaryFieldKey, SummaryFinding,
} from "./summary-checks";
export { checkNote, checkNotes, CURRENT_NOTE_HEADINGS, NOTE_PLACEHOLDER_PHRASES } from "./summary-note-checks";
export type { NoteChecks, NoteChecksAggregate, NoteChecksResult, NoteCheckKind, NoteFinding, NoteGroup } from "./summary-note-checks";
export { parseSummaryCheckArgs, runSummaryCheck } from "./summary-check";
export type { SummaryCheckOptions, SummaryCheckReport, SummaryCheckResult } from "./summary-check";
export {
  classifyContent, describePlan as describeSourcesPlan, fetchSources, parseSummarySourcesArgs, rebuildDailyInput, rebuildDetailInput, runSummarySources,
} from "./summary-sources";
export type {
  Budgets, ContentClass, FetchSourcesOptions, FetchSourcesPlan, FetchSourcesResult, SourceFailure, SourcePath, SourceRecord, SummarySourcesOptions,
} from "./summary-sources";
// Re-exported so script tests can compare a rebuilt model input with what the summarizers themselves send.
export { summarizeDailyPaper, summarizePaperDetail } from "@arxiv-daily/core";
export {
  aggregateGrounding, buildSourceIndex, checkGrounding, extractNumbers, groundNumbers, indexOfSource, SKIPPED_NUMBER_KINDS, STANDALONE_INTEGER_SKIP, stripFencedCode,
} from "./summary-grounding";
export type {
  ExtractedNumber, GroundedNumber, GroundingAggregate, GroundingResult, SourceIndex, SourceTexts, UngroundedNumber, WrittenText,
} from "./summary-grounding";
export {
  ABSENT_FROM_SOURCE_PHRASE, buildJudgeCorrection, buildJudgeSystemPrompt, checkJudgeOutput, explanationHanShare, MIN_EXPLANATION_HAN_SHARE, JUDGE_DIMENSIONS, JUDGE_VERDICTS, normalizeQuoteText,
  parseJudgeOutput, validateJudgment,
} from "./summary-judge-contract";
export type {
  JudgeCheck, JudgeContext, JudgeDimension, JudgeIssue, JudgePromptOptions, JudgeSeverity, JudgeVerdict, Judgment,
} from "./summary-judge-contract";
export {
  buildJudgeItems, DEFAULT_JUDGE_DAILY, DEFAULT_JUDGE_NOTES, DEFAULT_JUDGE_SEED, estimateJudgeInput, prepareJudgeInput, sampleJudgeTargets,
} from "./summary-judge-input";
export type {
  BuildItemsOptions, JudgeEstimate, JudgeItem, PreparedJudgeInput, PrepareJudgeOptions, SampleOptions, SampleTargets, SizeStats, SkippedItem, SkipReason,
} from "./summary-judge-input";
export { readSourceRecord } from "./summary-sources";
export { CHARS_PER_TOKEN, parseSummaryJudgeArgs, runSummaryJudge } from "./summary-judge-run";
export type { GeneratorConfig, JudgeCounts, JudgeModelLike, JudgePlan, SummaryJudgeDeps, SummaryJudgeOptions, SummaryJudgeResult } from "./summary-judge-run";
export { buildJudgeReport, loadJudgments, reasonCategory, renderJudgeReport, runJudgeReportOnly, writeJudgeReport } from "./summary-judge-report";
export type { CrossCheck, CrossCheckEntry, JudgeAggregate, JudgeReportData, JudgmentRecord } from "./summary-judge-report";
export { parseSummaryRegenerateArgs, runSummaryRegenerate, sha256Hex } from "./summary-regenerate";
export type { RegenerateCounts, RegeneratePlan, SummaryRegenerateDeps, SummaryRegenerateOptions, SummaryRegenerateResult } from "./summary-regenerate";
export { dailyPaperFromRecord, detailCallFromRecord } from "./summary-sources";
// Re-exported so script tests can simulate an unreachable model endpoint the way core's client reports it.
export { LlmTransientExhaustedError } from "@arxiv-daily/core";
export { prepareRegeneratedJudgeInput, readRegenerateRun } from "./summary-judge-regenerated";
export { hashText } from "./summary-judge-input";
export type { SummaryOrigin } from "./summary-judge-input";
