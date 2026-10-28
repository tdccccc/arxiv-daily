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
