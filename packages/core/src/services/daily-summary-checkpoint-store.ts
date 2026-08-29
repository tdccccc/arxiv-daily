import type { StorageAdapter } from "../core/adapters";
import { buildChatCompletionsUrl } from "../llm/client";
import {
  DAILY_PAPER_SUMMARY_MAX_ATTEMPTS,
  isDailyPaperSummaryValidationError,
  resolveTrustedOriginalAbstract,
  validateStructuredPaperSummaryValue,
  type DailyPaperSummaryInput,
} from "../pipeline/daily-paper-summary";
import type { DailyPaperResult } from "../pipeline/daily-summary-assembler";
import { normalizeSummaryLanguage } from "../settings/summary-language";
import { sha256Hex } from "../utils/digest";
import type { LlmSettings, OutputSettings, SummaryLanguage } from "../settings/types";
import { derivePaperInboxPaths } from "./paper-index";
import { paperKeyFromArxivId } from "./paper-key";

export const DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION = 1 as const;
export const DAILY_SUMMARY_PROMPT_CONTRACT_VERSION = 1 as const;
export const DAILY_SUMMARY_RESULT_CONTRACT_VERSION = 1 as const;
export const DAILY_SUMMARY_FINGERPRINT_VERSION = 2 as const;

export interface DailySummaryCheckpointCompatibilityInput {
  paper: DailyPaperSummaryInput;
  summaryLanguage?: SummaryLanguage;
  llm: Pick<
    LlmSettings,
    "provider" | "baseUrl" | "model" | "thinkingMode" | "reasoningEffort"
  > & { apiKey?: string };
  /** The effective per-paper temperature. Daily structured summaries currently use 0. */
  temperature?: number;
  promptContractVersion?: number;
  resultContractVersion?: number;
}

export interface CheckpointGenerationIdentity {
  provider: string;
  endpointDigest: string;
  model: string;
  mode:
    | { kind: "temperature"; temperature: number }
    | { kind: "anthropic-thinking"; budgetTokens: number }
    | { kind: "reasoning-thinking"; reasoningEffort: string };
}

export interface DailySummaryCheckpointFingerprintInput {
  fingerprintVersion: typeof DAILY_SUMMARY_FINGERPRINT_VERSION;
  paper: {
    paperKey: string;
    sourceContent: {
      id: string;
      title: string;
      authors: string;
      trustedOriginalAbstract: string;
      abstractConclusion: string;
      fullSections: string | null;
    };
  };
  generation: CheckpointGenerationIdentity & {
    summaryLanguage: SummaryLanguage;
  };
  promptContractVersion: number;
  resultContractVersion: number;
}

export interface DailySummaryCheckpointEntry {
  paperKey: string;
  fingerprint: string;
  fingerprintInput: DailySummaryCheckpointFingerprintInput;
  completedAt: string;
  result: DailyPaperResult;
}

export interface DailySummaryCheckpointDocument {
  schemaVersion: typeof DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION;
  reportDate: string;
  updatedAt: string;
  entries: Record<string, DailySummaryCheckpointEntry>;
}

export interface DailySummaryCheckpointPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
}

export interface DailySummaryCheckpointStoreOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
}

export class DailySummaryCheckpointStoreError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "DailySummaryCheckpointStoreError";
  }
}

const mutationQueues = new WeakMap<StorageAdapter, Map<string, Promise<unknown>>>();

type DocumentReadResult =
  | { kind: "missing" }
  | { kind: "corrupt"; error?: unknown }
  | { kind: "unreadable"; error: unknown }
  | { kind: "valid"; document: DailySummaryCheckpointDocument };

export function deriveDailySummaryCheckpointPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
  reportDate: string,
): DailySummaryCheckpointPaths {
  requireReportDate(reportDate);
  const { indexDir } = derivePaperInboxPaths(output, (path) =>
    storage.normalizePath(path),
  );
  const directory = storage.normalizePath(`${indexDir}/daily-summary-checkpoints`);
  const documentPath = storage.normalizePath(`${directory}/${reportDate}.json`);
  return {
    directory,
    documentPath,
    backupPath: storage.normalizePath(`${documentPath}.bak`),
  };
}

export function buildDailySummaryCheckpointFingerprintInput(
  input: DailySummaryCheckpointCompatibilityInput,
): DailySummaryCheckpointFingerprintInput {
  const id = requireNonEmpty("paper.id", input.paper.id);
  const paperKey = paperKeyFromArxivId(id);
  if (paperKey !== `arxiv:${id}`) {
    throw new DailySummaryCheckpointStoreError(
      "checkpoint paper.id must be a canonical bare arXiv ID",
    );
  }
  const temperature = input.temperature ?? 0;
  if (!Number.isFinite(temperature)) {
    throw new DailySummaryCheckpointStoreError("checkpoint temperature must be finite");
  }
  return {
    fingerprintVersion: DAILY_SUMMARY_FINGERPRINT_VERSION,
    paper: {
      paperKey: paperKeyFromArxivId(id),
      sourceContent: {
        id,
        title: input.paper.title,
        authors: input.paper.authors,
        trustedOriginalAbstract: resolveTrustedOriginalAbstract(input.paper),
        abstractConclusion: input.paper.abstractConclusion.trim(),
        fullSections: input.paper.fullSections?.trim() || null,
      },
    },
    generation: {
      summaryLanguage: normalizeSummaryLanguage(input.summaryLanguage),
      ...buildCheckpointGenerationIdentity(input.llm, temperature),
    },
    promptContractVersion:
      input.promptContractVersion ?? DAILY_SUMMARY_PROMPT_CONTRACT_VERSION,
    resultContractVersion:
      input.resultContractVersion ?? DAILY_SUMMARY_RESULT_CONTRACT_VERSION,
  };
}

export function createDailySummaryCompatibilityFingerprint(
  input: DailySummaryCheckpointCompatibilityInput,
): string {
  return `sha256:${sha256Hex(JSON.stringify(buildDailySummaryCheckpointFingerprintInput(input)))}`;
}

export function buildCheckpointGenerationIdentity(
  llm: Pick<LlmSettings, "provider" | "baseUrl" | "model" | "thinkingMode" | "reasoningEffort">,
  temperature: number,
): CheckpointGenerationIdentity {
  if (!Number.isFinite(temperature)) {
    throw new DailySummaryCheckpointStoreError("checkpoint temperature must be finite");
  }
  return {
    provider: llm.provider,
    endpointDigest: buildCheckpointEndpointDigest(llm.baseUrl),
    model: llm.model,
    mode: effectiveGenerationMode(llm, temperature),
  };
}

function effectiveGenerationMode(
  llm: Pick<LlmSettings, "provider" | "thinkingMode" | "reasoningEffort">,
  temperature: number,
): DailySummaryCheckpointFingerprintInput["generation"]["mode"] {
  if (!llm.thinkingMode) return { kind: "temperature", temperature };
  if (llm.provider === "anthropic") {
    const budgets: Record<string, number> = { low: 2048, medium: 8192, high: 16384 };
    return {
      kind: "anthropic-thinking",
      budgetTokens: budgets[llm.reasoningEffort] ?? 8192,
    };
  }
  return {
    kind: "reasoning-thinking",
    reasoningEffort: llm.reasoningEffort,
  };
}

function isEffectiveGenerationMode(value: unknown, provider: string): boolean {
  if (!isPlainObject(value) || typeof value.kind !== "string") return false;
  if (value.kind === "temperature") {
    return isExactObject(value, ["kind", "temperature"]) &&
      typeof value.temperature === "number" && Number.isFinite(value.temperature);
  }
  if (value.kind === "anthropic-thinking") {
    return provider === "anthropic" &&
      isExactObject(value, ["kind", "budgetTokens"]) &&
      Number.isSafeInteger(value.budgetTokens) && value.budgetTokens > 0;
  }
  if (value.kind === "reasoning-thinking") {
    return provider !== "anthropic" &&
      isExactObject(value, ["kind", "reasoningEffort"]) &&
      typeof value.reasoningEffort === "string";
  }
  return false;
}

/** Hash the exact effective chat request URL so no endpoint text is persisted. */
export function buildCheckpointEndpointDigest(baseUrl: string): string {
  let requestUrl: URL;
  try {
    requestUrl = new URL(buildChatCompletionsUrl(baseUrl));
  } catch (error) {
    throw new DailySummaryCheckpointStoreError("checkpoint endpoint must be an absolute URL", error);
  }
  if (requestUrl.protocol !== "http:" && requestUrl.protocol !== "https:") {
    throw new DailySummaryCheckpointStoreError("checkpoint endpoint must use http or https");
  }
  return `sha256:${sha256Hex(requestUrl.toString())}`;
}

export class DailySummaryCheckpointStore {
  constructor(
    private readonly storage: StorageAdapter,
    private readonly output: OutputSettings,
    private readonly options: DailySummaryCheckpointStoreOptions = {},
  ) {}

  pathsFor(reportDate: string): DailySummaryCheckpointPaths {
    return deriveDailySummaryCheckpointPaths(this.storage, this.output, reportDate);
  }

  async load(reportDate: string): Promise<DailySummaryCheckpointDocument> {
    const paths = this.pathsFor(reportDate);
    const primary = await this.readDocument(paths.documentPath, reportDate);
    if (primary.kind === "valid") return primary.document;
    const backup = await this.readDocument(paths.backupPath, reportDate);
    if (backup.kind === "valid") {
      this.warn(`daily summary checkpoint recovered from backup: ${paths.backupPath}`);
      return backup.document;
    }
    return emptyDocument(reportDate, this.now());
  }

  async lookupReusable(
    reportDate: string,
    input: DailySummaryCheckpointCompatibilityInput,
  ): Promise<DailyPaperResult | null> {
    const paperKey = paperKeyFromArxivId(requireNonEmpty("paper.id", input.paper.id));
    const entry = (await this.load(reportDate)).entries[paperKey];
    if (!entry) return null;
    const fingerprint = createDailySummaryCompatibilityFingerprint(input);
    if (entry.fingerprint !== fingerprint) return null;
    if (entry.result.kind === "fallback" && entry.result.reasonCode === "transport-exhausted") {
      return null;
    }
    return cloneResult(entry.result);
  }

  upsert(
    reportDate: string,
    input: DailySummaryCheckpointCompatibilityInput,
    result: DailyPaperResult,
  ): Promise<DailySummaryCheckpointEntry> {
    const paths = this.pathsFor(reportDate);
    return this.enqueue(paths.documentPath, async () => {
      const document = await this.loadForMutation(reportDate);
      const fingerprintInput = buildDailySummaryCheckpointFingerprintInput(input);
      if (
        fingerprintInput.promptContractVersion !== DAILY_SUMMARY_PROMPT_CONTRACT_VERSION ||
        fingerprintInput.resultContractVersion !== DAILY_SUMMARY_RESULT_CONTRACT_VERSION
      ) {
        throw new DailySummaryCheckpointStoreError(
          "cannot persist unsupported daily summary contract versions",
        );
      }
      const paperKey = fingerprintInput.paper.paperKey;
      const paperId = fingerprintInput.paper.sourceContent.id;
      const decodedResult = decodeDailyPaperResult(
        result,
        paperId,
        fingerprintInput.paper.sourceContent.trustedOriginalAbstract,
      );
      if (!decodedResult) {
        throw new DailySummaryCheckpointStoreError(`invalid checkpoint result for ${paperKey}`);
      }
      const entry: DailySummaryCheckpointEntry = {
        paperKey,
        fingerprint: `sha256:${sha256Hex(JSON.stringify(fingerprintInput))}`,
        fingerprintInput,
        completedAt: this.now().toISOString(),
        result: decodedResult,
      };
      document.schemaVersion = DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION;
      document.reportDate = reportDate;
      document.updatedAt = this.now().toISOString();
      document.entries[paperKey] = entry;
      await this.save(paths, document);
      return cloneEntry(entry);
    });
  }

  remove(reportDate: string, paperId: string): Promise<boolean> {
    const paths = this.pathsFor(reportDate);
    const paperKey = paperKeyOrNull(paperId);
    if (!paperKey) return Promise.resolve(false);
    return this.enqueue(paths.documentPath, async () => {
      const document = await this.loadForMutation(reportDate);
      if (!(paperKey in document.entries)) return false;
      delete document.entries[paperKey];
      document.updatedAt = this.now().toISOString();
      await this.save(paths, document);
      return true;
    });
  }

  removeAll(reportDate: string): Promise<void> {
    const paths = this.pathsFor(reportDate);
    return this.enqueue(paths.documentPath, async () => {
      await removeIfExists(this.storage, paths.documentPath);
      await removeIfExists(this.storage, paths.backupPath);
      await removeIfExists(this.storage, `${paths.documentPath}.tmp`);
      await removeIfExists(this.storage, `${paths.backupPath}.tmp`);
    });
  }

  private async loadForMutation(reportDate: string): Promise<DailySummaryCheckpointDocument> {
    const paths = this.pathsFor(reportDate);
    const primary = await this.readDocument(paths.documentPath, reportDate);
    if (primary.kind === "valid") return primary.document;
    if (primary.kind === "unreadable") {
      throw new DailySummaryCheckpointStoreError(
        `cannot mutate unreadable daily summary checkpoint: ${paths.documentPath}`,
        primary.error,
      );
    }

    const backup = await this.readDocument(paths.backupPath, reportDate);
    if (backup.kind === "valid") {
      this.warn(`daily summary checkpoint recovered from backup: ${paths.backupPath}`);
      return backup.document;
    }
    if (primary.kind === "missing" && backup.kind === "missing") {
      return emptyDocument(reportDate, this.now());
    }
    throw new DailySummaryCheckpointStoreError(
      `cannot mutate unreadable daily summary checkpoint: ${paths.documentPath}`,
      backup.kind === "unreadable" || backup.kind === "corrupt"
        ? backup.error
        : primary.kind === "corrupt"
          ? primary.error
          : undefined,
    );
  }

  private async readDocument(
    path: string,
    reportDate: string,
  ): Promise<DocumentReadResult> {
    let exists: boolean;
    try {
      exists = await this.storage.exists(path);
    } catch (error) {
      this.warn(`unreadable daily summary checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    if (!exists) return { kind: "missing" };

    let raw: string;
    try {
      raw = await this.storage.readText(path);
    } catch (error) {
      this.warn(`unreadable daily summary checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      this.warn(`corrupt daily summary checkpoint ignored: ${path}`, error);
      return { kind: "corrupt", error };
    }
    const document = decodeDocument(parsed, reportDate);
    if (!document) {
      this.warn(`invalid daily summary checkpoint ignored: ${path}`);
      return { kind: "corrupt" };
    }
    if (
      isPlainObject(parsed) &&
      isPlainObject(parsed.entries) &&
      Object.keys(parsed.entries).length !== Object.keys(document.entries).length
    ) {
      this.warn(`invalid daily summary checkpoint entries ignored: ${path}`);
    }
    return { kind: "valid", document };
  }

  private async save(
    paths: DailySummaryCheckpointPaths,
    document: DailySummaryCheckpointDocument,
  ): Promise<void> {
    await ensureDirDeep(this.storage, paths.directory);
    const content = `${JSON.stringify(document, null, 2)}\n`;
    try {
      // This document owns its .tmp/.bak lifecycle. Host atomic writers may reserve
      // the same suffixes and are therefore deliberately not composed here.
      await replaceWithBackup(this.storage, paths, content, document.reportDate);
    } catch (error) {
      throw new DailySummaryCheckpointStoreError(
        `failed to save daily summary checkpoint: ${paths.documentPath}`,
        error,
      );
    }
  }

  private enqueue<T>(path: string, operation: () => Promise<T>): Promise<T> {
    let queues = mutationQueues.get(this.storage);
    if (!queues) {
      queues = new Map();
      mutationQueues.set(this.storage, queues);
    }
    const next = (queues.get(path) ?? Promise.resolve())
      .catch(() => undefined)
      .then(operation);
    const tail = next.then(() => undefined, () => undefined);
    queues.set(path, tail);
    void tail.finally(() => {
      if (queues?.get(path) === tail) queues.delete(path);
    });
    return next;
  }

  private now(): Date {
    return this.options.now?.() ?? new Date();
  }

  private warn(message: string, error?: unknown): void {
    this.options.onWarning?.(message, error);
  }
}

function decodeDocument(
  value: unknown,
  reportDate: string,
): DailySummaryCheckpointDocument | null {
  if (!isExactObject(value, ["schemaVersion", "reportDate", "updatedAt", "entries"])) {
    return null;
  }
  if (
    value.schemaVersion !== DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION ||
    value.reportDate !== reportDate ||
    !isIsoDate(value.updatedAt) ||
    !isPlainObject(value.entries)
  ) return null;

  const entries: Record<string, DailySummaryCheckpointEntry> = {};
  for (const [paperId, rawEntry] of Object.entries(value.entries)) {
    const entry = decodeEntry(rawEntry, paperId);
    if (entry) entries[paperId] = entry;
  }
  return {
    schemaVersion: DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION,
    reportDate,
    updatedAt: value.updatedAt,
    entries,
  };
}

function decodeEntry(value: unknown, mapKey: string): DailySummaryCheckpointEntry | null {
  if (!isExactObject(value, [
    "paperKey", "fingerprint", "fingerprintInput", "completedAt", "result",
  ])) return null;
  if (
    value.paperKey !== mapKey ||
    typeof value.fingerprint !== "string" ||
    !/^sha256:[0-9a-f]{64}$/.test(value.fingerprint) ||
    !isIsoDate(value.completedAt)
  ) return null;
  const fingerprintInput = decodeFingerprintInput(value.fingerprintInput);
  if (!fingerprintInput || fingerprintInput.paper.paperKey !== mapKey) return null;
  const paperId = fingerprintInput.paper.sourceContent.id;
  if (paperKeyOrNull(paperId) !== mapKey) return null;
  if (`sha256:${sha256Hex(JSON.stringify(fingerprintInput))}` !== value.fingerprint) return null;
  const result = decodeDailyPaperResult(
    value.result,
    paperId,
    fingerprintInput.paper.sourceContent.trustedOriginalAbstract,
  );
  if (!result) return null;
  return {
    paperKey: mapKey,
    fingerprint: value.fingerprint,
    fingerprintInput,
    completedAt: value.completedAt,
    result,
  };
}

function decodeFingerprintInput(value: unknown): DailySummaryCheckpointFingerprintInput | null {
  if (!isExactObject(value, [
    "fingerprintVersion", "paper", "generation", "promptContractVersion", "resultContractVersion",
  ])) return null;
  if (
    value.fingerprintVersion !== DAILY_SUMMARY_FINGERPRINT_VERSION ||
    value.promptContractVersion !== DAILY_SUMMARY_PROMPT_CONTRACT_VERSION ||
    value.resultContractVersion !== DAILY_SUMMARY_RESULT_CONTRACT_VERSION ||
    !isExactObject(value.paper, ["paperKey", "sourceContent"]) ||
    !isNonEmptyString(value.paper.paperKey) ||
    !isExactObject(value.paper.sourceContent, [
      "id", "title", "authors", "trustedOriginalAbstract", "abstractConclusion", "fullSections",
    ]) ||
    !isNonEmptyString(value.paper.sourceContent.id) ||
    typeof value.paper.sourceContent.title !== "string" ||
    typeof value.paper.sourceContent.authors !== "string" ||
    typeof value.paper.sourceContent.trustedOriginalAbstract !== "string" ||
    typeof value.paper.sourceContent.abstractConclusion !== "string" ||
    value.paper.sourceContent.abstractConclusion !== value.paper.sourceContent.abstractConclusion.trim() ||
    !(typeof value.paper.sourceContent.fullSections === "string" || value.paper.sourceContent.fullSections === null) ||
    (typeof value.paper.sourceContent.fullSections === "string" &&
      (!value.paper.sourceContent.fullSections ||
        value.paper.sourceContent.fullSections !== value.paper.sourceContent.fullSections.trim())) ||
    !isExactObject(value.generation, [
      "summaryLanguage", "provider", "endpointDigest", "model", "mode",
    ]) ||
    (value.generation.summaryLanguage !== "zh" && value.generation.summaryLanguage !== "en") ||
    typeof value.generation.provider !== "string" ||
    typeof value.generation.endpointDigest !== "string" ||
    !/^sha256:[0-9a-f]{64}$/.test(value.generation.endpointDigest) ||
    typeof value.generation.model !== "string" ||
    !isEffectiveGenerationMode(value.generation.mode, value.generation.provider)
  ) return null;
  return value as unknown as DailySummaryCheckpointFingerprintInput;
}

export function decodeDailyPaperResult(
  value: unknown,
  expectedPaperId: string,
  trustedOriginalAbstract?: string,
): DailyPaperResult | null {
  if (!isPlainObject(value)) return null;
  if (value.kind === "structured") {
    if (!isExactObject(value, ["kind", "summary"])) return null;
    try {
      return {
        kind: "structured",
        summary: validateStructuredPaperSummaryValue(value.summary, expectedPaperId),
      };
    } catch (error) {
      if (isDailyPaperSummaryValidationError(error)) return null;
      throw error;
    }
  }
  if (value.kind === "fallback") {
    if (!isExactObject(value, ["kind", "reasonCode", "attempts", "originalAbstract"])) {
      return null;
    }
    if (
      (value.reasonCode !== "validation-exhausted" && value.reasonCode !== "transport-exhausted") ||
      !Number.isSafeInteger(value.attempts) ||
      value.attempts < 1 ||
      value.attempts > DAILY_PAPER_SUMMARY_MAX_ATTEMPTS ||
      typeof value.originalAbstract !== "string" ||
      (trustedOriginalAbstract !== undefined && value.originalAbstract !== trustedOriginalAbstract)
    ) return null;
    return {
      kind: "fallback",
      reasonCode: value.reasonCode,
      attempts: value.attempts,
      originalAbstract: value.originalAbstract,
    };
  }
  return null;
}

async function replaceWithBackup(
  storage: StorageAdapter,
  paths: DailySummaryCheckpointPaths,
  content: string,
  reportDate: string,
): Promise<void> {
  const tmp = `${paths.documentPath}.tmp`;
  const backupTmp = `${paths.backupPath}.tmp`;
  await removeIfExists(storage, tmp);
  await removeIfExists(storage, backupTmp);

  try {
    await writePrivateCheckpointText(storage, tmp, content);

    let previous: string | null = null;
    if (await storage.exists(paths.documentPath)) {
      previous = await storage.readText(paths.documentPath);
      if (!decodeRawDocument(previous, reportDate)) previous = null;
    }
    let recoveryContent = previous;
    if (recoveryContent === null && await storage.exists(paths.backupPath)) {
      const backup = await storage.readText(paths.backupPath);
      if (decodeRawDocument(backup, reportDate)) recoveryContent = backup;
    }

    if (previous !== null) {
      // Keep the primary intact while publishing its replacement backup. Some
      // adapters reject rename when the destination exists, so remove the old
      // backup only after backupTmp is complete. A publish failure still leaves
      // the valid primary untouched and aborts before promotion.
      await writePrivateCheckpointText(storage, backupTmp, previous);
      await removeIfExists(storage, paths.backupPath);
      await storage.rename(backupTmp, paths.backupPath);
    }

    if (await storage.exists(paths.documentPath)) {
      await storage.remove(paths.documentPath);
    }
    try {
      await storage.rename(tmp, paths.documentPath);
    } catch (error) {
      if (recoveryContent !== null) {
        await removeIfExists(storage, tmp);
        await writePrivateCheckpointText(storage, tmp, recoveryContent);
        await storage.rename(tmp, paths.documentPath);
      }
      throw error;
    }
  } finally {
    await removeIfExists(storage, tmp);
    await removeIfExists(storage, backupTmp);
  }
}

function emptyDocument(reportDate: string, now: Date): DailySummaryCheckpointDocument {
  return {
    schemaVersion: DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION,
    reportDate,
    updatedAt: now.toISOString(),
    entries: {},
  };
}

function decodeRawDocument(raw: string, reportDate: string): DailySummaryCheckpointDocument | null {
  try {
    return decodeDocument(JSON.parse(raw), reportDate);
  } catch {
    return null;
  }
}

function cloneEntry(entry: DailySummaryCheckpointEntry): DailySummaryCheckpointEntry {
  return JSON.parse(JSON.stringify(entry)) as DailySummaryCheckpointEntry;
}

function cloneResult(result: DailyPaperResult): DailyPaperResult {
  return JSON.parse(JSON.stringify(result)) as DailyPaperResult;
}

function paperKeyOrNull(paperId: string): string | null {
  try {
    return paperKeyFromArxivId(paperId);
  } catch {
    return null;
  }
}

function requireReportDate(value: string): void {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  const year = Number(match?.[1]);
  const month = Number(match?.[2]);
  const day = Number(match?.[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    !match ||
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    throw new DailySummaryCheckpointStoreError(`invalid checkpoint report date: ${value}`);
  }
}

function requireNonEmpty(name: string, value: string): string {
  if (!value.trim()) throw new DailySummaryCheckpointStoreError(`${name} must be non-empty`);
  return value;
}

function isIsoDate(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isExactObject(value: unknown, keys: readonly string[]): value is Record<string, any> {
  if (!isPlainObject(value)) return false;
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

async function writePrivateCheckpointText(
  storage: StorageAdapter,
  path: string,
  content: string,
): Promise<void> {
  if (storage.writeTextWithMode) {
    await storage.writeTextWithMode(path, content, 0o600);
    return;
  }
  await storage.writeText(path, content);
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}

async function removeIfExists(storage: StorageAdapter, path: string): Promise<void> {
  if (await storage.exists(path)) await storage.remove(path);
}

/** @internal Test vector seam; production callers use compatibility fingerprints. */
export function sha256ForCheckpointTests(input: string): string {
  return sha256Hex(input);
}
