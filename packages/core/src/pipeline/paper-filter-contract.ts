import type { ChatMessage } from "../llm/client";
import type { CheckpointGenerationIdentity } from "../services/daily-summary-checkpoint-store";
import { buildCheckpointGenerationIdentity } from "../services/daily-summary-checkpoint-store";
import { formatArxivCategories } from "../settings/categories";
import type { ArxivSettings, LlmSettings, Topic } from "../settings/types";
import filterSystemTemplate from "../prompts/paper-filter.system.md";
import injectionGuardZh from "../prompts/injection-guard.md";
import { renderPrompt } from "../prompts/render";
import type { PaperMeta } from "./arxiv-parser";
import { escapePaperDataFence } from "./prompt-safety";

export const DAILY_FILTER_FINGERPRINT_VERSION = 1 as const;
export const DAILY_FILTER_PROMPT_CONTRACT_VERSION = 1 as const;
export const DAILY_FILTER_RESULT_CONTRACT_VERSION = 1 as const;

export interface PaperFilterRequest {
  messages: ChatMessage[];
  options: { temperature: 0 };
  identity: {
    knownIds: string[];
    validTags: string[];
  };
}

export interface FilterRecord {
  id: string;
  category: string;
}

export type FilterRecordDecodeResult =
  | { ok: true; value: FilterRecord[] }
  | { ok: false; reason: string };

export interface DailyFilterCheckpointCompatibilityInput {
  papers: PaperMeta[];
  arxivSettings: ArxivSettings;
  llm: Pick<
    LlmSettings,
    "provider" | "baseUrl" | "model" | "thinkingMode" | "reasoningEffort"
  > & { apiKey?: string };
  promptContractVersion?: number;
  resultContractVersion?: number;
}

export interface DailyFilterCheckpointFingerprintInput {
  fingerprintVersion: typeof DAILY_FILTER_FINGERPRINT_VERSION;
  request: {
    messages: ChatMessage[];
    identity: {
      knownIds: string[];
      validTags: string[];
    };
  };
  generation: CheckpointGenerationIdentity;
  promptContractVersion: number;
  resultContractVersion: number;
}

export interface PreparedDailyFilterCheckpoint {
  readonly request: PaperFilterRequest;
  readonly fingerprintInput: DailyFilterCheckpointFingerprintInput;
}

const preparedSnapshots = new WeakSet<object>();

/**
 * Capture the one exact immutable request and compatibility identity for a filter run.
 * Store ports accept only snapshots created here, so persisted compatibility cannot
 * drift from the request consumed by the live LLM call.
 */
export function prepareDailyFilterCheckpoint(
  input: DailyFilterCheckpointCompatibilityInput,
): PreparedDailyFilterCheckpoint {
  const request = buildPaperFilterRequest(input.papers, input.arxivSettings);
  const snapshot: PreparedDailyFilterCheckpoint = {
    request: clone(request),
    fingerprintInput: {
      fingerprintVersion: DAILY_FILTER_FINGERPRINT_VERSION,
      request: {
        messages: clone(request.messages),
        identity: clone(request.identity),
      },
      generation: buildCheckpointGenerationIdentity(input.llm, request.options.temperature),
      promptContractVersion:
        input.promptContractVersion ?? DAILY_FILTER_PROMPT_CONTRACT_VERSION,
      resultContractVersion:
        input.resultContractVersion ?? DAILY_FILTER_RESULT_CONTRACT_VERSION,
    },
  };
  deepFreeze(snapshot);
  preparedSnapshots.add(snapshot);
  return snapshot;
}

export function isPreparedDailyFilterCheckpoint(
  value: unknown,
): value is PreparedDailyFilterCheckpoint {
  return typeof value === "object" && value !== null && preparedSnapshots.has(value);
}

/** Construct the exact request consumed by the live paper-filter LLM call. */
export function buildPaperFilterRequest(
  papers: PaperMeta[],
  arxivSettings: ArxivSettings,
): PaperFilterRequest {
  const topics: Topic[] = arxivSettings.topics ?? [];
  const topicLines = topics.map((t) => `- ${t.tag}: ${t.description}`).join("\n");
  const tagOptions = topics.map((t) => t.tag).join("|") + "|skip";
  const papersText = papers
    .map(
      (p) =>
        `---\nID: ${escapePaperDataFence(p.id)}\n` +
        `Title: ${escapePaperDataFence(p.title)}\n` +
        `Abstract: ${escapePaperDataFence(p.abstract)}\n`,
    )
    .join("");
  return {
    messages: [
      {
        role: "system",
        content: renderPrompt(filterSystemTemplate, {
          topicLines,
          tagOptions,
          injectionGuard: injectionGuardZh,
        }),
      },
      {
        role: "user",
        content: `以下是今日 arXiv ${formatArxivCategories(arxivSettings)} 的所有新论文：\n\n<paper_data>\n${papersText}</paper_data>`,
      },
    ],
    options: { temperature: 0 },
    identity: {
      knownIds: papers.map((paper) => paper.id),
      validTags: topics.map((topic) => topic.tag),
    },
  };
}

/** Strictly decode validated model decisions while preserving record order and omissions. */
export function decodePaperFilterRecords(
  value: unknown,
  knownIds: ReadonlySet<string>,
  validTags: ReadonlySet<string>,
): FilterRecordDecodeResult {
  if (!isPlainObject(value) || !hasExactKeys(value, ["papers"]) || !Array.isArray(value.papers)) {
    return { ok: false, reason: "root must be exactly {papers:[...]}" };
  }

  const seen = new Set<string>();
  const records: FilterRecord[] = [];
  for (const record of value.papers) {
    if (!isPlainObject(record) || !hasExactKeys(record, ["id", "category"])) {
      return { ok: false, reason: "paper record has an invalid shape" };
    }
    if (typeof record.id !== "string" || !knownIds.has(record.id)) {
      return { ok: false, reason: "paper record has an unknown id" };
    }
    if (seen.has(record.id)) {
      return { ok: false, reason: "paper record has a duplicate id" };
    }
    if (
      typeof record.category !== "string" ||
      (record.category !== "skip" && !validTags.has(record.category))
    ) {
      return { ok: false, reason: `paper ${record.id} has an invalid category` };
    }
    seen.add(record.id);
    records.push({ id: record.id, category: record.category });
  }
  return { ok: true, value: records };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index]);
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
