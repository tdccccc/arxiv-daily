import { createHash } from "node:crypto";
import { buildCheckpointGenerationIdentity, type LlmSettings } from "@arxiv-daily/core";
import { LLM_TEMPERATURE } from "../../packages/core/src/llm/client";

export interface SummaryCacheIdentity {
  schemaVersion: 1;
  modelHash: string;
  requestHash: string;
}

const hash = (value: unknown): string => createHash("sha256").update(JSON.stringify(value)).digest("hex");

/** Bind cached work to the effective model settings and complete input without persisting secrets. */
export function buildSummaryCacheIdentity(
  settings: LlmSettings,
  messages: ReadonlyArray<{ role: string; content: string }>,
  temperature = LLM_TEMPERATURE,
  context?: unknown,
): SummaryCacheIdentity {
  const generation = buildCheckpointGenerationIdentity(settings, temperature);
  // GLM enables thinking but never sends reasoning_effort; changing that inactive setting is not a new experiment.
  const mode = settings.provider === "zhipu" && settings.thinkingMode ? { kind: "zhipu-thinking" } : generation.mode;
  return { schemaVersion: 1, modelHash: hash({ ...generation, mode }), requestHash: hash({ messages, context }) };
}

export function matchesSummaryCacheIdentity(value: unknown, expected: SummaryCacheIdentity): boolean {
  if (!value || typeof value !== "object") return false;
  const actual = value as Partial<SummaryCacheIdentity>;
  return actual.schemaVersion === expected.schemaVersion
    && actual.modelHash === expected.modelHash && actual.requestHash === expected.requestHash;
}
