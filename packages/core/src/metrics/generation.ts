export const GENERATION_METRICS_MARKER = "<!-- arxiv-daily:generation-metrics -->";

export interface TokenUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

export interface LlmCallMetrics extends TokenUsage {
  logicalCalls: number;
  attempts: number;
  elapsedMs: number;
  usageComplete: boolean;
}

export interface GenerationMetrics extends LlmCallMetrics {
  pipelineElapsedMs?: number;
}

export type MetricsObserver = (metrics: LlmCallMetrics) => void;

export class GenerationMetricsCollector {
  private value: GenerationMetrics = emptyGenerationMetrics();

  record(metrics: LlmCallMetrics): void {
    this.value.logicalCalls += metrics.logicalCalls;
    this.value.attempts += metrics.attempts;
    this.value.elapsedMs += metrics.elapsedMs;
    this.value.usageComplete = this.value.usageComplete && metrics.usageComplete;
    addUsage(this.value, metrics);
  }

  setPipelineElapsedMs(elapsedMs: number): void {
    this.value.pipelineElapsedMs = nonNegativeInteger(elapsedMs);
  }

  snapshot(): GenerationMetrics {
    return { ...this.value };
  }
}

export function emptyGenerationMetrics(): GenerationMetrics {
  return {
    logicalCalls: 0,
    attempts: 0,
    elapsedMs: 0,
    usageComplete: true,
  };
}

export function parseTokenUsage(value: unknown): TokenUsage | undefined {
  if (!isRecord(value)) return undefined;
  const usage = isRecord(value.usage) ? value.usage : value;
  const inputTokens = firstNumber(usage, [
    "prompt_tokens", "input_tokens", "inputTokens", "promptTokens",
  ]);
  const outputTokens = firstNumber(usage, [
    "completion_tokens", "output_tokens", "outputTokens", "completionTokens",
  ]);
  const reportedTotal = firstNumber(usage, ["total_tokens", "totalTokens"]);
  const totalTokens = reportedTotal ??
    (inputTokens != null && outputTokens != null ? inputTokens + outputTokens : undefined);
  if (inputTokens == null && outputTokens == null && totalTokens == null) return undefined;
  return { inputTokens, outputTokens, totalTokens };
}

export function usageIsComplete(usage: TokenUsage | undefined): boolean {
  return usage?.inputTokens != null && usage.outputTokens != null;
}

export function generationMetricsCallout(metrics: GenerationMetrics): string {
  const usage = metrics.usageComplete
    ? `${formatCount(metrics.inputTokens)} input / ${formatCount(metrics.outputTokens)} output / ${formatCount(metrics.totalTokens)} total`
    : "unavailable or incomplete";
  const wall = metrics.pipelineElapsedMs == null
    ? ""
    : `\n> - Pipeline wall time: ${formatDuration(metrics.pipelineElapsedMs)}`;
  return (
    `${GENERATION_METRICS_MARKER}\n` +
    `> [!info]- Generation metrics\n` +
    `> - LLM calls: ${metrics.logicalCalls} logical, ${metrics.attempts} HTTP attempt${metrics.attempts === 1 ? "" : "s"}\n` +
    `> - LLM duration: ${formatDuration(metrics.elapsedMs)}${wall}\n` +
    `> - Provider token usage: ${usage}`
  );
}

export function appendGenerationMetrics(markdown: string, metrics?: GenerationMetrics): string {
  if (!metrics) return markdown;
  const base = stripGenerationMetrics(markdown).replace(/\s+$/, "");
  return `${base}\n\n${generationMetricsCallout(metrics)}\n`;
}

export function stripGenerationMetrics(markdown: string): string {
  const marker = markdown.lastIndexOf(GENERATION_METRICS_MARKER);
  return marker < 0 ? markdown : markdown.slice(0, marker).replace(/\s+$/, "");
}

function addUsage(target: TokenUsage, source: TokenUsage): void {
  target.inputTokens = addOptional(target.inputTokens, source.inputTokens);
  target.outputTokens = addOptional(target.outputTokens, source.outputTokens);
  target.totalTokens = addOptional(target.totalTokens, source.totalTokens);
}

function addOptional(current: number | undefined, value: number | undefined): number | undefined {
  return value == null ? current : (current ?? 0) + value;
}

function firstNumber(record: Record<string, unknown>, aliases: string[]): number | undefined {
  for (const alias of aliases) {
    const value = record[alias];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      return nonNegativeInteger(value);
    }
  }
  return undefined;
}

function nonNegativeInteger(value: number): number {
  return Math.max(0, Math.round(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatCount(value: number | undefined): string {
  return value == null ? "unavailable" : String(value);
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}
