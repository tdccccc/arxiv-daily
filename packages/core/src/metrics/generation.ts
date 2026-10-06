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
  /** Persisted when the report is generated, never synthesized when reading. */
  generatedAt?: string;
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
    `> <!-- arxiv-daily:generation-metrics:v1 ${JSON.stringify(normalizeMetrics(metrics))} -->\n` +
    (validTimestamp(metrics.generatedAt) ? `> - Generated at: ${metrics.generatedAt}\n` : "") +
    `> - LLM calls: ${metrics.logicalCalls} logical, ${metrics.attempts} HTTP attempt${metrics.attempts === 1 ? "" : "s"}\n` +
    `> - LLM duration: ${formatDuration(metrics.elapsedMs)}${wall}\n` +
    `> - Provider token usage: ${usage}`
  );
}

export function appendGenerationMetrics(markdown: string, metrics?: GenerationMetrics): string {
  if (!metrics) return markdown;
  const base = stripGenerationMetrics(markdown).trimEnd();
  return `${base}\n\n${generationMetricsCallout(metrics)}\n`;
}

/** Split only recognized generated metrics, keeping any later user-authored text. */
export function splitGenerationMetrics(markdown: string): { body: string; metrics: GenerationMetrics | null } {
  const lines = markdown.split("\n");
  const kept: string[] = [];
  let metrics: GenerationMetrics | null = null;
  let fence: { char: string; length: number } | null = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const fenced = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (fenced) {
      const run = fenced[1]!;
      if (!fence) fence = { char: run[0]!, length: run.length };
      else if (run[0] === fence.char && run.length >= fence.length && !fenced[2]!.trim()) fence = null;
      kept.push(line);
      continue;
    }
    if (!fence && line.trim() === GENERATION_METRICS_MARKER &&
      lines[i + 1]?.trim() === "> [!info]- Generation metrics") {
      const fields: string[] = [];
      let end = i + 2;
      // Only our known lines belong to the generated block. A later quote,
      // heading or user note must never be swallowed by metrics removal.
      while (end < lines.length && isMetricsLine(lines[end]!.trimEnd())) fields.push(lines[end++]!.trimEnd());
      const parsed = readMetricsFields(fields);
      if (parsed) {
        metrics = parsed;
        while (kept.length && !kept[kept.length - 1]!.trim()) kept.pop();
        if (lines[end]?.trim()) kept.push("");
        i = end - 1;
        continue;
      }
    }
    kept.push(line);
  }
  return { body: metrics ? kept.join("\n").trimEnd() : markdown, metrics };
}

export function stripGenerationMetrics(markdown: string): string {
  return splitGenerationMetrics(markdown).body;
}

function isMetricsLine(line: string): boolean {
  return line.length <= 4096 && /^> (?:<!-- arxiv-daily:generation-metrics:v1 .* -->|- (?:LLM calls|LLM duration|Pipeline wall time|Provider token usage|Generated at): .*)$/.test(line);
}

function readMetricsFields(lines: string[]): GenerationMetrics | null {
  const encoded = lines.find((line) => line.startsWith("> <!-- arxiv-daily:generation-metrics:v1 "));
  if (encoded) {
    try {
      const parsed: unknown = JSON.parse(encoded.slice("> <!-- arxiv-daily:generation-metrics:v1 ".length, -4));
      const valid = normalizeMetrics(parsed);
      if (valid) return valid;
    } catch { /* A malformed optional record can still have a valid legacy callout. */ }
  }
  const text = lines.join("\n");
  const calls = /^> - LLM calls: (\d+) logical, (\d+) HTTP attempts?$/m.exec(text);
  const duration = /^> - LLM duration: (\d+(?:\.\d+)?) (ms|s)$/m.exec(text);
  if (!calls || !duration) return null;
  const wall = /^> - Pipeline wall time: (\d+(?:\.\d+)?) (ms|s)$/m.exec(text);
  const usage = /^> - Provider token usage: (\d+) input \/ (\d+) output \/ (\d+) total$/m.exec(text);
  const generated = /^> - Generated at: (.+)$/m.exec(text);
  return normalizeMetrics({
    logicalCalls: Number(calls[1]), attempts: Number(calls[2]),
    elapsedMs: Number(duration[1]) * (duration[2] === "s" ? 1000 : 1),
    ...(wall ? { pipelineElapsedMs: Number(wall[1]) * (wall[2] === "s" ? 1000 : 1) } : {}),
    usageComplete: Boolean(usage),
    ...(usage ? { inputTokens: Number(usage[1]), outputTokens: Number(usage[2]), totalTokens: Number(usage[3]) } : {}),
    ...(generated ? { generatedAt: generated[1] } : {}),
  });
}

function normalizeMetrics(value: unknown): GenerationMetrics | null {
  if (!isRecord(value)) return null;
  const validNumber = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= Number.MAX_SAFE_INTEGER;
  if (!validNumber(value.logicalCalls) || !validNumber(value.attempts) || !validNumber(value.elapsedMs)) return null;
  const result: GenerationMetrics = {
    logicalCalls: value.logicalCalls, attempts: value.attempts, elapsedMs: value.elapsedMs,
    usageComplete: value.usageComplete === true && validNumber(value.inputTokens) && validNumber(value.outputTokens),
  };
  for (const key of ["inputTokens", "outputTokens", "totalTokens", "pipelineElapsedMs"] as const) {
    if (validNumber(value[key])) result[key] = value[key];
  }
  if (validTimestamp(value.generatedAt)) result.generatedAt = value.generatedAt;
  return result;
}

function validTimestamp(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/.test(value) && Number.isFinite(Date.parse(value));
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
