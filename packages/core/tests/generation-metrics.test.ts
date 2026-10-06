import { describe, expect, it } from "vitest";
import {
  GENERATION_METRICS_MARKER,
  GenerationMetricsCollector,
  appendGenerationMetrics,
  parseTokenUsage,
  stripGenerationMetrics,
  splitGenerationMetrics,
} from "../src/metrics/generation";
import { extractPaperSummaries } from "../src/pipeline/daily-summary-parser";
import { looksLikeDetailSummary } from "../src/dashboard/detail-summary";

describe("generation metrics", () => {
  it("roundtrips exact metrics, persisted generation time and following notes", () => {
    const metrics = { logicalCalls: 1, attempts: 2, elapsedMs: 1234, pipelineElapsedMs: 5678,
      usageComplete: false, inputTokens: 42, generatedAt: "2026-10-05T01:02:03.456Z" };
    const result = splitGenerationMetrics(appendGenerationMetrics("body", metrics) + "\n## My notes\nKeep this.\n");
    expect(result.metrics).toEqual(metrics);
    expect(result.body).toContain("body");
    expect(result.body).toContain("## My notes\nKeep this.");
    expect(result.body).not.toContain(GENERATION_METRICS_MARKER);
    expect(result.metrics?.outputTokens).toBeUndefined();
  });

  it("reads legacy metrics without inventing a generation date", () => {
    const result = splitGenerationMetrics("body\n\n" + GENERATION_METRICS_MARKER +
      "\n> [!info]- Generation metrics\n> - LLM calls: 2 logical, 3 HTTP attempts\n> - LLM duration: 1.2 s\n> - Pipeline wall time: 2.5 s\n> - Provider token usage: 10 input / 5 output / 15 total\n\nUser notes");
    expect(result.metrics).toEqual({ logicalCalls: 2, attempts: 3, elapsedMs: 1200,
      pipelineElapsedMs: 2500, usageComplete: true, inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    expect(result.body).toBe("body\n\nUser notes");
  });

  it("does not interpret fenced examples or unrelated markers as metadata", () => {
    const example = appendGenerationMetrics("example", { logicalCalls: 1, attempts: 1, elapsedMs: 1, usageComplete: false });
    for (const fence of ["```", "~~~~"]) {
      const source = fence + "markdown\n" + example + fence + "\n";
      expect(splitGenerationMetrics(source)).toEqual({ body: source, metrics: null });
    }
    const unknown = "body\n" + GENERATION_METRICS_MARKER + "\nKeep my prose";
    expect(splitGenerationMetrics(unknown)).toEqual({ body: unknown, metrics: null });
  });

  it("reads Windows line endings and leaves a later quoted note intact", () => {
    const original = appendGenerationMetrics("body", { logicalCalls: 1, attempts: 1, elapsedMs: 1, usageComplete: false });
    const result = splitGenerationMetrics((original + "\n> My private note\n").replace(/\n/g, "\r\n"));
    expect(result.metrics?.elapsedMs).toBe(1);
    expect(result.body).toContain("> My private note");
    expect(result.body).not.toContain("Generation metrics");
  });

  it("discards invalid optional metadata without claiming complete token usage", () => {
    const record = { logicalCalls: 1, attempts: 1, elapsedMs: 20, inputTokens: 10,
      outputTokens: -2, totalTokens: "unknown", pipelineElapsedMs: -1, usageComplete: true,
      generatedAt: "not a date" };
    const markdown = GENERATION_METRICS_MARKER + "\n> [!info]- Generation metrics\n" +
      "> <!-- arxiv-daily:generation-metrics:v1 " + JSON.stringify(record) + " -->\n";
    expect(splitGenerationMetrics(markdown).metrics).toEqual({ logicalCalls: 1, attempts: 1,
      elapsedMs: 20, inputTokens: 10, usageComplete: false });
  });

  it("replaces a recognized block without deleting later user notes", () => {
    const metrics = { logicalCalls: 1, attempts: 1, elapsedMs: 5, usageComplete: false };
    const first = appendGenerationMetrics("body", metrics) + "\n## User notes\nKeep it";
    const replaced = appendGenerationMetrics(first, { ...metrics, elapsedMs: 10 });
    expect(replaced.match(new RegExp(GENERATION_METRICS_MARKER, "g"))).toHaveLength(1);
    expect(splitGenerationMetrics(replaced).metrics?.elapsedMs).toBe(10);
    expect(splitGenerationMetrics(replaced).body).toContain("## User notes\nKeep it");
  });

  it("parses OpenAI and input/output usage aliases", () => {
    expect(parseTokenUsage({ usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 } }))
      .toEqual({ inputTokens: 10, outputTokens: 4, totalTokens: 14 });
    expect(parseTokenUsage({ usage: { input_tokens: 7, output_tokens: 3 } }))
      .toEqual({ inputTokens: 7, outputTokens: 3, totalTokens: 10 });
  });

  it("aggregates calls, retries, elapsed time and incomplete usage honestly", () => {
    const collector = new GenerationMetricsCollector();
    collector.record({ logicalCalls: 1, attempts: 2, elapsedMs: 100, usageComplete: true, inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    collector.record({ logicalCalls: 1, attempts: 1, elapsedMs: 50, usageComplete: false });
    collector.setPipelineElapsedMs(500);
    expect(collector.snapshot()).toEqual({
      logicalCalls: 2, attempts: 3, elapsedMs: 150, usageComplete: false,
      inputTokens: 10, outputTokens: 5, totalTokens: 15, pipelineElapsedMs: 500,
    });
  });

  it("leaves markdown byte-compatible without stats and appends one marked callout at the absolute end", () => {
    const original = "---\ntitle: x\n---\n\nbody\n";
    expect(appendGenerationMetrics(original)).toBe(original);
    const written = appendGenerationMetrics(original, {
      logicalCalls: 1, attempts: 1, elapsedMs: 1200, usageComplete: false,
    });
    expect(written.match(new RegExp(GENERATION_METRICS_MARKER, "g"))).toHaveLength(1);
    expect(written).toMatch(/Provider token usage: unavailable or incomplete\n$/);
    expect(written.indexOf(GENERATION_METRICS_MARKER)).toBeGreaterThan(written.indexOf("body"));
  });

  it("trims trailing whitespace the same way before and after stripping metrics", () => {
    expect(stripGenerationMetrics(`body\n\n  \t${GENERATION_METRICS_MARKER}\n> [!info]- Generation metrics\n> - LLM calls: 1 logical, 1 HTTP attempt\n> - LLM duration: 1 ms\n> - Provider token usage: unavailable or incomplete`)).toBe("body");
    expect(appendGenerationMetrics("body\n\n  \t", {
      logicalCalls: 1, attempts: 1, elapsedMs: 1, usageComplete: false,
    })).toMatch(/^body\n\n<!--/);
  });

  it("completes quickly on long trailing whitespace before an existing marker (CodeQL js/polynomial-redos)", () => {
    const adversarial = `body\n${" ".repeat(200_000)}${GENERATION_METRICS_MARKER}\n> [!info]- Generation metrics\n> - LLM calls: 1 logical, 1 HTTP attempt\n> - LLM duration: 1 ms\n> - Provider token usage: unavailable or incomplete`;
    const start = performance.now();
    expect(stripGenerationMetrics(adversarial)).toBe("body");
    expect(performance.now() - start).toBeLessThan(200);
  });

  it("keeps metrics out of daily summary fields and detail detection", () => {
    const daily = appendGenerationMetrics(
      "### Paper [2607.12345]\n- **Research problem**: actual summary",
      { logicalCalls: 1, attempts: 1, elapsedMs: 1, usageComplete: false },
    );
    expect(extractPaperSummaries(daily)["2607.12345"]?.coreProblem).toBe("actual summary");

    const detailBody = `# Paper\n\n## 研究问题\n${"a".repeat(150)}\n\n## 方法设计\n${"b".repeat(150)}\n\n## 主要结论\n${"c".repeat(150)}`;
    expect(looksLikeDetailSummary(appendGenerationMetrics(detailBody, {
      logicalCalls: 1, attempts: 1, elapsedMs: 1, usageComplete: false,
    }))).toBe(true);
  });
});
