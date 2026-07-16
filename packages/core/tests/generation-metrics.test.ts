import { describe, expect, it } from "vitest";
import {
  GENERATION_METRICS_MARKER,
  GenerationMetricsCollector,
  appendGenerationMetrics,
  parseTokenUsage,
} from "../src/metrics/generation";
import { extractPaperSummaries } from "../src/pipeline/daily-summary-parser";
import { looksLikeDetailSummary } from "../src/dashboard/detail-summary";

describe("generation metrics", () => {
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
