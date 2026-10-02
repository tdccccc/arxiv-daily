import { performance } from "node:perf_hooks";
import { describe, expect, it } from "vitest";
import { buildModelUrlCandidates, normalizeOpenAiBaseUrl } from "../../src/llm/client";

describe("endpoint normalization", () => {
  it.each([
    ["", ""],
    [" /// ", ""],
    [" https://gateway.example/// ", "https://gateway.example/v1"],
    ["https://gateway.example/v1///", "https://gateway.example/v1"],
    ["https://gateway.example/custom///", "https://gateway.example/custom"],
    ["not-a-url///", "not-a-url"],
  ])("normalizes %j without changing endpoint identity", (input, expected) => {
    expect(normalizeOpenAiBaseUrl(input)).toBe(expected);
  });

  it("retains model fallback order when the base has trailing slashes", () => {
    expect(buildModelUrlCandidates("https://gateway.example/api/coding////")).toEqual([
      "https://gateway.example/api/coding/v1/models",
      "https://gateway.example/v1/models",
      "https://gateway.example/api/coding/models",
    ]);
  });

  it.each([
    ["chat", normalizeOpenAiBaseUrl],
    ["models", buildModelUrlCandidates],
  ] as const)("handles long internal slash runs promptly for %s", (_name, normalize) => {
    const input = `https://gateway.example/${"/".repeat(100_000)}path///`;
    const start = performance.now();
    const result = normalize(input);
    const elapsed = performance.now() - start;
    const base = input.slice(0, -3);
    expect(result).toEqual(typeof result === "string" ? base : [`${base}/v1/models`, `${base}/models`]);
    // A generous ceiling for a linear scan; backtracking takes seconds on this bounded input.
    expect(elapsed).toBeLessThan(1_000);
  }, 30_000);
});
