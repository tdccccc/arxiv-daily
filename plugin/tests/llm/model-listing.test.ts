import { describe, it, expect } from "vitest";
import { buildModelUrlCandidates } from "../../src/llm/client";

describe("Model listing", () => {
  it("tries /models directly when the base URL already includes /v1", () => {
    const candidates = buildModelUrlCandidates("https://api.deepseek.com/v1");

    expect(candidates[0]).toBe("https://api.deepseek.com/v1/models");
    expect(candidates).not.toContain("https://api.deepseek.com/v1/v1/models");
  });

  it("keeps fallback candidates unique and in priority order", () => {
    const candidates = buildModelUrlCandidates("https://llm.example.com");

    expect(candidates).toEqual([
      "https://llm.example.com/v1/models",
      "https://llm.example.com/models",
    ]);
    expect(new Set(candidates).size).toBe(candidates.length);
  });
});
