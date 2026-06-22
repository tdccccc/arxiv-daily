import { describe, it, expect } from "vitest";
import type { PipelineResult } from "../../src/pipeline/pipeline";

describe("PipelineResult types", () => {
  it("should support pending result kind", () => {
    const result: PipelineResult = { kind: "pending", reason: "no papers from arXiv" };
    expect(result.kind).toBe("pending");
    expect(result.reason).toBe("no papers from arXiv");
  });
});
