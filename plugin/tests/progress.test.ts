import { describe, it, expect } from "vitest";
import { NoopProgressReporter, type ProgressStage } from "../src/services/progress";

describe("NoopProgressReporter", () => {
  it("implements all methods and returns void", () => {
    const r = new NoopProgressReporter();
    expect(() => r.setBatch(1, 1, "2026-05-11")).not.toThrow();
    expect(() => r.setStage("filter" as ProgressStage)).not.toThrow();
    expect(() => r.setStage("fetch-content" as ProgressStage, 1, 3)).not.toThrow();
    expect(() => r.setIdle()).not.toThrow();
    expect(() => r.setIdle("2026-05-11")).not.toThrow();
    expect(() => r.setIdle("2026-05-11", "weekend")).not.toThrow();
    expect(() => r.setDisabled()).not.toThrow();
  });
});
