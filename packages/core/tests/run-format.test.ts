import { describe, expect, it } from "vitest";
import {
  describeManualResult,
  describeResult,
  describeRunResults,
} from "../src/run-format";

describe("run-format", () => {
  it("formats scheduler and pipeline results consistently", () => {
    expect(describeResult({ kind: "completed", papersWritten: 3 })).toBe(
      "done (3 papers)",
    );
    expect(describeResult({ kind: "failed_transient", reason: "timeout" })).toBe(
      "transient: timeout",
    );
    expect(describeResult({ kind: "failed_permanent", reason: "bad key" })).toBe(
      "permanent: bad key",
    );
    expect(describeResult({ kind: "skipped", reason: "already done" })).toBe(
      "skipped: already done",
    );
  });

  it("formats manual fetch results with the shared arrow convention", () => {
    expect(describeManualResult({ kind: "done", path: "papers/2606.12345.md" })).toBe(
      "done → papers/2606.12345.md",
    );
    expect(describeManualResult({ kind: "no_html", reason: "404" })).toBe(
      "no full text: 404",
    );
    expect(describeManualResult({
      kind: "note_conflict",
      path: "papers/2606.12345.md",
      reason: "protected handwritten note",
    })).toBe("note conflict at papers/2606.12345.md: protected handwritten note");
  });

  it("formats batched run results line by line", () => {
    expect(
      describeRunResults([
        { date: "2026-06-24", result: { kind: "completed", papersWritten: 2 } },
        { date: "2026-06-25", result: { kind: "skipped", reason: "weekend" } },
      ]),
    ).toBe("2026-06-24: done (2 papers)\n2026-06-25: skipped: weekend");
  });
});
