import { describe, expect, it } from "vitest";
import {
  isLlmTransientExhaustedError,
  LLM_TRANSIENT_EXHAUSTED_ERROR_CODE,
  LlmTransientExhaustedError,
} from "../src/llm/client";
import {
  DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE,
  DailyPaperSummaryValidationError,
  isDailyPaperSummaryValidationError,
} from "../src/pipeline/daily-paper-summary";
import {
  DAILY_SUMMARY_ASSEMBLY_RUNTIME_ERROR_CODE,
  DailySummaryAssemblyRuntimeError,
  isDailySummaryAssemblyRuntimeError,
} from "../src/pipeline/daily-summary-assembler";
import {
  DAILY_SUMMARY_RESCUE_EXHAUSTED_ERROR_CODE,
  DAILY_SUMMARY_RESCUE_VALIDATION_ERROR_CODE,
  DailySummaryRescueExhaustedError,
  DailySummaryRescueValidationError,
  isDailySummaryRescueExhaustedError,
  isDailySummaryRescueValidationError,
} from "../src/pipeline/daily-summary-rescue";

function foreignError(fields: Record<string, unknown>): Error & Record<string, unknown> {
  return Object.assign(new Error(String(fields.message ?? "foreign")), fields);
}

describe("typed fallback error guards", () => {
  it("uses instanceof fast paths and accepts code/shape-compatible Error instances", () => {
    const cause = new Error("network");
    expect(isDailyPaperSummaryValidationError(
      new DailyPaperSummaryValidationError("2607.00001", "invalid"),
    )).toBe(true);
    expect(isDailyPaperSummaryValidationError(foreignError({
      name: "DailyPaperSummaryValidationError",
      code: DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE,
      paperId: "2607.00001",
    }))).toBe(true);
    expect(isLlmTransientExhaustedError(new LlmTransientExhaustedError(cause))).toBe(true);
    expect(isLlmTransientExhaustedError(foreignError({
      name: "LlmTransientExhaustedError",
      code: LLM_TRANSIENT_EXHAUSTED_ERROR_CODE,
      cause,
    }))).toBe(true);

    expect(isDailySummaryAssemblyRuntimeError(
      new DailySummaryAssemblyRuntimeError(cause),
    )).toBe(true);
    expect(isDailySummaryAssemblyRuntimeError(foreignError({
      name: "DailySummaryAssemblyRuntimeError",
      code: DAILY_SUMMARY_ASSEMBLY_RUNTIME_ERROR_CODE,
      cause,
    }))).toBe(true);

    const validation = foreignError({
      name: "DailySummaryRescueValidationError",
      code: DAILY_SUMMARY_RESCUE_VALIDATION_ERROR_CODE,
      failure: "mismatch",
    });
    expect(isDailySummaryRescueValidationError(validation)).toBe(true);
    expect(isDailySummaryRescueExhaustedError(foreignError({
      name: "DailySummaryRescueExhaustedError",
      code: DAILY_SUMMARY_RESCUE_EXHAUSTED_ERROR_CODE,
      attempts: 3,
      cause: validation,
    }))).toBe(true);
    expect(isDailySummaryRescueExhaustedError(
      new DailySummaryRescueExhaustedError(
        new DailySummaryRescueValidationError("mismatch"),
      ),
    )).toBe(true);
  });

  it.each([
    { name: "plain spoof", value: { name: "LlmTransientExhaustedError", message: "x", code: LLM_TRANSIENT_EXHAUSTED_ERROR_CODE, cause: new Error("x") } },
    { name: "wrong code", value: foreignError({ name: "LlmTransientExhaustedError", code: "WRONG", cause: new Error("x") }) },
    { name: "non-error cause", value: foreignError({ name: "LlmTransientExhaustedError", code: LLM_TRANSIENT_EXHAUSTED_ERROR_CODE, cause: { message: "x" } }) },
  ])("rejects unsafe LLM guard shape: $name", ({ value }) => {
    expect(isLlmTransientExhaustedError(value)).toBe(false);
  });

  it("rejects plain and incomplete validation-error spoof shapes", () => {
    expect(isDailyPaperSummaryValidationError({
      name: "DailyPaperSummaryValidationError",
      message: "invalid",
      stack: "foreign stack",
      code: DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE,
      paperId: "2607.00001",
    })).toBe(false);
    expect(isDailyPaperSummaryValidationError(foreignError({
      name: "DailyPaperSummaryValidationError",
      code: DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE,
    }))).toBe(false);
  });

  it("rejects incomplete rescue and assembly spoof shapes", () => {
    expect(isDailySummaryAssemblyRuntimeError({
      name: "DailySummaryAssemblyRuntimeError",
      message: "x",
      code: DAILY_SUMMARY_ASSEMBLY_RUNTIME_ERROR_CODE,
      cause: new Error("x"),
    })).toBe(false);
    expect(isDailySummaryRescueValidationError(foreignError({
      name: "DailySummaryRescueValidationError",
      code: DAILY_SUMMARY_RESCUE_VALIDATION_ERROR_CODE,
    }))).toBe(false);
    expect(isDailySummaryRescueExhaustedError(foreignError({
      name: "DailySummaryRescueExhaustedError",
      code: DAILY_SUMMARY_RESCUE_EXHAUSTED_ERROR_CODE,
      attempts: 3,
      cause: new Error("wrong cause"),
    }))).toBe(false);
  });
});
