import { redactText } from "@arxiv-daily/core";

const DIAGNOSTIC_FIELDS = new Set(["error", "message", "output"]);

/** Preserve business data and JSON syntax; only diagnostic text can echo credentials. */
export function diagnosticJson(value: unknown, secrets: readonly string[]): string {
  return JSON.stringify(value, (key, field: unknown) =>
    typeof field === "string" && DIAGNOSTIC_FIELDS.has(key)
      ? redactText(field, { secrets })
      : field,
  );
}
