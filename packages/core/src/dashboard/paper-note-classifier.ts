import { looksLikeDetailSummary } from "./detail-summary";
import { modernArxivResources } from "../utils/arxiv";

export type PaperNoteClassification =
  | { kind: "verified_detail"; arxivId: string }
  | { kind: "replaceable"; arxivId: string; form: "empty" | "frontmatter_only" | "generated_empty_stub" }
  | { kind: "conflict"; arxivId: string; reason: "identity_mismatch" | "identity_invalid" | "user_content" };

/**
 * Classify a managed paper-note path without interpreting general YAML.
 * Only exact top-level arxiv_id/arxiv scalar lines are treated as identity.
 */
export function classifyPaperNote(
  markdown: string,
  expectedArxivId: string,
): PaperNoteClassification {
  const expected = modernArxivResources(expectedArxivId)?.id;
  if (!expected) {
    return { kind: "conflict", arxivId: expectedArxivId, reason: "identity_invalid" };
  }

  if (markdown.trim().length === 0) {
    return { kind: "replaceable", arxivId: expected, form: "empty" };
  }

  const frontmatter = splitStrictFrontmatter(markdown);
  const identity = frontmatter
    ? readStrictArxivIdentity(frontmatter.yaml, expected)
    : { kind: "missing" as const };
  if (identity.kind === "invalid") {
    return { kind: "conflict", arxivId: expected, reason: "identity_invalid" };
  }
  if (identity.kind === "mismatch") {
    return { kind: "conflict", arxivId: expected, reason: "identity_mismatch" };
  }

  if (frontmatter && frontmatter.body.trim().length === 0) {
    return { kind: "replaceable", arxivId: expected, form: "frontmatter_only" };
  }

  if (identity.kind === "match" && looksLikeDetailSummary(markdown)) {
    return { kind: "verified_detail", arxivId: expected };
  }

  if (
    frontmatter &&
    identity.kind === "match" &&
    isExactGeneratedEmptyStub(frontmatter.body, expected)
  ) {
    return { kind: "replaceable", arxivId: expected, form: "generated_empty_stub" };
  }

  return { kind: "conflict", arxivId: expected, reason: "user_content" };
}

function splitStrictFrontmatter(
  markdown: string,
): { yaml: string; body: string } | null {
  const match = /^---\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)([\s\S]*)$/.exec(markdown);
  return match ? { yaml: match[1] ?? "", body: match[2] ?? "" } : null;
}

function readStrictArxivIdentity(
  yaml: string,
  expected: string,
): { kind: "missing" | "match" | "mismatch" | "invalid" } {
  const ids: string[] = [];
  for (const line of yaml.split(/\r?\n/)) {
    const key = /^(arxiv_id|arxiv):(?:[ \t]*(.*))?$/.exec(line);
    if (!key) continue;
    const value = (key[2] ?? "").trim();
    const quoted = /^(?:"([^"]*)"|'([^']*)')$/.exec(value);
    const scalar = quoted ? (quoted[1] ?? quoted[2] ?? "") : value;
    if (!/^\d{4}\.\d{4,5}(?:v\d+)?$/.test(scalar)) return { kind: "invalid" };
    const normalized = modernArxivResources(scalar)?.id;
    if (!normalized) return { kind: "invalid" };
    ids.push(normalized);
  }
  if (ids.length === 0) return { kind: "missing" };
  return ids.every((id) => id === expected)
    ? { kind: "match" }
    : { kind: "mismatch" };
}

function isExactGeneratedEmptyStub(body: string, id: string): boolean {
  const normalized = body.replace(/\r\n/g, "\n").trimEnd();
  const match = /^# ([^\n]+)\n\n- \*\*arXiv\*\*: \[([^\]]+)\]\(https:\/\/arxiv\.org\/abs\/([^\s)]+)\)\n- \*\*PDF\*\*: \[PDF\]\(https:\/\/arxiv\.org\/pdf\/([^\s)]+)\)\n\n## Notes$/.exec(normalized);
  if (!match || !(match[1] ?? "").trim()) return false;
  return [match[2], match[3], match[4]].every(
    (value) => value === id,
  );
}
