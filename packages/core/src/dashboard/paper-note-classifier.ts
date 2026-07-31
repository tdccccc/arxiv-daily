import { looksLikeDetailSummary } from "./detail-summary";
import { modernArxivResources } from "../utils/arxiv";

export interface VerifiedDetailMetadata {
  arxivId: string;
  title?: string;
  authors?: string;
  primaryTopic?: string;
  published?: string;
}

export type PaperNoteClassification =
  | { kind: "verified_detail"; arxivId: string; metadata: VerifiedDetailMetadata }
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
    return {
      kind: "verified_detail",
      arxivId: expected,
      metadata: readVerifiedDetailMetadata(frontmatter?.yaml ?? "", expected),
    };
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

function readVerifiedDetailMetadata(
  yaml: string,
  arxivId: string,
): VerifiedDetailMetadata {
  const metadata: VerifiedDetailMetadata = { arxivId };
  const values = new Map<string, string>();
  for (const line of yaml.split(/\r?\n/)) {
    const match = /^(title|authors|primary_topic|published):(?:[ \t]*(.*))?$/.exec(line);
    if (!match?.[1]) continue;
    const raw = (match[2] ?? "").trim();
    const value = decodeYamlScalar(raw)?.trim();
    if (value) values.set(match[1], value);
  }
  const title = values.get("title");
  const authors = values.get("authors");
  const primaryTopic = values.get("primary_topic");
  const published = values.get("published");
  const date = published ? /\d{4}-\d{2}-\d{2}/.exec(published)?.[0] : undefined;
  if (title) metadata.title = title;
  if (authors) metadata.authors = authors;
  if (primaryTopic) metadata.primaryTopic = primaryTopic;
  if (date) metadata.published = date;
  return metadata;
}

function decodeYamlScalar(raw: string): string | null {
  if (raw.startsWith('"')) {
    if (!raw.endsWith('"') || raw.length < 2) return null;
    const body = raw.slice(1, -1);
    let out = "";
    for (let index = 0; index < body.length; index += 1) {
      const char = body[index];
      if (char !== "\\") {
        out += char;
        continue;
      }
      const escaped = body[index + 1];
      // Exact inverse of markdown-writer escapeYaml. Reject other YAML escape
      // sequences rather than silently transforming metadata we did not write.
      if (escaped !== "\\" && escaped !== '"') return null;
      out += escaped;
      index += 1;
    }
    return out;
  }
  if (raw.startsWith("'")) {
    if (!raw.endsWith("'") || raw.length < 2) return null;
    return raw.slice(1, -1).replace(/''/g, "'");
  }
  return raw;
}

function isExactGeneratedEmptyStub(body: string, id: string): boolean {
  const normalized = body.replace(/\r\n/g, "\n").trimEnd();
  const match = /^# ([^\n]+)\n\n- \*\*arXiv\*\*: \[([^\]]+)\]\(https:\/\/arxiv\.org\/abs\/([^\s)]+)\)\n- \*\*PDF\*\*: \[PDF\]\(https:\/\/arxiv\.org\/pdf\/([^\s)]+)\)\n\n## Notes$/.exec(normalized);
  if (!match || !(match[1] ?? "").trim()) return false;
  return [match[2], match[3], match[4]].every(
    (value) => value === id,
  );
}
