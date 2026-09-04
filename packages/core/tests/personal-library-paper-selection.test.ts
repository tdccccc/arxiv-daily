import { describe, expect, it, vi } from "vitest";
import {
  PERSONAL_LIBRARY_DIRECTION_ABSTRACT_TRUNCATION_MARKER,
  PERSONAL_LIBRARY_DIRECTION_GENERATION_CONTRACT,
  PERSONAL_LIBRARY_DIRECTION_MAX_ABSTRACT_CODE_UNITS,
  PERSONAL_LIBRARY_DIRECTION_MAX_COMPLETION_TOKENS,
  PERSONAL_LIBRARY_DIRECTION_MAX_GROUPING_OUTPUT_CODE_UNITS,
  PERSONAL_LIBRARY_DIRECTION_MAX_SELECTED_PAPERS,
  buildPersonalLibraryDirectionExtractionBatches,
  proposePersonalLibraryDirections,
  renderPersonalLibraryDirectionPaper,
  selectPersonalLibraryDirectionPapers,
  validatePersonalLibraryDirectionGrouping,
} from "../src/library/personal-library-direction-proposer";
import type {
  PersonalLibraryCatalog,
  PersonalLibraryPaperRecord,
} from "../src/library/personal-library-catalog";

const scopeFingerprint = `sha256:${"a".repeat(64)}`;
const identificationFingerprint = `sha256:${"b".repeat(64)}`;
const timestamp = "2026-08-03T12:34:56.000Z";

function paper(index: number, overrides: Partial<PersonalLibraryPaperRecord> = {}): PersonalLibraryPaperRecord {
  const externalId = `2608.${String(index).padStart(5, "0")}`;
  return {
    paperKey: `arxiv:${externalId}`,
    source: "arxiv",
    externalId,
    title: `Paper ${index}`,
    authors: ["A. Author", "B. Author"],
    abstract: `Abstract ${index}`,
    published: "2026-08-01T00:00:00.000Z",
    updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI",
    categories: ["cs.AI", "cs.LG"],
    evidenceDepth: "metadata-and-abstract",
    filePaths: [`private/root/paper-${index}.pdf`],
    ...overrides,
  };
}

function catalog(papers: PersonalLibraryPaperRecord[]): PersonalLibraryCatalog {
  return {
    schemaVersion: 1,
    revision: 4,
    scopeFingerprint,
    identificationFingerprint,
    updatedAt: timestamp,
    lastScan: null,
    files: Object.fromEntries(papers.map((entry, index) => [entry.filePaths[0]!, {
      path: entry.filePaths[0]!,
      status: "ready" as const,
      observationFingerprint: `sha256:${(index % 16).toString(16).repeat(64)}`,
      paperKey: entry.paperKey,
      arxivId: entry.externalId,
      updatedAt: timestamp,
    }])),
    papers: Object.fromEntries(papers.map((entry) => [entry.paperKey, entry])),
  };
}

function candidate(keys: string[], overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({ candidates: [{
    name: "Reliable agents",
    description: "Methods for reliable agentic systems.",
    discoveryCues: ["agent evaluation", "reliable agents"],
    representativePaperKeys: keys,
    ...overrides,
  }] });
}

function groupingReply(data: any[]): string {
  const keys = data.map((entry: any) => entry.paperKey);
  const half = Math.ceil(keys.length / 2);
  return JSON.stringify({ groups: [
    { name: "Group A", description: "First half of the library papers.", paperKeys: keys.slice(0, half) },
    { name: "Group B", description: "Second half of the library papers.", paperKeys: keys.slice(half) },
  ] });
}

function isGroupingData(data: any): boolean {
  return Array.isArray(data)
    && data.every((entry: any) => typeof entry?.paperKey === "string"
      && typeof entry?.title === "string" && !("abstract" in entry));
}

function paperData(messages: ChatMessage[]): any {
  const content = messages.find(({ role }) => role === "user")!.content;
  const match = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(content);
  if (!match) throw new Error("missing paper_data");
  return JSON.parse(match[1]!.replaceAll("&lt;/paper_data&gt;", "</paper_data>"));
}

class AutomaticLlm implements PersonalLibraryDirectionLlmPort {
  calls: Array<{ messages: ChatMessage[]; options?: CallOptions }> = [];
  async call(messages: ChatMessage[], options?: CallOptions): Promise<string> {
    this.calls.push({ messages, options });
    const data = paperData(messages);
    const isGrouping = Array.isArray(data)
      && data.every((entry: any) => typeof entry?.paperKey === "string"
        && typeof entry?.title === "string" && !("abstract" in entry));
    if (isGrouping) {
      const keys = data.map((entry: any) => entry.paperKey);
      const half = Math.ceil(keys.length / 2);
      return JSON.stringify({ groups: [
        { name: "Group A", description: "First half of the library papers.", paperKeys: keys.slice(0, half) },
        { name: "Group B", description: "Second half of the library papers.", paperKeys: keys.slice(half) },
      ] });
    }
    const keys = Array.isArray(data)
      ? [data[0].paperKey]
      : [data.candidates[0].representativePaperKeys[0]];
    return candidate(keys);
  }
}

function ids(kind: "proposal" | "candidate", ordinal: number): string {
  return `${kind}.${ordinal}`;
}

function proposeOptions(entries: PersonalLibraryPaperRecord[], llm: PersonalLibraryDirectionLlmPort) {
  return { catalog: catalog(entries), llm, now: () => new Date(timestamp), createId: ids };
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

describe("personal-library direction evidence selection", () => {
  it("selects the first 200 canonical code-unit keys without mutating shuffled input", () => {
    const entries = Array.from({ length: 205 }, (_, index) => paper(index + 1)).reverse();
    const input = catalog(entries);
    const before = JSON.stringify(input);
    const selected = selectPersonalLibraryDirectionPapers(input);
    expect(selected).toHaveLength(PERSONAL_LIBRARY_DIRECTION_MAX_SELECTED_PAPERS);
    expect(selected[0]?.paperKey).toBe("arxiv:2608.00001");
    expect(selected.at(-1)?.paperKey).toBe("arxiv:2608.00200");
    expect(selected.some(({ paperKey }) => paperKey === "arxiv:2608.00201")).toBe(false);
    expect(JSON.stringify(input)).toBe(before);
  });

  it("prefers recently published papers when the catalog exceeds the selection limit", () => {
    const entries = Array.from({ length: 205 }, (_, index) => paper(index + 1, {
      published: new Date(Date.UTC(2026, 0, 1 + index)).toISOString(),
    }));
    const selected = selectPersonalLibraryDirectionPapers(catalog(entries));
    expect(selected).toHaveLength(PERSONAL_LIBRARY_DIRECTION_MAX_SELECTED_PAPERS);
    // paper(1) is the oldest by published date and must be cut; paper(205) is the newest.
    expect(selected.some(({ paperKey }) => paperKey === "arxiv:2608.00001")).toBe(false);
    expect(selected.some(({ paperKey }) => paperKey === "arxiv:2608.00006")).toBe(true);
    expect(selected.some(({ paperKey }) => paperKey === "arxiv:2608.00205")).toBe(true);
  });

  it("deterministically truncates a long abstract with the shared marker", () => {
    const long = paper(1, { abstract: "x".repeat(PERSONAL_LIBRARY_DIRECTION_MAX_ABSTRACT_CODE_UNITS + 200) });
    const rendered = renderPersonalLibraryDirectionPaper(long);
    expect(rendered.abstract).toHaveLength(PERSONAL_LIBRARY_DIRECTION_MAX_ABSTRACT_CODE_UNITS);
    expect(rendered.abstract.endsWith(PERSONAL_LIBRARY_DIRECTION_ABSTRACT_TRUNCATION_MARKER)).toBe(true);
  });
});
