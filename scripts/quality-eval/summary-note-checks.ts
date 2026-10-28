import detailPromptEn from "../../packages/core/src/prompts/paper-detail.en.system.md";
import detailPromptZh from "../../packages/core/src/prompts/paper-detail.system.md";
import {
  BANNED_GENERIC_PHRASES, distribution, findMathIssues, MATH_SPAN, rate, snippetAt,
  type LengthDistribution, type Rate,
} from "./summary-checks";
import type { DetailNote } from "./summary-corpus";

/** The sections the current detail prompt asks for, read from the prompt files so the check follows prompt edits. */
function promptHeadings(prompt: string): string[] {
  return [...prompt.matchAll(/^##[ \t]+(.+?)[ \t]*$/gm)].map((match) => match[1]!);
}
export const CURRENT_NOTE_HEADINGS = { zh: promptHeadings(detailPromptZh), en: promptHeadings(detailPromptEn) };

/** What the prompts tell the model to write when the source lacks something; counted per section. */
export const NOTE_PLACEHOLDER_PHRASES: readonly string[] = [
  "原文未说明", "原文未讨论", "原文信息不足以判断", "输入中未包含图表",
  "Not specified in the source text", "Not discussed in the source text",
  "The source text is insufficient to assess", "No figure or table information is included in the input",
];

export type NoteCheckKind = "structure" | "math" | "generic-phrase";

export interface NoteFinding {
  kind: NoteCheckKind;
  code: string;
  /** The section the finding is in; null for a finding about the whole note. */
  heading: string | null;
  snippet: string;
}

export interface NoteChecks {
  id: string;
  topic: string;
  path: string;
  title: string;
  date: string | null;
  /** Current: has every heading of the current prompt. Older: written under an earlier prompt, not a violation. */
  version: "current" | "older";
  language: "zh" | "en";
  headingSequence: string[];
  /** For an older note, the current headings it lacks; information only. */
  missingHeadings: string[];
  findings: NoteFinding[];
  /** Section heading, then placeholder phrase, then occurrences. */
  placeholders: Record<string, Record<string, number>>;
  /** Latin words over Latin words plus Han characters in prose; null when the prose has neither. */
  englishShare: number | null;
}

const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/u;
const DISPLAY_SPAN = /\$\$[^$]*?\$\$/gu;

interface BodyScan {
  /** Lines to check for inline math, with prose lines for the English share. */
  lines: string[];
  prose: string[];
  unclosedDisplay: string | null;
  /** Opening lines of `\\[ .. \\]` blocks the prompt forbids; their content is not checked line by line. */
  bracketBlocks: string[];
}

/** Walk a section body, setting aside fenced code and `$$` display blocks (allowed in notes). */
function scanBody(body: string): BodyScan {
  const scan: BodyScan = { lines: [], prose: [], unclosedDisplay: null, bracketBlocks: [] };
  let bracket = false;
  let fence: { char: string; length: number } | null = null;
  let display: string | null = null;
  for (const line of body.split(/\r?\n/)) {
    const fenced = FENCE.exec(line);
    if (!display && fenced) {
      const run = fenced[1]!;
      if (!fence) fence = { char: run[0]!, length: run.length };
      else if (run[0] === fence.char && run.length >= fence.length && !fenced[2]!.trim()) fence = null;
      continue;
    }
    if (fence) continue;
    if (bracket) {
      if (line.trimEnd().endsWith("\\]")) bracket = false;
      continue;
    }
    if (display === null && line.trim() === "\\[") {
      bracket = true;
      scan.bracketBlocks.push(line.trim());
      continue;
    }
    if (display !== null) {
      if (line.includes("$$")) display = null;
      continue;
    }
    let text = line.replace(DISPLAY_SPAN, " ");
    const open = text.indexOf("$$");
    if (open >= 0) {
      display = line;
      text = text.slice(0, open);
    }
    if (text.trim()) {
      scan.lines.push(text);
      scan.prose.push(text);
    }
  }
  scan.unclosedDisplay = display;
  return scan;
}

function englishShare(proseLines: string[]): number | null {
  const prose = proseLines.join("\n")
    .replace(MATH_SPAN, " ")
    .replace(/`[^`]*`/gu, " ")
    .replace(/\]\([^)]*\)/gu, "] ")
    .replace(/https?:\/\/\S+/gu, " ");
  const latin = prose.match(/[A-Za-z]+(?:['’-][A-Za-z]+)*/gu)?.length ?? 0;
  const han = prose.match(/\p{Script=Han}/gu)?.length ?? 0;
  return latin + han === 0 ? null : latin / (latin + han);
}

function classify(sequence: string[]): { version: "current" | "older"; language: "zh" | "en"; headings: string[] } {
  const matches = (headings: string[]) => headings.filter((heading) => sequence.includes(heading)).length;
  const language = matches(CURRENT_NOTE_HEADINGS.en) > matches(CURRENT_NOTE_HEADINGS.zh) ? "en" : "zh";
  const headings = CURRENT_NOTE_HEADINGS[language];
  return { version: headings.every((heading) => sequence.includes(heading)) ? "current" : "older", language, headings };
}

/** Run every per-note check; pure, so the same note always gives the same findings. */
export function checkNote(note: DetailNote): NoteChecks {
  const headingSequence = note.sections.map((section) => section.heading);
  const { version, language, headings } = classify(headingSequence);
  const findings: NoteFinding[] = [];

  if (version === "current") {
    const seen = new Set<string>();
    for (const heading of headingSequence) {
      if (!headings.includes(heading)) findings.push({ kind: "structure", code: "extra-heading", heading, snippet: heading });
      else if (seen.has(heading)) findings.push({ kind: "structure", code: "duplicate-heading", heading, snippet: heading });
      seen.add(heading);
    }
    const order = [...new Set(headingSequence.filter((heading) => headings.includes(heading)))];
    if (order.join("\n") !== headings.join("\n")) {
      findings.push({ kind: "structure", code: "out-of-order", heading: null, snippet: order.join(" > ") });
    }
  }

  const placeholders: NoteChecks["placeholders"] = {};
  const prose: string[] = [];
  for (const section of note.sections) {
    const scan = scanBody(section.body);
    prose.push(...scan.prose);
    for (const line of scan.lines) {
      for (const { code, snippet } of findMathIssues(line, { rendered: false })) {
        // `$$` was set aside above, so core's display-math code can only come from a stray pair; it is still allowed here.
        if (code === "display-math") continue;
        findings.push({ kind: "math", code, heading: section.heading, snippet });
      }
    }
    for (const block of scan.bracketBlocks) {
      findings.push({ kind: "math", code: "paren-delimiter", heading: section.heading, snippet: `${block} ... \\]` });
    }
    if (scan.unclosedDisplay !== null) {
      findings.push({ kind: "math", code: "unclosed-display-math", heading: section.heading, snippet: snippetAt(scan.unclosedDisplay, 0, 0) });
    }

    const lowered = section.body.toLowerCase();
    for (const phrase of NOTE_PLACEHOLDER_PHRASES) {
      const count = lowered.split(phrase.toLowerCase()).length - 1;
      if (count > 0) {
        const bySection = (placeholders[section.heading] ??= {});
        bySection[phrase] = (bySection[phrase] ?? 0) + count;
      }
    }
    for (const phrase of BANNED_GENERIC_PHRASES) {
      const at = lowered.indexOf(phrase.toLowerCase());
      if (at >= 0) findings.push({ kind: "generic-phrase", code: phrase, heading: section.heading, snippet: snippetAt(section.body, at, phrase.length) });
    }
  }

  return {
    id: note.id, topic: note.topic, path: note.path, title: note.title, date: note.date,
    version, language, headingSequence,
    missingHeadings: version === "older" ? headings.filter((heading) => !headingSequence.includes(heading)) : [],
    findings, placeholders, englishShare: englishShare(prose),
  };
}

export interface NoteGroup {
  sequence: string[];
  version: "current" | "older";
  count: number;
  exampleIds: string[];
}

export interface NoteChecksAggregate {
  notes: number;
  versions: { current: number; older: number };
  /** Notes by exact heading sequence, largest group first. */
  groups: NoteGroup[];
  /** Structure findings, over current-version notes only. */
  structure: { notes: Rate; codes: Record<string, number> };
  notesWith: Record<"math" | "generic-phrase", Rate>;
  codes: Record<"math" | "generic-phrase", Record<string, number>>;
  /** Heading, then phrase: how many notes carry it in that section and how often it occurs. */
  placeholders: Record<string, Record<string, { notes: number; occurrences: number }>>;
  englishShare: LengthDistribution;
}

export interface NoteChecksResult { results: NoteChecks[]; aggregate: NoteChecksAggregate }

const GROUP_EXAMPLES = 3;

/** Check every note and add up the heading groups, rates and distributions. */
export function checkNotes(notes: readonly DetailNote[]): NoteChecksResult {
  const results = notes.map(checkNote);
  const groups = new Map<string, NoteGroup>();
  const structureCodes: Record<string, number> = {};
  const codes: NoteChecksAggregate["codes"] = { math: {}, "generic-phrase": {} };
  const withKind = { math: 0, "generic-phrase": 0, structure: 0 };
  const placeholders: NoteChecksAggregate["placeholders"] = {};
  const versions = { current: 0, older: 0 };

  for (const result of results) {
    versions[result.version] += 1;
    const key = result.headingSequence.join("\n");
    const group = groups.get(key) ?? { sequence: result.headingSequence, version: result.version, count: 0, exampleIds: [] };
    group.count += 1;
    if (group.exampleIds.length < GROUP_EXAMPLES) group.exampleIds.push(result.id);
    groups.set(key, group);

    const kinds = new Set<NoteCheckKind>();
    for (const finding of result.findings) {
      kinds.add(finding.kind);
      const bucket = finding.kind === "structure" ? structureCodes : codes[finding.kind];
      bucket[finding.code] = (bucket[finding.code] ?? 0) + 1;
    }
    for (const kind of kinds) withKind[kind] += 1;
    for (const [heading, phrases] of Object.entries(result.placeholders)) {
      for (const [phrase, count] of Object.entries(phrases)) {
        const entry = ((placeholders[heading] ??= {})[phrase] ??= { notes: 0, occurrences: 0 });
        entry.notes += 1;
        entry.occurrences += count;
      }
    }
  }

  return {
    results,
    aggregate: {
      notes: results.length,
      versions,
      // Array.sort is stable, so equal groups keep the order in which they first appeared.
      groups: [...groups.values()].sort((a, b) => b.count - a.count),
      structure: { notes: rate(withKind.structure, versions.current), codes: structureCodes },
      notesWith: { math: rate(withKind.math, results.length), "generic-phrase": rate(withKind["generic-phrase"], results.length) },
      codes,
      placeholders,
      englishShare: distribution(results.flatMap((result) => result.englishShare === null ? [] : [result.englishShare])),
    },
  };
}
