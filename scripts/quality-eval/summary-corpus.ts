import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parseDailyReportPapers, stripGenerationMetrics, type PaperSummary } from "@arxiv-daily/core";

export const DEFAULT_DAILY_DIR = "arxiv-daily/daily";
export const DEFAULT_PAPERS_DIR = "arxiv-daily/papers";

const FIELD_KEYS = ["coreProblem", "keyMethod", "mainResult", "whyRelevant", "limitations"] as const;
type FieldKey = (typeof FIELD_KEYS)[number];
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const REPORT_FILE_RE = /^(\d{4}-\d{2}-\d{2})\.md$/;
// The first section of a note written by the current prompt; its presence marks the current format.
const CURRENT_NOTE_HEADINGS = new Set(["研究问题", "Research Problem"]);

export interface CorpusOptions {
  vaultRoot: string;
  /** Relative to the vault. */
  dailyDir?: string;
  /** Relative to the vault. */
  papersDir?: string;
  /** Inclusive YYYY-MM-DD bounds on the report date (notes: their linked report date or `published`). */
  from?: string;
  to?: string;
}

export interface SummaryEntry {
  date: string;
  topic: string | null;
  id: string;
  title: string;
  /** As the report prints them (usually the first author and "et al."). */
  authors: string;
  fields: Record<FieldKey, string>;
}

export interface FallbackEntry {
  date: string;
  topic: string | null;
  id: string;
  title: string;
}

export interface NoteSection { heading: string; body: string }

export interface DetailNote {
  id: string;
  topic: string;
  path: string;
  title: string;
  /** From the note's frontmatter, as written (usually the first author and "et al."). */
  authors: string;
  /** The report date the note links, else a plain `published` date; null when the note carries neither. */
  date: string | null;
  sections: NoteSection[];
}

export interface SummaryCorpus {
  range: { from: string | null; to: string | null };
  entries: SummaryEntry[];
  fallbacks: FallbackEntry[];
  notes: DetailNote[];
  /** Reports with neither a five-field nor a fallback entry (including pre-June schema): counted, not evaluated. */
  legacyReportDates: string[];
  counts: {
    reports: number;
    evaluableReports: number;
    legacyReports: number;
    entries: number;
    /** Entries of a five-field report with some of the five fields missing; the missing ones are empty strings. */
    incompleteEntries: number;
    fallbacks: number;
    notes: number;
    legacyNotes: number;
  };
}

/** Read the evaluable daily entries and detail notes of a vault; legacy reports and notes are only counted. */
export async function loadSummaryCorpus(options: CorpusOptions): Promise<SummaryCorpus> {
  const from = checkDate(options.from, "--from");
  const to = checkDate(options.to, "--to");
  if (from && to && from > to) throw new Error(`--to ${to} is before --from ${from}; the range would be empty.`);
  const inRange = (date: string | null) => date === null || ((!from || date >= from) && (!to || date <= to));

  const corpus: SummaryCorpus = {
    range: { from: from ?? null, to: to ?? null },
    entries: [], fallbacks: [], notes: [], legacyReportDates: [],
    counts: { reports: 0, evaluableReports: 0, legacyReports: 0, entries: 0, incompleteEntries: 0, fallbacks: 0, notes: 0, legacyNotes: 0 },
  };

  const dailyPath = path.join(options.vaultRoot, options.dailyDir ?? DEFAULT_DAILY_DIR);
  for (const name of await listMarkdown(dailyPath)) {
    const date = REPORT_FILE_RE.exec(name)?.[1];
    if (!date || !inRange(date)) continue;
    const papers = parseDailyReportPapers(await readFile(path.join(dailyPath, name), "utf8"));
    // The pre-June schema also had a "主要结果" bullet, so only labels it never had mark a report as five-field.
    const currentSchema = papers.some((paper) => !paper.fallback && (paper.summary?.coreProblem || paper.summary?.keyMethod));
    let evaluable = false;
    for (const paper of papers) {
      if (paper.fallback) {
        evaluable = true;
        corpus.fallbacks.push({ date, topic: paper.section, id: paper.id, title: paper.title });
      } else if (currentSchema) {
        evaluable = true;
        const fields = readFields(paper.summary ?? {});
        if (FIELD_KEYS.some((key) => !fields[key])) corpus.counts.incompleteEntries += 1;
        corpus.entries.push({ date, topic: paper.section, id: paper.id, title: paper.title, authors: paper.authors, fields });
      }
    }
    corpus.counts.reports += 1;
    if (evaluable) corpus.counts.evaluableReports += 1;
    else {
      corpus.counts.legacyReports += 1;
      corpus.legacyReportDates.push(date);
    }
  }

  const papersPath = path.join(options.vaultRoot, options.papersDir ?? DEFAULT_PAPERS_DIR);
  for (const name of await listMarkdown(papersPath)) {
    const file = path.join(papersPath, name);
    const markdown = await readFile(file, "utf8");
    const note = readDetailNote(markdown, path.relative(options.vaultRoot, file));
    if (note) {
      if (inRange(note.date)) corpus.notes.push(note);
    } else if (inRange(legacyNoteDate(markdown))) {
      corpus.counts.legacyNotes += 1;
    }
  }

  corpus.counts.entries = corpus.entries.length;
  corpus.counts.fallbacks = corpus.fallbacks.length;
  corpus.counts.notes = corpus.notes.length;
  return corpus;
}

function checkDate(value: string | undefined, flag: string): string | undefined {
  if (value === undefined) return undefined;
  const parsed = new Date(`${value}T00:00:00Z`);
  const valid = DATE_RE.test(value) && !Number.isNaN(parsed.getTime()) && parsed.toISOString().startsWith(value);
  if (!valid) throw new Error(`${flag} must be a real date written YYYY-MM-DD, got "${value}".`);
  return value;
}

async function listMarkdown(directory: string): Promise<string[]> {
  try {
    const entries = await readdir(directory, { withFileTypes: true });
    return entries.filter((entry) => entry.isFile() && entry.name.endsWith(".md")).map((entry) => entry.name).sort();
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

function readFields(summary: PaperSummary): Record<FieldKey, string> {
  return Object.fromEntries(FIELD_KEYS.map((key) => [key, summary[key] ?? ""])) as Record<FieldKey, string>;
}

/** The note in the current format, or null for anything older or not a generated detail note. */
function readDetailNote(markdown: string, notePath: string): DetailNote | null {
  const { values, body } = splitFrontmatter(markdown);
  const id = values.get("arxiv_id");
  const topic = values.get("primary_topic");
  if (!id || !topic) return null;
  const parsed = parseBody(stripGenerationMetrics(body));
  if (!parsed.sections.some((section) => CURRENT_NOTE_HEADINGS.has(section.heading))) return null;
  return { id, topic, path: notePath, title: parsed.title, authors: values.get("authors") ?? "", date: noteDate(values.get("published")), sections: parsed.sections };
}

function legacyNoteDate(markdown: string): string | null {
  const { values } = splitFrontmatter(markdown);
  return noteDate(values.get("published")) ?? noteDate(values.get("date"));
}

/** A linked report date wins over the bare value, because `published` may be the paper's own date. */
function noteDate(value: string | undefined): string | null {
  if (!value) return null;
  return /daily\/(\d{4}-\d{2}-\d{2})/.exec(value)?.[1] ?? /^\d{4}-\d{2}-\d{2}$/.exec(value)?.[0] ?? null;
}

function splitFrontmatter(markdown: string): { values: Map<string, string>; body: string } {
  const values = new Map<string, string>();
  const match = /^---\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)([\s\S]*)$/.exec(markdown);
  if (!match) return { values, body: markdown };
  for (const line of (match[1] ?? "").split(/\r?\n/)) {
    const entry = /^([A-Za-z_][\w-]*):(.*)$/.exec(line);
    if (entry) values.set(entry[1]!, decodeScalar(entry[2]!.trim()));
  }
  return { values, body: match[2] ?? "" };
}

function decodeScalar(raw: string): string {
  if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) return raw.slice(1, -1).replace(/\\(["\\])/g, "$1");
  if (raw.length >= 2 && raw.startsWith("'") && raw.endsWith("'")) return raw.slice(1, -1).replace(/''/g, "'");
  return raw;
}

/** Split a note body into its `#` title and ordered `##` sections, ignoring headings inside code fences. */
export function parseBody(body: string): { title: string; sections: NoteSection[] } {
  let title = "";
  const sections: Array<{ heading: string; lines: string[] }> = [];
  let fence: { char: string; length: number } | null = null;
  for (const line of body.split(/\r?\n/)) {
    const fenced = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (fenced) {
      const run = fenced[1]!;
      if (!fence) fence = { char: run[0]!, length: run.length };
      else if (run[0] === fence.char && run.length >= fence.length && !fenced[2]!.trim()) fence = null;
    } else if (!fence) {
      const heading = /^(#{1,2})[ \t]+(.*?)[ \t]*$/.exec(line);
      if (heading?.[1] === "##") {
        sections.push({ heading: heading[2]!, lines: [] });
        continue;
      }
      if (heading && !title && sections.length === 0) {
        title = heading[2]!;
        continue;
      }
    }
    sections.at(-1)?.lines.push(line);
  }
  return { title, sections: sections.map(({ heading, lines }) => ({ heading, body: lines.join("\n").trim() })) };
}
