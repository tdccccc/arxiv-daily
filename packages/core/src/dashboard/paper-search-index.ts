import type { PaperIndexEntry } from "../services/paper-index";

export const PAPER_SEARCH_FIELD_WEIGHTS = {
  title: 5,
  topics: 4,
  categories: 2.5,
  authors: 2,
  coreProblem: 1.8,
  keyMethod: 1.8,
  mainResult: 1.7,
  whyRelevant: 1.4,
  limitations: 0.8,
  sourceSections: 0.6,
} as const;

type SearchField = keyof typeof PAPER_SEARCH_FIELD_WEIGHTS;

export interface PaperSearchReason {
  field: SearchField | "arxivId";
  terms: string[];
  text: string;
}

export interface PaperSearchResult {
  entry: PaperIndexEntry;
  score: number;
  reasons: PaperSearchReason[];
}

export interface SimilarPaperOptions {
  limit?: number;
  includeIgnored?: boolean;
}

interface IndexedDocument {
  entry: PaperIndexEntry;
  fields: Record<SearchField, Map<string, number>>;
  lengths: Record<SearchField, number>;
  allTokens: Set<string>;
  canonicalId: string;
}

interface QueryClause {
  display: string;
  tokens: string[];
}

const FIELD_LABELS: Record<SearchField, string> = {
  title: "title",
  topics: "topic",
  categories: "category",
  authors: "author",
  coreProblem: "core problem",
  keyMethod: "method",
  mainResult: "result",
  whyRelevant: "relevance",
  limitations: "limitations",
  sourceSections: "source sections",
};

const SEARCH_FIELDS = Object.keys(PAPER_SEARCH_FIELD_WEIGHTS) as SearchField[];
const HAN_RE = /\p{Script=Han}/u;
const TOKEN_PART_RE = /[\p{Script=Han}]+|[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*/gu;
const MODERN_ARXIV_ID_RE = /^(\d{4}\.\d{4,5})(?:v\d+)?$/i;

/** Deterministic, locale-independent tokenization for English technical text and Han text. */
export function tokenizePaperSearchText(value: string): string[] {
  const normalized = value.normalize("NFKC").toLowerCase();
  const tokens: string[] = [];
  for (const match of normalized.matchAll(TOKEN_PART_RE)) {
    const part = match[0];
    if (HAN_RE.test(part)) {
      const chars = Array.from(part);
      if (chars.length === 1) tokens.push(part);
      for (let i = 0; i + 1 < chars.length; i += 1) {
        tokens.push(`${chars[i]}${chars[i + 1]}`);
      }
      continue;
    }
    tokens.push(part);
    if (part.includes("-")) {
      tokens.push(...part.split("-").filter(Boolean));
    }
  }
  return tokens;
}

/** Extract a canonical modern arXiv ID, stripping prefixes, URL forms, PDF suffixes and versions. */
export function normalizeArxivSearchId(value: string): string | null {
  let candidate = value.normalize("NFKC").trim().toLowerCase();
  candidate = candidate.replace(/^arxiv\s*:\s*/i, "");
  try {
    const url = new URL(candidate);
    if (url.hostname === "arxiv.org" || url.hostname.endsWith(".arxiv.org")) {
      candidate = url.pathname.replace(/^\/(?:abs|pdf)\//i, "");
    }
  } catch {
    candidate = candidate.replace(/^https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\//i, "");
  }
  candidate = (candidate.split(/[?#]/, 1)[0] ?? "").replace(/\.pdf$/i, "").replace(/^\/+|\/+$/g, "");
  const match = MODERN_ARXIV_ID_RE.exec(candidate);
  return match?.[1] ?? null;
}

export class PaperSearchIndex {
  private readonly documents: IndexedDocument[];
  private readonly documentFrequency = new Map<string, number>();
  private readonly averageLengths: Record<SearchField, number>;

  constructor(entries: readonly PaperIndexEntry[]) {
    this.documents = entries.map((entry) => indexDocument(entry));
    this.averageLengths = emptyFieldNumbers();
    for (const document of this.documents) {
      for (const token of document.allTokens) {
        this.documentFrequency.set(token, (this.documentFrequency.get(token) ?? 0) + 1);
      }
      for (const field of SEARCH_FIELDS) {
        this.averageLengths[field] += document.lengths[field];
      }
    }
    for (const field of SEARCH_FIELDS) {
      this.averageLengths[field] = this.documents.length
        ? this.averageLengths[field] / this.documents.length
        : 1;
    }
  }

  search(query: string): PaperSearchResult[] {
    const clauses = queryClauses(query);
    if (clauses.length === 0) return [];
    const canonicalQueryId = normalizeArxivSearchId(query);
    const partialId = partialArxivId(query);
    const results: PaperSearchResult[] = [];

    for (const document of this.documents) {
      const matched = clauses.every((clause) =>
        clause.tokens.some((token) => document.allTokens.has(token)),
      );
      const idMatch = idMatchRank(document.canonicalId, canonicalQueryId, partialId);
      if (!matched && idMatch === 0) continue;
      results.push(this.scoreDocument(document, clauses.flatMap((clause) => clause.tokens), idMatch));
    }
    return sortResults(results);
  }

  similar(source: PaperIndexEntry | string, options: SimilarPaperOptions = {}): PaperSearchResult[] {
    const sourceDocument = typeof source === "string"
      ? this.documents.find((document) => document.canonicalId === normalizeArxivSearchId(source))
      : this.documents.find((document) => document.entry === source || document.canonicalId === normalizeArxivSearchId(source.arxivId));
    if (!sourceDocument) return [];

    const terms = this.similarityTerms(sourceDocument, 16);
    const results: PaperSearchResult[] = [];
    for (const document of this.documents) {
      if (document === sourceDocument || document.canonicalId === sourceDocument.canonicalId) continue;
      if (!options.includeIgnored && document.entry.status === "ignored") continue;
      const matchedTerms = terms.filter((term) => document.allTokens.has(term));
      if (matchedTerms.length === 0) continue;
      results.push(this.scoreDocument(document, matchedTerms, 0));
    }
    return sortResults(results).slice(0, Math.max(0, options.limit ?? 10));
  }

  private similarityTerms(source: IndexedDocument, limit: number): string[] {
    const candidates = new Map<string, number>();
    for (const field of SEARCH_FIELDS) {
      if (field === "limitations" || field === "sourceSections") continue;
      for (const [token, frequency] of source.fields[field]) {
        if (token.length < 2 || /^\d+$/.test(token)) continue;
        const rarity = inverseDocumentFrequency(this.documents.length, this.documentFrequency.get(token) ?? 0);
        const value = PAPER_SEARCH_FIELD_WEIGHTS[field] * rarity * (1 + Math.log(frequency));
        candidates.set(token, Math.max(candidates.get(token) ?? 0, value));
      }
    }
    return [...candidates]
      .sort((a, b) => b[1] - a[1] || compareText(a[0], b[0]))
      .slice(0, limit)
      .map(([token]) => token);
  }

  private scoreDocument(document: IndexedDocument, terms: string[], idRank: number): PaperSearchResult {
    const uniqueTerms = [...new Set(terms)];
    let score = idRank;
    const reasons: PaperSearchReason[] = [];
    if (idRank > 0) {
      reasons.push({ field: "arxivId", terms: [document.canonicalId], text: idRank >= 1_000_000 ? "Exact arXiv ID" : "Partial arXiv ID" });
    }
    for (const field of SEARCH_FIELDS) {
      const matched: string[] = [];
      let fieldScore = 0;
      for (const term of uniqueTerms) {
        const frequency = document.fields[field].get(term) ?? 0;
        if (!frequency) continue;
        matched.push(term);
        const averageLength = Math.max(1, this.averageLengths[field]);
        const length = document.lengths[field];
        const normalizedTf = (frequency * 2.2) / (frequency + 1.2 * (0.25 + 0.75 * length / averageLength));
        fieldScore += PAPER_SEARCH_FIELD_WEIGHTS[field]
          * inverseDocumentFrequency(this.documents.length, this.documentFrequency.get(term) ?? 0)
          * normalizedTf;
      }
      if (matched.length) {
        score += fieldScore;
        reasons.push({
          field,
          terms: matched.sort(compareText),
          text: `Matched ${FIELD_LABELS[field]}: ${matched.sort(compareText).join(", ")}`,
        });
      }
    }
    reasons.sort((a, b) => reasonWeight(b.field) - reasonWeight(a.field) || compareText(a.text, b.text));
    return { entry: document.entry, score: roundScore(score), reasons };
  }
}

function indexDocument(entry: PaperIndexEntry): IndexedDocument {
  const topics = uniqueNormalized([entry.primaryTopic, ...entry.topics]);
  const categories = uniqueNormalized([entry.category, ...(entry.categories ?? [])]);
  const values: Record<SearchField, string[]> = {
    title: [entry.title],
    topics,
    categories,
    authors: entry.authors,
    coreProblem: [entry.summary?.coreProblem ?? ""],
    keyMethod: [entry.summary?.keyMethod ?? ""],
    mainResult: [entry.summary?.mainResult ?? ""],
    whyRelevant: [entry.summary?.whyRelevant ?? ""],
    limitations: [entry.summary?.limitations ?? ""],
    sourceSections: [entry.summary?.sourceSections ?? ""],
  };
  const fields = {} as Record<SearchField, Map<string, number>>;
  const lengths = emptyFieldNumbers();
  const allTokens = new Set<string>();
  for (const field of SEARCH_FIELDS) {
    const tokens = values[field].flatMap(tokenizePaperSearchText);
    fields[field] = termFrequency(tokens);
    lengths[field] = tokens.length;
    for (const token of tokens) allTokens.add(token);
  }
  const canonicalId = normalizeArxivSearchId(entry.arxivId) ?? entry.arxivId.normalize("NFKC").toLowerCase();
  allTokens.add(canonicalId);
  return { entry, fields, lengths, allTokens, canonicalId };
}

function queryClauses(query: string): QueryClause[] {
  const normalizedId = normalizeArxivSearchId(query);
  if (normalizedId) return [{ display: normalizedId, tokens: [normalizedId] }];
  const normalized = query.normalize("NFKC").toLowerCase();
  const clauses: QueryClause[] = [];
  for (const match of normalized.matchAll(TOKEN_PART_RE)) {
    const part = match[0];
    if (HAN_RE.test(part)) {
      const tokens = tokenizePaperSearchText(part);
      for (const token of tokens) clauses.push({ display: token, tokens: [token] });
    } else {
      const tokens = part.includes("-") ? [part, ...part.split("-").filter(Boolean)] : [part];
      clauses.push({ display: part, tokens: [...new Set(tokens)] });
    }
  }
  return clauses;
}

function partialArxivId(query: string): string | null {
  let value = query.normalize("NFKC").trim().toLowerCase();
  value = value.replace(/^arxiv\s*:\s*/i, "").replace(/^https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\//i, "");
  value = (value.split(/[?#]/, 1)[0] ?? "").replace(/\.pdf$/i, "").replace(/v\d+$/i, "");
  return /^\d{2,4}(?:\.\d{0,5})?$/.test(value) && value.length >= 4 ? value : null;
}

function idMatchRank(documentId: string, canonicalId: string | null, partialId: string | null): number {
  if (canonicalId && documentId === canonicalId) return 1_000_000;
  if (partialId && documentId.includes(partialId)) return 100_000 + partialId.length * 100;
  return 0;
}

function inverseDocumentFrequency(total: number, frequency: number): number {
  return Math.log(1 + (total - frequency + 0.5) / (frequency + 0.5));
}

function sortResults(results: PaperSearchResult[]): PaperSearchResult[] {
  return results.sort((a, b) => b.score - a.score || compareText(a.entry.arxivId, b.entry.arxivId));
}

function uniqueNormalized(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = value.normalize("NFKC").trim();
    const key = normalized.toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(normalized);
  }
  return out;
}

function termFrequency(tokens: string[]): Map<string, number> {
  const frequencies = new Map<string, number>();
  for (const token of tokens) frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
  return frequencies;
}

function emptyFieldNumbers(): Record<SearchField, number> {
  return {
    title: 0,
    topics: 0,
    categories: 0,
    authors: 0,
    coreProblem: 0,
    keyMethod: 0,
    mainResult: 0,
    whyRelevant: 0,
    limitations: 0,
    sourceSections: 0,
  };
}

function reasonWeight(field: PaperSearchReason["field"]): number {
  return field === "arxivId" ? 1_000 : PAPER_SEARCH_FIELD_WEIGHTS[field];
}

function roundScore(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
