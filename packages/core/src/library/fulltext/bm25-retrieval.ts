import { createEvidenceChunkId } from "./evidence-chunk";
import { LEGACY_EVIDENCE_DERIVATION, type FullTextPaperDocument } from "./knowledge-base";
import type { KnowledgeBaseChunkHit, KnowledgeBasePaperMatch } from "./retrieval";

const DEFAULT_LIMIT = 10;
const DEFAULT_MAX_HITS_PER_PAPER = 3;
const DEFAULT_K1 = 1.2;
const DEFAULT_B = 0.75;
const WORD_OR_HAN_RUN = /[\p{Script=Han}]+|[\p{L}\p{N}\p{M}]+/gu;
const HAN_RUN = /^\p{Script=Han}+$/u;
const SHORT_ALIAS_MAX_CHARS = 80;

export interface Bm25RetrievalStats {
  passes: number;
  chunksScanned: number;
  peakPaperCandidates: number;
  peakHitsPerPaper: number;
}

export interface SearchKnowledgeBaseBm25Input {
  papers: readonly FullTextPaperDocument[];
  queryText: string;
  titles?: ReadonlyMap<string, string>;
  limit?: number;
  maxHitsPerPaper?: number;
  k1?: number;
  b?: number;
  stats?: Bm25RetrievalStats;
}

interface RankedPaper extends KnowledgeBasePaperMatch {
  lexicalPriority: number;
}

/** Pure Unicode tokenizer: NFKC/case-folded words and overlapping Han bigrams. */
export function tokenizeUnicode(text: string): string[] {
  return tokenize(text, false);
}

/** Builder-only expansion used to precompute the single-Han query token stream. */
export function tokenizeUnicodeWithHanSingles(text: string): string[] {
  return tokenize(text, true);
}

function tokenize(text: string, includeHanSingles: boolean): string[] {
  const normalized = text.normalize("NFKC").toLocaleLowerCase("und");
  const tokens: string[] = [];
  for (const match of normalized.matchAll(WORD_OR_HAN_RUN)) {
    const run = match[0]!;
    if (!HAN_RUN.test(run)) {
      tokens.push(run);
      continue;
    }
    const characters = Array.from(run);
    if (characters.length === 1) {
      tokens.push(characters[0]!);
      continue;
    }
    for (let index = 0; index + 1 < characters.length; index += 1) {
      tokens.push(characters[index]! + characters[index + 1]!);
    }
    if (includeHanSingles) tokens.push(...characters);
  }
  return tokens;
}

/**
 * Query-term-only two-pass chunk BM25. It retains only bounded paper/hit top-k
 * arrays; tokenized chunks, a corpus vocabulary, and joined paper text are never
 * retained. Each paper is aggregated before entering the paper top-k.
 */
export function searchKnowledgeBaseBm25(input: SearchKnowledgeBaseBm25Input): KnowledgeBasePaperMatch[] {
  const limit = nonNegativeInteger(input.limit, "limit", DEFAULT_LIMIT);
  const maxHitsPerPaper = positiveInteger(input.maxHitsPerPaper, "maxHitsPerPaper", DEFAULT_MAX_HITS_PER_PAPER);
  const k1 = finiteNonNegative(input.k1, "k1", DEFAULT_K1);
  const b = finiteRange(input.b, "b", DEFAULT_B);
  resetStats(input.stats);
  if (limit === 0 || input.papers.length === 0) return [];

  const queryTokens = tokenizeUnicode(input.queryText);
  if (queryTokens.length === 0) return [];
  const queryFrequency = frequencies(queryTokens);
  const queryTerms = [...queryFrequency.keys()];
  const includeHanSingles = queryTerms.some((term) => Array.from(term).length === 1 && HAN_RUN.test(term));
  const compactAlias = input.queryText.trim().length <= SHORT_ALIAS_MAX_CHARS
    ? compactUnicode(input.queryText)
    : "";
  const documentFrequency = new Map(queryTerms.map((term) => [term, 0]));
  let chunkCount = 0;
  let totalDocumentLength = 0;

  input.stats && (input.stats.passes = 1);
  for (const paper of input.papers) {
    for (const chunk of paper.chunks) {
      const tokens = tokenize(chunk.text, includeHanSingles);
      chunkCount += 1;
      totalDocumentLength += tokens.length;
      const present = new Set<string>();
      for (const token of tokens) if (queryFrequency.has(token)) present.add(token);
      for (const term of present) documentFrequency.set(term, documentFrequency.get(term)! + 1);
      if (input.stats) input.stats.chunksScanned += 1;
    }
  }
  if (chunkCount === 0) return [];

  const averageDocumentLength = totalDocumentLength / chunkCount || 1;
  const paperTop: RankedPaper[] = [];
  input.stats && (input.stats.passes = 2);
  for (const paper of input.papers) {
    const hitTop: KnowledgeBaseChunkHit[] = [];
    let bestBm25 = 0;
    for (const chunk of paper.chunks) {
      const tokens = tokenize(chunk.text, includeHanSingles);
      const termFrequency = new Map<string, number>();
      for (const token of tokens) {
        if (queryFrequency.has(token)) termFrequency.set(token, (termFrequency.get(token) ?? 0) + 1);
      }
      let score = 0;
      for (const term of queryTerms) {
        const tf = termFrequency.get(term) ?? 0;
        if (tf === 0) continue;
        const df = documentFrequency.get(term) ?? 0;
        const idf = Math.log(1 + (chunkCount - df + 0.5) / (df + 0.5));
        const lengthNorm = 1 - b + b * tokens.length / averageDocumentLength;
        score += queryFrequency.get(term)! * idf * (tf * (k1 + 1)) / (tf + k1 * lengthNorm);
      }
      if (score === 0 && compactAlias && compactUnicode(chunk.text).includes(compactAlias)) score = Number.EPSILON;
      if (score > 0) {
        bestBm25 = Math.max(bestBm25, score);
        insertBounded(hitTop, chunkHit(paper, chunk.index, score), maxHitsPerPaper, compareHits);
      }
      if (input.stats) input.stats.chunksScanned += 1;
    }

    const title = input.titles?.get(paper.paperKey) ?? paper.title;
    const lexicalPriority = title ? titlePriority(input.queryText, title) : 0;
    if (bestBm25 === 0 && lexicalPriority === 0) continue;
    if (hitTop.length === 0 && paper.chunks.length > 0) {
      insertBounded(hitTop, chunkHit(paper, paper.chunks[0]!.index, 0), maxHitsPerPaper, compareHits);
    }
    if (hitTop.length === 0) continue;
    insertBounded(paperTop, {
      paperKey: paper.paperKey,
      score: bestBm25,
      scoreKind: "bm25",
      rankingScore: bestBm25,
      rankingScoreKind: "bm25",
      hits: hitTop,
      chunkCount: paper.chunks.length,
      lexicalPriority,
    }, limit, comparePapers);
    if (input.stats) {
      input.stats.peakPaperCandidates = Math.max(input.stats.peakPaperCandidates, paperTop.length);
      input.stats.peakHitsPerPaper = Math.max(input.stats.peakHitsPerPaper, hitTop.length);
    }
  }
  return paperTop.map(({ lexicalPriority: _priority, ...match }) => match);
}

export function titlePriority(query: string, title: string): number {
  const queryWords = tokenizeUnicode(query);
  const titleWords = tokenizeUnicode(title);
  if (queryWords.length === 0 || titleWords.length === 0) return 0;
  const queryNormalized = queryWords.join(" ");
  const titleNormalized = titleWords.join(" ");
  const queryCompact = compactUnicode(query);
  const titleCompact = compactUnicode(title);
  if (queryNormalized === titleNormalized || queryCompact === titleCompact) return 3;
  if (query.trim().length <= SHORT_ALIAS_MAX_CHARS && titleCompact.startsWith(queryCompact)) return 2;
  if (titleWords.length > queryWords.length && titleWords.slice(0, queryWords.length).join(" ") === queryNormalized) return 2;
  if (query.trim().length <= SHORT_ALIAS_MAX_CHARS && titleCompact.includes(queryCompact)) return 1;
  return 0;
}

function compactUnicode(text: string): string {
  return text.normalize("NFKC").toLocaleLowerCase("und").replace(/[^\p{L}\p{N}]+/gu, "");
}

function chunkHit(paper: FullTextPaperDocument, chunkIndex: number, score: number): KnowledgeBaseChunkHit {
  const chunk = paper.chunks.find((candidate) => candidate.index === chunkIndex)!;
  const locator = chunk.locator ?? { pageStart: chunk.page };
  const headings = chunk.headings ?? [];
  const derivation = chunk.derivation ?? paper.derivation ?? LEGACY_EVIDENCE_DERIVATION;
  return {
    source: "lexical",
    scoreKind: "bm25",
    chunkIndex: chunk.index,
    chunkId: chunk.id ?? createEvidenceChunkId({ text: chunk.text, headings, locator, derivation }),
    headings,
    locator,
    page: chunk.page,
    text: chunk.text,
    score,
  };
}

function compareHits(left: KnowledgeBaseChunkHit, right: KnowledgeBaseChunkHit): number {
  if (right.score !== left.score) return right.score - left.score;
  return left.chunkIndex - right.chunkIndex;
}

function comparePapers(left: RankedPaper, right: RankedPaper): number {
  if (right.lexicalPriority !== left.lexicalPriority) return right.lexicalPriority - left.lexicalPriority;
  if (right.score !== left.score) return right.score - left.score;
  return left.paperKey.localeCompare(right.paperKey);
}

function insertBounded<T>(values: T[], value: T, limit: number, compare: (left: T, right: T) => number): void {
  let index = values.findIndex((current) => compare(value, current) < 0);
  if (index < 0) index = values.length;
  if (index >= limit) return;
  values.splice(index, 0, value);
  if (values.length > limit) values.pop();
}

function frequencies(tokens: readonly string[]): Map<string, number> {
  const result = new Map<string, number>();
  for (const token of tokens) result.set(token, (result.get(token) ?? 0) + 1);
  return result;
}

function resetStats(stats: Bm25RetrievalStats | undefined): void {
  if (!stats) return;
  stats.passes = 0;
  stats.chunksScanned = 0;
  stats.peakPaperCandidates = 0;
  stats.peakHitsPerPaper = 0;
}

function positiveInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(`searchKnowledgeBaseBm25: ${name} must be a positive integer`);
  return value;
}

function nonNegativeInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`searchKnowledgeBaseBm25: ${name} must be a non-negative integer`);
  return value;
}

function finiteNonNegative(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value < 0) throw new TypeError(`searchKnowledgeBaseBm25: ${name} must be non-negative`);
  return value;
}

function finiteRange(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new TypeError(`searchKnowledgeBaseBm25: ${name} must be in [0, 1]`);
  return value;
}
