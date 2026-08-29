import { titlePriority, tokenizeUnicode } from "./bm25-retrieval";
import type { EvidenceBlock, GenerationObjectReference, LexicalNamespace, LexicalPostingsBlock, PaperMetadataRecord } from "./generation-index-format";
import { lexicalTermBucket,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
} from "./generation-index-format";
import { FullTextGenerationIndexStoreError, type OpenedFullTextGeneration } from "./generation-index-store";
import type { KnowledgeBaseChunkHit, KnowledgeBasePaperMatch } from "./retrieval";

const DEFAULT_LIMIT = 10;
const DEFAULT_MAX_HITS = 3;
const DEFAULT_K1 = 1.2;
const DEFAULT_B = 0.75;
const MAX_LIMIT = 1_000;
const MAX_HITS = 100;
const MAX_QUERY_BYTES = 64 * 1024;
const MAX_QUERY_TERMS = 64;
const MAX_ROUTE_REFS = MAX_QUERY_TERMS * 4096;
const SHORT_ALIAS_MAX_CHARS = 80;
const HAN_SINGLE = /^\p{Script=Han}$/u;

export interface GenerationBm25Stats {
  dictionaryReads: number;
  postingsReads: number;
  /** Backward-compatible spelling retained for callers compiled against the initial stub. */
  postingReads?: number;
  metadataReads: number;
  evidenceReads: number;
  peakChunkAccumulators: number;
  peakCandidates: number;
  peakHits: number;
  totalRetainedHits: number;
  maxLiveBlocks: number;
}
export interface SearchGenerationBm25Input {
  readonly generation: OpenedFullTextGeneration; readonly queryText: string; readonly limit?: number;
  readonly maxHitsPerPaper?: number; readonly k1?: number; readonly b?: number;
  /** Query-time catalog titles override persisted generation metadata. */
  readonly titles?: ReadonlyMap<string, string>;
  readonly signal?: AbortSignal; readonly stats?: GenerationBm25Stats;
}

interface QueryTarget { namespace: LexicalNamespace; term: string; queryIndex: number | null }
interface HitCandidate { chunkOrdinal: number; chunkIndex: number; score: number }
interface PaperCandidate { ordinal: number; paperKey: string; chunkCount: number; score: number; lexicalPriority: number; hits: HitCandidate[] }

/** Streaming BM25 reader over one pinned immutable schema-v4 generation. */
export async function searchGenerationBm25(input: SearchGenerationBm25Input): Promise<KnowledgeBasePaperMatch[]> {
  const limit = boundedInteger(input.limit, "limit", DEFAULT_LIMIT, 0, MAX_LIMIT);
  const maxHits = boundedInteger(input.maxHitsPerPaper, "maxHitsPerPaper", DEFAULT_MAX_HITS, 1, MAX_HITS);
  const k1 = finiteRange(input.k1, "k1", DEFAULT_K1, 0, Number.POSITIVE_INFINITY);
  const b = finiteRange(input.b, "b", DEFAULT_B, 0, 1);
  if (typeof input.queryText !== "string" || new TextEncoder().encode(input.queryText).byteLength > MAX_QUERY_BYTES) {
    throw new TypeError(`searchGenerationBm25: queryText must be UTF-8 text no longer than ${MAX_QUERY_BYTES} bytes`);
  }
  resetStats(input.stats);
  throwIfCancelled(input.signal);
  const descriptor = input.generation.descriptor;
  if (descriptor.schemaVersion !== GENERATION_DESCRIPTOR_SCHEMA_VERSION || descriptor.lexicalCapability !== "bm25-v1") {
    throw new FullTextGenerationIndexStoreError("generation BM25 is unavailable for this generation", "capability-unsupported");
  }
  if (limit === 0 || descriptor.corpusStats.chunkCount === 0) return [];

  const queryTokens = tokenizeUnicode(input.queryText);
  if (queryTokens.length === 0) return [];
  const queryFrequency = new Map<string, number>();
  for (const token of queryTokens) {
    if (!queryFrequency.has(token) && queryFrequency.size === MAX_QUERY_TERMS) {
      throw new TypeError(`searchGenerationBm25: query has more than ${MAX_QUERY_TERMS} unique terms`);
    }
    queryFrequency.set(token, (queryFrequency.get(token) ?? 0) + 1);
  }
  const queryTerms = [...queryFrequency.keys()];
  const namespace: LexicalNamespace = queryTerms.some((term) => HAN_SINGLE.test(term)) ? "expanded" : "base";
  const compactAlias = input.queryText.trim().length <= SHORT_ALIAS_MAX_CHARS ? compactUnicode(input.queryText) : "";
  const aliasSelector = selectAliasGram(compactAlias);
  const targets: QueryTarget[] = queryTerms.map((term, queryIndex) => ({ namespace, term, queryIndex }));
  if (aliasSelector) targets.push({ namespace: "alias", term: aliasSelector, queryIndex: null });

  const postingRefs = refsOfKind(descriptor.objects, "lexical-postings");
  const dictionaryRefs = refsOfKind(descriptor.objects, "lexical-dictionary");
  const routedOrdinals = new Set<number>();
  for (const target of targets) for (const ordinal of descriptor.lexicalRouting[lexicalTermBucket(target.namespace, target.term)]!) routedOrdinals.add(ordinal);
  const routedDictionaries = [...routedOrdinals].map((ordinal) => {
    const reference = dictionaryRefs[ordinal];
    if (!reference) throw corruption("lexical routing references an unknown dictionary");
    return { ordinal, reference };
  }).sort((left, right) => left.reference.recordStart - right.reference.recordStart || left.reference.path.localeCompare(right.reference.path));
  const df = new Array<number>(queryTerms.length).fill(0);
  const postingTargets = new Map<number, Set<number>>();
  let routeRefCount = 0;
  for (const { ordinal, reference } of routedDictionaries) {
    throwIfCancelled(input.signal);
    const object = await input.generation.readLexicalDictionary(reference);
    if (input.stats) input.stats.dictionaryReads += 1;
    afterRead(input.signal);
    const block = object.block;
    for (const target of targets) {
      const bucket = lexicalTermBucket(target.namespace, target.term);
      if (!descriptor.lexicalRouting[bucket]!.includes(ordinal)) continue;
      for (const catalogIndex of block.queryCatalog) {
        const entry = block.entries[catalogIndex]!;
        const entryBucket = lexicalTermBucket(entry.namespace, entry.term);
        if (entryBucket < bucket) continue;
        if (entryBucket > bucket) break;
        if (entry.namespace !== target.namespace || entry.term !== target.term) continue;
        routeRefCount += 1;
        if (routeRefCount > MAX_ROUTE_REFS) throw corruption("generation BM25 route reference cap exceeded");
        let refs = postingTargets.get(entry.postingOrdinal); if (!refs) postingTargets.set(entry.postingOrdinal, refs = new Set());
        refs.add(target.queryIndex ?? queryTerms.length);
        if (target.queryIndex !== null) df[target.queryIndex]! += entry.chunkDf;
      }
    }
    await yieldToTimer(input.signal);
  }

  const routedPostings = [...postingTargets.keys()].sort((left, right) => left - right);
  for (const ordinal of routedPostings) if (!postingRefs[ordinal]) throw corruption("dictionary route points outside postings objects");
  const avgdl = namespace === "expanded" ? descriptor.corpusStats.avgdlWithHanSingles : descriptor.corpusStats.avgdl;
  const topPapers: PaperCandidate[] = [];
  let retainedHits = 0;
  let routedIndex = 0;
  let activePosting: { ordinal: number; block: LexicalPostingsBlock } | null = null;
  let activeMetadata: object | null = null;
  for (const reference of refsOfKind(descriptor.objects, "paper-metadata")) {
    throwIfCancelled(input.signal);
    const object = await input.generation.readPaperMetadata(reference); activeMetadata = object.block;
    if (input.stats) input.stats.metadataReads += 1;
    afterRead(input.signal);
    observeLive(input.stats, activeMetadata, activePosting?.block ?? null);
    for (const metadata of object.block.records) {
      const hits: HitCandidate[] = [];
      let best = 0;
      const paperEnd = metadata.chunkStart + metadata.chunkCount;
      while (true) {
        if (activePosting === null) {
          while (routedIndex < routedPostings.length
            && postingRefs[routedPostings[routedIndex]!]!.recordStart + postingRefs[routedPostings[routedIndex]!]!.recordCount <= metadata.chunkStart) routedIndex += 1;
          const ordinal = routedPostings[routedIndex];
          if (ordinal === undefined || postingRefs[ordinal]!.recordStart >= paperEnd) break;
          throwIfCancelled(input.signal);
          const postingObject = await input.generation.readLexicalPostings(postingRefs[ordinal]!);
          if (input.stats) { input.stats.postingsReads += 1; input.stats.postingReads = input.stats.postingsReads; }
          afterRead(input.signal);
          activePosting = { ordinal, block: postingObject.block };
          observeLive(input.stats, activeMetadata, activePosting.block);
        }
        const blockEnd = activePosting.block.chunkStart + activePosting.block.chunks.length;
        const blockHits = scorePostingsRange(activePosting.block, postingTargets.get(activePosting.ordinal)!, queryTerms, queryFrequency, df, descriptor.corpusStats.chunkCount, avgdl || 1, namespace, compactAlias, metadata, k1, b);
        if (input.stats) input.stats.peakChunkAccumulators = Math.max(input.stats.peakChunkAccumulators, blockHits.length);
        for (const hit of blockHits) {
          best = Math.max(best, hit.score);
          insertBounded(hits, hit, maxHits, compareHits);
          if (input.stats) input.stats.peakHits = Math.max(input.stats.peakHits, retainedHits + hits.length);
        }
        if (blockEnd <= paperEnd) { activePosting = null; routedIndex += 1; await yieldToTimer(input.signal); }
        if (blockEnd >= paperEnd) break;
      }
      const title = input.titles?.get(metadata.paperKey) ?? metadata.title;
      const priority = title === undefined ? 0 : titlePriority(input.queryText, title);
      if (best === 0 && priority === 0) continue;
      if (hits.length === 0) hits.push({ chunkOrdinal: metadata.chunkStart, chunkIndex: 0, score: 0 });
      if (input.stats) input.stats.peakHits = Math.max(input.stats.peakHits, retainedHits + hits.length);
      const candidate: PaperCandidate = { ordinal: metadata.paperOrdinal, paperKey: metadata.paperKey, chunkCount: metadata.chunkCount, score: best, lexicalPriority: priority, hits };
      const change = insertBounded(topPapers, candidate, limit, comparePapers);
      if (change.inserted) retainedHits += hits.length - (change.removed?.hits.length ?? 0);
      if (input.stats) { input.stats.peakCandidates = Math.max(input.stats.peakCandidates, topPapers.length); input.stats.peakHits = Math.max(input.stats.peakHits, retainedHits); input.stats.totalRetainedHits = retainedHits; }
    }
    activeMetadata = null;
    await yieldToTimer(input.signal);
  }
  if (routedIndex !== routedPostings.length) throw corruption("routed postings chunks fall outside metadata coverage");

  const selectedRows = new Map<number, { paper: PaperCandidate; hit: HitCandidate }>();
  for (const paper of topPapers) for (const hit of paper.hits) selectedRows.set(hit.chunkOrdinal, { paper, hit });
  const materialized = new Map<number, KnowledgeBaseChunkHit>();
  for (const reference of refsOfKind(descriptor.objects, "evidence")) {
    const selected = [...selectedRows.keys()].filter((row) => row >= reference.recordStart && row < reference.recordStart + reference.recordCount);
    if (selected.length === 0) continue;
    throwIfCancelled(input.signal);
    const object = await input.generation.readObject(reference);
    if (input.stats) input.stats.evidenceReads += 1;
    afterRead(input.signal);
    if (object.reference.kind !== "evidence") throw corruption("selected evidence decoded as the wrong kind");
    const evidence = object.block as EvidenceBlock; observeLive(input.stats, null, evidence);
    for (const row of selected) materializeEvidence(evidence, reference, row, selectedRows.get(row)!, materialized);
    await yieldToTimer(input.signal);
  }
  topPapers.sort(comparePapers);
  return topPapers.map((paper) => ({
    paperKey: paper.paperKey, score: paper.score, scoreKind: "bm25", rankingScore: paper.score, rankingScoreKind: "bm25",
    hits: paper.hits.map((hit) => materialized.get(hit.chunkOrdinal) ?? (() => { throw corruption("selected lexical evidence was not materialized"); })()),
    chunkCount: paper.chunkCount,
  }));
}

function scorePostingsRange(block: LexicalPostingsBlock, routed: ReadonlySet<number>, queryTerms: readonly string[], qtf: ReadonlyMap<string, number>, df: readonly number[], chunkCount: number, avgdl: number, namespace: LexicalNamespace, compactAlias: string, paper: PaperMetadataRecord, k1: number, b: number): HitCandidate[] {
  const start = Math.max(block.chunkStart, paper.chunkStart);
  const end = Math.min(block.chunkStart + block.chunks.length, paper.chunkStart + paper.chunkCount);
  const byChunk = new Map<number, Map<number, number>>();
  const aliasChunks = new Set<number>();
  for (const occurrence of block.occurrences) {
    if (occurrence.chunkOrdinal < start || occurrence.chunkOrdinal >= end) continue;
    if (occurrence.namespace === "alias" && routed.has(queryTerms.length) && compactAlias
      && block.chunks[occurrence.chunkOrdinal - block.chunkStart]!.compactText.includes(compactAlias)) aliasChunks.add(occurrence.chunkOrdinal);
    if (occurrence.namespace !== namespace) continue;
    const queryIndex = queryTerms.indexOf(occurrence.term); if (queryIndex < 0 || !routed.has(queryIndex)) continue;
    let terms = byChunk.get(occurrence.chunkOrdinal); if (!terms) byChunk.set(occurrence.chunkOrdinal, terms = new Map()); terms.set(queryIndex, occurrence.tf);
  }
  const hits: HitCandidate[] = [];
  for (let ordinal = start; ordinal < end; ordinal += 1) {
    const terms = byChunk.get(ordinal); const chunk = block.chunks[ordinal - block.chunkStart]!;
    if (chunk.paperOrdinal !== paper.paperOrdinal || chunk.chunkIndex !== ordinal - paper.chunkStart) {
      throw corruption("postings chunk identity does not match paper metadata");
    }
    let score = 0;
    if (terms) {
      const dl = namespace === "expanded" ? chunk.expandedLength : chunk.baseLength;
      for (let index = 0; index < queryTerms.length; index += 1) {
        const tf = terms.get(index) ?? 0; if (tf === 0) continue;
        const termDf = df[index] ?? 0; const idf = Math.log(1 + (chunkCount - termDf + 0.5) / (termDf + 0.5));
        const norm = 1 - b + b * dl / avgdl;
        score += qtf.get(queryTerms[index]!)! * idf * (tf * (k1 + 1)) / (tf + k1 * norm);
      }
    }
    if (score === 0 && aliasChunks.has(ordinal)) score = Number.EPSILON;
    if (score > 0) hits.push({ chunkOrdinal: ordinal, chunkIndex: ordinal - paper.chunkStart, score });
  }
  return hits;
}

function materializeEvidence(block: EvidenceBlock, reference: GenerationObjectReference, row: number, selected: { paper: PaperCandidate; hit: HitCandidate }, output: Map<number, KnowledgeBaseChunkHit>): void {
  const record = block.records[row - reference.recordStart];
  if (!record || record.vectorRow !== row || record.paperIndex !== selected.paper.ordinal || record.paperKey !== selected.paper.paperKey || record.chunk.index !== selected.hit.chunkIndex) throw corruption("selected evidence does not match lexical candidate");
  output.set(row, { source: "lexical", scoreKind: "bm25", chunkIndex: record.chunk.index, chunkId: record.chunk.id, headings: record.chunk.headings, locator: record.chunk.locator, page: record.chunk.page, text: record.chunk.text, score: selected.hit.score });
}
function selectAliasGram(value: string): string { const chars = Array.from(value); if (chars.length === 0) return ""; if (chars.length <= 2) return chars.join(""); let best = chars.slice(0, 3).join(""); for (let index = 1; index + 2 < chars.length; index += 1) { const gram = chars.slice(index, index + 3).join(""); if (compareUtf8(gram, best) < 0) best = gram; } return best; }
function compactUnicode(text: string): string { return text.normalize("NFKC").toLocaleLowerCase("und").replace(/[^\p{L}\p{N}]+/gu, ""); }
function compareUtf8(left: string, right: string): number { const encoder = new TextEncoder(); const a = encoder.encode(left); const b = encoder.encode(right); for (let index = 0; index < Math.min(a.length, b.length); index += 1) if (a[index] !== b[index]) return a[index]! - b[index]!; return a.length - b.length; }
function compareHits(left: HitCandidate, right: HitCandidate): number { return right.score !== left.score ? right.score - left.score : left.chunkIndex - right.chunkIndex; }
function comparePapers(left: PaperCandidate, right: PaperCandidate): number { return right.lexicalPriority !== left.lexicalPriority ? right.lexicalPriority - left.lexicalPriority : right.score !== left.score ? right.score - left.score : left.paperKey.localeCompare(right.paperKey); }
function insertBounded<T>(items: T[], candidate: T, limit: number, compare: (left: T, right: T) => number): { inserted: boolean; removed?: T } { if (items.length === limit && compare(candidate, items[items.length - 1]!) >= 0) return { inserted: false }; let index = 0; while (index < items.length && compare(items[index]!, candidate) <= 0) index += 1; items.splice(index, 0, candidate); const removed = items.length > limit ? items.pop() : undefined; return { inserted: true, ...(removed === undefined ? {} : { removed }) }; }
function refsOfKind<K extends GenerationObjectReference["kind"]>(refs: readonly GenerationObjectReference[], kind: K): Array<GenerationObjectReference & { kind: K }> { return refs.filter((ref): ref is GenerationObjectReference & { kind: K } => ref.kind === kind); }
function boundedInteger(value: number | undefined, name: string, fallback: number, min: number, max: number): number { const actual = value ?? fallback; if (!Number.isSafeInteger(actual) || actual < min || actual > max) throw new TypeError(`searchGenerationBm25: ${name} must be an integer from ${min} through ${max}`); return actual; }
function finiteRange(value: number | undefined, name: string, fallback: number, min: number, max: number): number { const actual = value ?? fallback; if (!Number.isFinite(actual) || actual < min || actual > max) throw new TypeError(`searchGenerationBm25: ${name} must be in [${min}, ${max}]`); return actual; }
function corruption(message: string): FullTextGenerationIndexStoreError { return new FullTextGenerationIndexStoreError(message, "corrupt-or-unreadable"); }
function resetStats(stats?: GenerationBm25Stats): void { if (!stats) return; stats.dictionaryReads = 0; stats.postingsReads = 0; stats.postingReads = 0; stats.metadataReads = 0; stats.evidenceReads = 0; stats.peakChunkAccumulators = 0; stats.peakCandidates = 0; stats.peakHits = 0; stats.totalRetainedHits = 0; stats.maxLiveBlocks = 0; }
function observeLive(stats: GenerationBm25Stats | undefined, metadata: object | null, postings: object | null): void { if (stats) stats.maxLiveBlocks = Math.max(stats.maxLiveBlocks, Number(metadata !== null) + Number(postings !== null)); }
function throwIfCancelled(signal?: AbortSignal): void { if (!signal?.aborted) return; if (typeof DOMException === "function") throw new DOMException("The operation was aborted", "AbortError"); const error = new Error("The operation was aborted"); error.name = "AbortError"; throw error; }
function afterRead(signal?: AbortSignal): void { throwIfCancelled(signal); }
async function yieldToTimer(signal?: AbortSignal): Promise<void> { throwIfCancelled(signal); await new Promise<void>((resolve) => setTimeout(resolve, 0)); throwIfCancelled(signal); }
