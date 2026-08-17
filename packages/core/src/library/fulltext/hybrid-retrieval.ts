import type { KnowledgeBaseChunkHit, KnowledgeBasePaperMatch } from "./retrieval";

const DEFAULT_LIMIT = 10;
const DEFAULT_CANDIDATE_LIMIT = 50;
const DEFAULT_MAX_HITS_PER_PAPER = 3;
const DEFAULT_RRF_K = 60;

export interface FusePaperRankingsInput {
  rankings: readonly (readonly KnowledgeBasePaperMatch[])[];
  limit?: number;
  candidateLimit?: number;
  maxHitsPerPaper?: number;
  rrfK?: number;
}

interface Accumulator {
  paperKey: string;
  rankingScore: number;
  chunkCount: number;
  channelHits: KnowledgeBaseChunkHit[][];
  denseScore?: number;
  lexicalScore?: number;
}

/** Paper-level RRF; ranking score and channel evidence scores remain distinct. */
export function fusePaperRankingsRrf(input: FusePaperRankingsInput): KnowledgeBasePaperMatch[] {
  const limit = nonNegativeInteger(input.limit, "limit", DEFAULT_LIMIT);
  const candidateLimit = positiveInteger(input.candidateLimit, "candidateLimit", DEFAULT_CANDIDATE_LIMIT);
  const maxHitsPerPaper = positiveInteger(input.maxHitsPerPaper, "maxHitsPerPaper", DEFAULT_MAX_HITS_PER_PAPER);
  const rrfK = finiteNonNegative(input.rrfK, "rrfK", DEFAULT_RRF_K);
  if (limit === 0 || input.rankings.length === 0) return [];

  const papers = new Map<string, Accumulator>();
  input.rankings.forEach((ranking, channelIndex) => {
    const unique = uniquePapers(ranking).slice(0, candidateLimit);
    unique.forEach((match, index) => {
      let accumulator = papers.get(match.paperKey);
      if (!accumulator) {
        accumulator = {
          paperKey: match.paperKey,
          rankingScore: 0,
          chunkCount: match.chunkCount,
          channelHits: Array.from({ length: input.rankings.length }, () => []),
        };
        papers.set(match.paperKey, accumulator);
      }
      accumulator.rankingScore += 1 / (rrfK + index + 1);
      accumulator.chunkCount = Math.max(accumulator.chunkCount, match.chunkCount);
      accumulator.channelHits[channelIndex]!.push(...match.hits);
      if (match.scoreKind === "cosine" && accumulator.denseScore === undefined) accumulator.denseScore = match.score;
      if (match.scoreKind === "bm25" && accumulator.lexicalScore === undefined) accumulator.lexicalScore = match.score;
    });
  });

  return [...papers.values()]
    .sort((left, right) => right.rankingScore - left.rankingScore || left.paperKey.localeCompare(right.paperKey))
    .slice(0, limit)
    .map((entry) => {
      const dense = entry.denseScore !== undefined;
      const score = dense ? entry.denseScore! : entry.lexicalScore ?? 0;
      return {
        paperKey: entry.paperKey,
        score,
        scoreKind: dense ? "cosine" : "bm25",
        rankingScore: entry.rankingScore,
        rankingScoreKind: "rrf",
        hits: roundRobinHits(entry.channelHits, maxHitsPerPaper),
        chunkCount: entry.chunkCount,
      };
    });
}

function uniquePapers(ranking: readonly KnowledgeBasePaperMatch[]): KnowledgeBasePaperMatch[] {
  const seen = new Set<string>();
  const unique: KnowledgeBasePaperMatch[] = [];
  for (const match of ranking) {
    if (seen.has(match.paperKey)) continue;
    seen.add(match.paperKey);
    unique.push(match);
  }
  return unique;
}

function roundRobinHits(channels: readonly (readonly KnowledgeBaseChunkHit[])[], limit: number): KnowledgeBaseChunkHit[] {
  const result: KnowledgeBaseChunkHit[] = [];
  const seen = new Set<string>();
  const cursors = channels.map(() => 0);
  while (result.length < limit) {
    let added = false;
    for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
      const hits = channels[channelIndex]!;
      while (cursors[channelIndex]! < hits.length) {
        const hit = hits[cursors[channelIndex]!]!;
        cursors[channelIndex]! += 1;
        if (seen.has(hit.chunkId)) continue;
        seen.add(hit.chunkId);
        result.push(hit);
        added = true;
        break;
      }
      if (result.length === limit) break;
    }
    // Every cursor only moves forward. No addition means every remaining hit,
    // if any, was a duplicate and all channels are now exhausted.
    if (!added) break;
  }
  return result;
}

function positiveInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(`fusePaperRankingsRrf: ${name} must be positive`);
  return value;
}

function nonNegativeInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`fusePaperRankingsRrf: ${name} must be non-negative`);
  return value;
}

function finiteNonNegative(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value < 0) throw new TypeError(`fusePaperRankingsRrf: ${name} must be non-negative`);
  return value;
}
