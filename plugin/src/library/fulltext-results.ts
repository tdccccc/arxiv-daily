import type {
  FullTextKnowledgeBaseManifest,
  KnowledgeBaseChunkHit,
  KnowledgeBasePaperMatch,
} from "@arxiv-daily/core";

/** A presentation-safe projection of one Core full-text match. */
export interface LibraryFullTextMatch {
  readonly paperKey: string;
  readonly title: string;
  /** First known PDF path relative to the selected personal library root. */
  readonly filePath?: string;
  readonly score: number;
  readonly scoreKind: "cosine" | "bm25";
  readonly rankingScore: number;
  readonly rankingScoreKind: "cosine" | "bm25" | "rrf";
  readonly hits: readonly KnowledgeBaseChunkHit[];
}

export function projectLibraryFullTextMatches(input: {
  readonly catalogTitles: ReadonlyMap<string, string>;
  readonly manifest: FullTextKnowledgeBaseManifest;
  readonly matches: readonly KnowledgeBasePaperMatch[];
}): LibraryFullTextMatch[] {
  return input.matches.map((match) => {
    const record = input.manifest.papers[match.paperKey];
    const filePath = record?.filePaths[0];
    return {
      paperKey: match.paperKey,
      title: input.catalogTitles.get(match.paperKey) ?? record?.title ?? match.paperKey,
      ...(filePath ? { filePath } : {}),
      score: match.score,
      scoreKind: match.scoreKind,
      rankingScore: match.rankingScore,
      rankingScoreKind: match.rankingScoreKind,
      hits: match.hits,
    };
  });
}
