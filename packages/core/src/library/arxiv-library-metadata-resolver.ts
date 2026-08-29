import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import type {
  PersonalLibraryMetadataResolver,
  PersonalLibraryResolvedMetadata,
} from "./personal-library-reconciliation";

export class ArxivLibraryMetadataResolver implements PersonalLibraryMetadataResolver {
  constructor(private readonly fetcher: Pick<ArxivFetcher, "fetchMetadataByIds">) {}

  async resolve(
    arxivIds: string[],
    signal?: AbortSignal,
  ): Promise<Map<string, PersonalLibraryResolvedMetadata>> {
    const metadata = await this.fetcher.fetchMetadataByIds(arxivIds, signal);
    const resolved = new Map<string, PersonalLibraryResolvedMetadata>();
    for (const [arxivId, paper] of metadata) {
      resolved.set(arxivId, {
        arxivId,
        title: paper.title,
        authors: [...paper.authorNames],
        abstract: paper.abstract,
        published: canonicalIsoDate(paper.published),
        updated: canonicalIsoDate(paper.updated),
        primaryCategory: paper.primaryCategory,
        categories: [...paper.categories],
      });
    }
    return resolved;
  }
}

function canonicalIsoDate(value: string): string {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : value;
}
