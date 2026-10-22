/**
 * Stands in for `abstractConclusion` when none of a paper's content could be
 * fetched. It carries no paper text, so nothing may be summarized from it.
 */
export const CONTENT_FETCH_FAILED_PREFIX = "[获取失败]";

export function contentFetchFailedText(arxivId: string): string {
  return `${CONTENT_FETCH_FAILED_PREFIX} arXiv ID: ${arxivId}`;
}

export function isContentFetchFailure(text: string): boolean {
  return text.trimStart().startsWith(CONTENT_FETCH_FAILED_PREFIX);
}
