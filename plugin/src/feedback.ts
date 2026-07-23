/** Public repository and feedback entry points for arXiv Daily. */

export const ARXIV_DAILY_REPO_URL = "https://github.com/tdccccc/arxiv-daily";
export const ARXIV_DAILY_DOCS_URL =
  "https://github.com/tdccccc/arxiv-daily/blob/main/docs/getting-started.md";
export const ARXIV_DAILY_ISSUES_URL = `${ARXIV_DAILY_REPO_URL}/issues`;

/** Build a low-friction GitHub issue URL with only the plugin version. */
export function buildBugReportUrl(pluginVersion: string): string {
  const body = [
    `- arXiv Daily: ${pluginVersion}`,
    "",
    "",
  ].join("\n");

  const params = new URLSearchParams({ body });
  return `${ARXIV_DAILY_ISSUES_URL}/new?${params.toString()}`;
}

/** Feature requests: blank issue body for maximum freedom. */
export function buildFeatureRequestUrl(): string {
  return `${ARXIV_DAILY_ISSUES_URL}/new`;
}
