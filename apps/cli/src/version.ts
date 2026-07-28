/**
 * Package version for the CLI binary.
 * Injected at build time via esbuild `define` as a string literal.
 * Fallback is for tests / unbundled ts execution.
 */
declare const __ARXIV_DAILY_VERSION__: string | undefined;

export const CLI_PACKAGE_NAME = "arxiv-daily";

export function getCliVersion(): string {
  try {
    if (typeof __ARXIV_DAILY_VERSION__ === "string" && __ARXIV_DAILY_VERSION__) {
      return __ARXIV_DAILY_VERSION__;
    }
  } catch {
    /* not defined */
  }
  return "0.0.0-dev";
}

/** Compare dotted numeric versions: -1 if a<b, 0 if equal, 1 if a>b. */
export function compareSemver(a: string, b: string): number {
  const pa = a.replace(/^v/, "").split(".").map((x) => Number.parseInt(x, 10) || 0);
  const pb = b.replace(/^v/, "").split(".").map((x) => Number.parseInt(x, 10) || 0);
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i++) {
    const da = pa[i] ?? 0;
    const db = pb[i] ?? 0;
    if (da < db) return -1;
    if (da > db) return 1;
  }
  return 0;
}
