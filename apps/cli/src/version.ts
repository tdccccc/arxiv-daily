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

// SemVer 2.0.0 (https://semver.org), captured as core / prerelease / build.
const SEMVER =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;

/** True for a strict SemVer 2.0.0 version (no leading "v"); safe to embed in an npm spec. */
export function isSemVer(value: unknown): value is string {
  return typeof value === "string" && value.length <= 128 && SEMVER.test(value);
}

function parseSemver(value: string): { core: number[]; pre: string[] } | null {
  const match = SEMVER.exec(value.replace(/^v/, ""));
  if (!match) return null;
  return { core: [Number(match[1]), Number(match[2]), Number(match[3])], pre: match[4] ? match[4].split(".") : [] };
}

/**
 * Compare versions by SemVer precedence: -1 if a<b, 0 if equal, 1 if a>b.
 * A prerelease sorts below its release (0.7.0-rc.1 < 0.7.0); build metadata is ignored.
 * Input that is not valid SemVer falls back to comparing the leading numeric parts.
 */
export function compareSemver(a: string, b: string): number {
  const pa = parseSemver(a);
  const pb = parseSemver(b);
  if (!pa || !pb) {
    const x = a.replace(/^v/, "").split(".").map((part) => Number.parseInt(part, 10) || 0);
    const y = b.replace(/^v/, "").split(".").map((part) => Number.parseInt(part, 10) || 0);
    for (let i = 0; i < Math.max(x.length, y.length); i++) {
      const da = x[i] ?? 0;
      const db = y[i] ?? 0;
      if (da !== db) return da < db ? -1 : 1;
    }
    return 0;
  }
  for (let i = 0; i < 3; i++) {
    if (pa.core[i] !== pb.core[i]) return pa.core[i]! < pb.core[i]! ? -1 : 1;
  }
  if (!pa.pre.length || !pb.pre.length) {
    return pa.pre.length === pb.pre.length ? 0 : pa.pre.length ? -1 : 1;
  }
  for (let i = 0; i < Math.max(pa.pre.length, pb.pre.length); i++) {
    const left = pa.pre[i];
    const right = pb.pre[i];
    if (left === undefined) return -1;
    if (right === undefined) return 1;
    if (left === right) continue;
    const leftNumeric = /^\d+$/.test(left);
    const rightNumeric = /^\d+$/.test(right);
    if (leftNumeric && rightNumeric) return Number(left) < Number(right) ? -1 : 1;
    if (leftNumeric !== rightNumeric) return leftNumeric ? -1 : 1;
    return left < right ? -1 : 1;
  }
  return 0;
}
