import type { App } from "obsidian";

export type LibraryPdfOpenTarget =
  | { readonly kind: "vault"; readonly path: string }
  | { readonly kind: "external"; readonly url: string };

export function resolveLibraryPdfOpenTarget(input: {
  readonly canonicalRoot: string;
  readonly logicalPath: string;
  readonly page: number;
  readonly vaultRoot?: string;
}): LibraryPdfOpenTarget {
  const segments = logicalPathSegments(input.logicalPath);
  if (!Number.isSafeInteger(input.page) || input.page < 1) {
    throw new TypeError("The PDF page must be a positive integer");
  }
  const absolutePath = joinLibraryPath(input.canonicalRoot, segments);
  const vaultPath = vaultRelativePath(input.vaultRoot, absolutePath);
  if (vaultPath !== null) {
    return { kind: "vault", path: `${vaultPath}#page=${input.page}` };
  }
  return { kind: "external", url: externalFileUrl(absolutePath, input.page) };
}

export async function openLibraryPdfAtPage(input: {
  readonly app: Pick<App, "workspace">;
  readonly target: LibraryPdfOpenTarget;
}): Promise<"page-targeted" | "file-fallback"> {
  if (input.target.kind === "external") {
    try {
      window.activeWindow.open(input.target.url, "_blank", "noopener");
      return "page-targeted";
    } catch {
      window.activeWindow.open(input.target.url.replace(/#page=\d+$/u, ""), "_blank", "noopener");
      return "file-fallback";
    }
  }
  try {
    await input.app.workspace.openLinkText(input.target.path, "", false);
    return "page-targeted";
  } catch {
    const filePath = input.target.path.replace(/#page=\d+$/u, "");
    await input.app.workspace.openLinkText(filePath, "", false);
    return "file-fallback";
  }
}

function logicalPathSegments(logicalPath: string): string[] {
  if (
    !logicalPath
    || logicalPath.includes("\\")
    || logicalPath.includes("\0")
    || logicalPath.startsWith("/")
    || /^[A-Za-z]:/.test(logicalPath)
  ) {
    throw new TypeError("The library path must be a safe relative path");
  }
  const segments = logicalPath.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    throw new TypeError("The library path must be a safe relative path");
  }
  return segments;
}

function joinLibraryPath(canonicalRoot: string, segments: readonly string[]): string {
  const root = normalizeAbsolutePath(canonicalRoot);
  return `${root === "/" ? "" : root}/${segments.join("/")}`;
}

function vaultRelativePath(vaultRoot: string | undefined, absolutePath: string): string | null {
  if (!vaultRoot) return null;
  let root: string;
  try {
    root = normalizeAbsolutePath(vaultRoot);
  } catch {
    return null;
  }
  const normalizeForComparison = isWindowsStylePath(root)
    ? (value: string) => value.toLowerCase()
    : (value: string) => value;
  const comparedPath = normalizeForComparison(absolutePath);
  const prefix = root === "/" ? "/" : `${root}/`;
  const comparedPrefix = normalizeForComparison(prefix);
  if (!comparedPath.startsWith(comparedPrefix)) return null;
  const relativePath = absolutePath.slice(prefix.length);
  return relativePath || null;
}

function externalFileUrl(absolutePath: string, page: number): string {
  if (absolutePath.startsWith("//")) {
    const [host, ...segments] = absolutePath.slice(2).split("/");
    if (!host || segments.length === 0) throw new TypeError("The selected library root must be absolute");
    return `file://${encodeURIComponent(host)}/${encodeFilePath(segments)}#page=${page}`;
  }
  const path = absolutePath.startsWith("/") ? absolutePath.slice(1) : absolutePath;
  return `file:///${encodeFilePath(path.split("/"))}#page=${page}`;
}

function encodeFilePath(segments: readonly string[]): string {
  return segments.map((segment) => encodeURIComponent(segment).replace(/%3A/giu, ":")).join("/");
}

function normalizeAbsolutePath(value: string): string {
  if (typeof value !== "string" || !value) {
    throw new TypeError("The selected library root must be absolute");
  }
  const path = value.replace(/\\/g, "/");
  if (path.startsWith("//")) {
    const normalized = path.slice(2).replace(/\/{2,}/g, "/").replace(/\/+$/g, "");
    if (!normalized) throw new TypeError("The selected library root must be absolute");
    return `//${normalized}`;
  }
  if (path.startsWith("/")) {
    return `/${path.slice(1).replace(/\/{2,}/g, "/").replace(/\/+$/g, "")}`;
  }
  if (/^[A-Za-z]:\//.test(path)) {
    return path.replace(/\/{2,}/g, "/").replace(/\/+$/g, "");
  }
  throw new TypeError("The selected library root must be absolute");
}

function isWindowsStylePath(path: string): boolean {
  return /^[A-Za-z]:\//.test(path) || path.startsWith("//");
}
