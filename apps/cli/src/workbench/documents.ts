import { open, readdir, realpath, stat, type FileHandle } from "node:fs/promises";
import path from "node:path";
import { derivePaperInboxPaths, modernArxivResources } from "@arxiv-daily/core";
import type { CliRuntimeConfig } from "../config";
import { describeMarkdown, renderMarkdown } from "./markdown";

export interface DocumentEntry {
  path: string;
  kind: "daily" | "papers";
  title: string;
  date: string;
  authors: string;
  arxivId: string;
  modifiedAt: string;
  size: number;
}
export class WorkbenchError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
const MIME: Record<string, string> = {
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
  ".webp": "image/webp", ".avif": "image/avif", ".svg": "image/svg+xml", ".pdf": "application/pdf",
};

/** Only configured reports and supported attachments, never arbitrary Vault files. */
export class WorkbenchDocuments {
  private cache = new Map<string, { stamp: string; entry: DocumentEntry }>();
  private readonly roots: Array<{ kind: "daily" | "papers"; path: string }>;
  private readonly pdfRoot: string;
  constructor(private readonly config: CliRuntimeConfig) {
    this.roots = [{ kind: "daily", path: config.settings.output.dailyDir }, { kind: "papers", path: config.settings.output.papersDir }];
    this.pdfRoot = `${derivePaperInboxPaths(config.settings.output).rootDir || "arxiv-daily"}/pdfs`;
  }

  async list(): Promise<DocumentEntry[]> {
    const entries: DocumentEntry[] = [];
    for (const root of this.roots) await this.walk(root.path, root.kind, entries, 0);
    const live = new Set(entries.map(entry => entry.path));
    for (const key of this.cache.keys()) if (!live.has(key)) this.cache.delete(key);
    return entries.sort((a, b) => b.date.localeCompare(a.date) || b.modifiedAt.localeCompare(a.modifiedAt) || a.path.localeCompare(b.path));
  }

  private async walk(directory: string, kind: DocumentEntry["kind"], result: DocumentEntry[], depth: number): Promise<void> {
    if (depth > 24 || result.length > 10000) throw new WorkbenchError(413, "文档数量或目录层级过多，请缩小输出目录。");
    const actual = await this.contained(directory, false);
    if (!actual) return;
    let children;
    try { children = await readdir(actual, { withFileTypes: true }); }
    catch (error) { if (missing(error)) return; throw error; }
    for (const child of children) {
      if (child.name.startsWith(".") || child.isSymbolicLink()) continue;
      const relative = path.posix.join(directory, child.name);
      if (child.isDirectory()) await this.walk(relative, kind, result, depth + 1);
      else if (child.isFile() && /\.md$/i.test(child.name)) {
        try {
          const file = await this.contained(relative, false);
          if (!file) continue;
          const info = await stat(file);
          const stamp = `${info.mtimeMs}:${info.ctimeMs}:${info.size}`;
          const cached = this.cache.get(relative);
          if (cached?.stamp === stamp) { result.push(cached.entry); continue; }
          const handle = await open(file, "r");
          let prefix: string;
          try {
            const buffer = Buffer.alloc(Math.min(info.size, 16384));
            const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
            prefix = buffer.subarray(0, bytesRead).toString("utf8");
          } finally { await handle.close(); }
          const { title, metadata } = describeMarkdown(prefix);
          const entry: DocumentEntry = {
            path: relative, kind, title: title || child.name.replace(/\.md$/i, ""),
            date: metadata.date || /\d{4}-\d{2}-\d{2}/.exec(metadata.published || child.name)?.[0] || "",
            authors: metadata.authors || "", arxivId: metadata.arxiv_id || modernArxivResources(child.name.replace(/\.md$/i, ""))?.id || "",
            modifiedAt: info.mtime.toISOString(), size: info.size,
          };
          this.cache.set(relative, { stamp, entry });
          result.push(entry);
        } catch (error) { if (!missing(error)) throw error; }
      }
    }
  }

  async document(relative: string) {
    const entries = await this.list();
    const entry = entries.find(item => item.path === relative);
    if (!entry) throw new WorkbenchError(404, "找不到这份文档，文件可能已移动或删除。");
    const source = await this.raw(relative);
    const rendered = renderMarkdown(source, { resolveLink: (target, kind) => this.resolveLink(target, relative, entries, kind) });
    const resources = modernArxivResources(rendered.metadata.arxiv_id || entry.arxivId);
    const related: Array<{ title: string; path: string }> = [];
    const published = /^\[\[([^|\]]+)(?:\|([^\]]+))?\]\]$/.exec(rendered.metadata.published || "");
    if (published) {
      const target = this.findDocument(published[1]!, relative, entries);
      if (target) related.push({ title: published[2] || target.date || target.title, path: target.path });
    }
    const pdfPath = resources ? `${this.pdfRoot}/${resources.id}.pdf` : "";
    const localPdf = pdfPath && await this.contained(pdfPath, true);
    return {
      ...entry, ...rendered, title: rendered.title || entry.title, related,
      originalUrl: resources?.absUrl ?? null,
      pdfUrl: localPdf ? `api/asset?path=${encodeURIComponent(pdfPath)}` : resources?.pdfUrl ?? null,
    };
  }

  async raw(relative: string): Promise<string> {
    if (!/\.md$/i.test(relative)) throw new WorkbenchError(404, "找不到文档。");
    const file = await this.contained(relative, false);
    if (!file) throw new WorkbenchError(404, "找不到文档。");
    let handle: FileHandle | undefined;
    try {
      handle = await open(file, "r");
      const info = await handle.stat();
      if (!info.isFile()) throw new WorkbenchError(404, "找不到文档。");
      if (info.size > 8 * 1024 * 1024) throw new WorkbenchError(413, "文档超过 8 MB，请使用本地编辑器打开。");
      return await handle.readFile("utf8");
    } catch (error) { if (missing(error)) throw new WorkbenchError(404, "找不到文档。"); throw error; }
    finally { await handle?.close(); }
  }

  async asset(relative: string) {
    const type = MIME[path.posix.extname(relative).toLowerCase()];
    const file = type && await this.contained(relative, true);
    if (!file) throw new WorkbenchError(404, "找不到附件。");
    const info = await stat(file);
    if (!info.isFile()) throw new WorkbenchError(404, "找不到附件。");
    return { file, type, size: info.size };
  }

  private findDocument(target: string, from: string, entries: DocumentEntry[]) {
    const withExtension = /\.md$/i.test(target) ? target : `${target}.md`;
    const relative = path.posix.normalize(path.posix.join(path.posix.dirname(from), withExtension));
    const direct = entries.find(item => item.path === withExtension || item.path === relative);
    if (direct) return direct;
    if (target.includes("/")) return undefined;
    const matches = entries.filter(item => path.posix.basename(item.path) === withExtension);
    return matches.length === 1 ? matches[0] : undefined;
  }

  private resolveLink(raw: string, from: string, entries: DocumentEntry[], kind: "link" | "image"): string | null {
    if (/^(https?:|mailto:)/i.test(raw)) return raw;
    if (raw.startsWith("#")) return kind === "link" ? raw : null;
    if (/^[a-z][a-z\d+.-]*:/i.test(raw) || raw.startsWith("/") || raw.includes("\\")) return null;
    let target: string;
    try { target = decodeURIComponent(raw); } catch { return null; }
    const [name, ...hash] = target.split("#");
    const fragment = hash.length ? `#${hash.join("#")}` : "";
    if (kind === "link") {
      const document = this.findDocument(name || "", from, entries);
      if (document) return `?document=${encodeURIComponent(document.path)}${fragment}`;
    }
    const asset = path.posix.normalize(path.posix.join(path.posix.dirname(from), name || ""));
    const rooted = this.allowedPath(name || "", true) ? name! : asset;
    if (this.allowedPath(rooted, true) && MIME[path.posix.extname(rooted).toLowerCase()]) return `api/asset?path=${encodeURIComponent(rooted)}${fragment}`;
    return null;
  }

  private allowedPath(relative: string, assets: boolean): string | undefined {
    if (!relative || relative.includes("\\") || relative.includes("\0") || path.posix.isAbsolute(relative) || relative.split("/").some(part => part.startsWith("."))) return undefined;
    const roots = [...this.roots.map(root => root.path), ...(assets && /\.pdf$/i.test(relative) ? [this.pdfRoot] : [])];
    return roots.find(root => relative === root || relative.startsWith(`${root}/`));
  }

  private async contained(relative: string, assets: boolean): Promise<string | null> {
    const root = this.allowedPath(relative, assets);
    if (!root) return null;
    try {
      const [vault, actualRoot, actual] = await Promise.all([
        realpath(this.config.vaultRoot), realpath(path.join(this.config.vaultRoot, root)), realpath(path.join(this.config.vaultRoot, relative)),
      ]);
      if (!inside(vault, actualRoot) || !inside(actualRoot, actual)) return null;
      if (path.relative(actualRoot, actual).split(path.sep).some(part => part.startsWith("."))) return null;
      return actual;
    } catch (error) { if (missing(error)) return null; throw error; }
  }
}
function inside(root: string, target: string): boolean {
  const relative = path.relative(root, target);
  return !path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(`..${path.sep}`);
}
function missing(error: unknown): boolean {
  return ["ENOENT", "ENOTDIR", "ELOOP"].includes((error as NodeJS.ErrnoException).code || "");
}
