import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import type { PaperIndexEntry, PaperIndexStore } from "./paper-index";
import type { Logger } from "./logger";

export type ProjectNoteResult =
  | {
      kind: "done";
      arxivId: string;
      projectPath: string;
      appended: boolean;
      entryUpdated: boolean;
    }
  | { kind: "invalid_project"; reason: string };

export interface ProjectNotesServiceDeps {
  storage: StorageAdapter;
  paperIndex?: PaperIndexStore;
  output: OutputSettings;
  logger: Logger;
}

export class ProjectNotesService {
  constructor(private deps: ProjectNotesServiceDeps) {}

  async addPaperToProject(
    entry: PaperIndexEntry,
    rawProjectPath: string,
  ): Promise<ProjectNoteResult> {
    const projectPath = normalizeProjectPath(
      rawProjectPath,
      this.deps.storage,
    );
    if (!projectPath) {
      return {
        kind: "invalid_project",
        reason: "project note path is required",
      };
    }

    const marker = projectMarker(entry.arxivId);
    const existing = (await this.deps.storage.exists(projectPath))
      ? await this.deps.storage.readText(projectPath)
      : defaultProjectNote(projectPath);
    const appended = !existing.includes(marker);
    const next = appended
      ? appendProjectPaper(existing, projectLine(entry, projectPath, this.deps.output, marker))
      : existing;
    if (appended || !(await this.deps.storage.exists(projectPath))) {
      await ensureDirDeep(this.deps.storage, parentDir(projectPath));
      await this.deps.storage.writeText(projectPath, next);
    }

    let entryUpdated = false;
    if (this.deps.paperIndex) {
      const updated = await this.deps.paperIndex.addProject(
        entry.arxivId,
        projectPath,
      );
      entryUpdated = Boolean(updated);
    }

    return {
      kind: "done",
      arxivId: entry.arxivId,
      projectPath,
      appended,
      entryUpdated,
    };
  }
}

function normalizeProjectPath(
  rawPath: string,
  storage: StorageAdapter,
): string {
  const trimmed = rawPath.trim().replace(/^\/+/, "");
  if (!trimmed) return "";
  const withExt = /\.md$/i.test(trimmed) ? trimmed : `${trimmed}.md`;
  return storage.normalizePath(withExt);
}

function defaultProjectNote(path: string): string {
  return `# ${titleFromPath(path)}\n\n## Papers\n\n`;
}

function appendProjectPaper(markdown: string, line: string): string {
  const body = markdown.trimEnd();
  const separator = body.includes("\n## Papers") ? "\n" : "\n\n## Papers\n";
  return `${body}${separator}${line}\n`;
}

function projectLine(
  entry: PaperIndexEntry,
  projectPath: string,
  output: OutputSettings,
  marker: string,
): string {
  const link = entry.paperPath
    ? paperLink(projectPath, entry, output)
    : `[${entry.arxivId}](${entry.arxivUrl})`;
  return `- ${link} — ${entry.title} ${marker}`;
}

function paperLink(
  fromProjectPath: string,
  entry: PaperIndexEntry,
  output: OutputSettings,
): string {
  const target = entry.paperPath ?? "";
  if ((output.linkStyle ?? "wikilink") === "relative") {
    return `[${entry.arxivId}](${encodeRelativeLinkTarget(relativePath(fromProjectPath, target))})`;
  }
  return `[[${target.replace(/\.md$/i, "")}|${entry.arxivId}]]`;
}

function projectMarker(arxivId: string): string {
  return `<!-- arxiv-daily-project:${arxivId} -->`;
}

function titleFromPath(path: string): string {
  return path
    .split("/")
    .pop()!
    .replace(/\.md$/i, "")
    .replace(/[-_]+/g, " ")
    .trim();
}

function relativePath(fromFile: string, toFile: string): string {
  const fromDir = parentParts(fromFile);
  const toParts = pathParts(toFile);
  let i = 0;
  while (i < fromDir.length && i < toParts.length && fromDir[i] === toParts[i]) {
    i++;
  }
  const up = Array(fromDir.length - i).fill("..");
  const down = toParts.slice(i);
  return [...up, ...down].join("/") || toParts[toParts.length - 1] || ".";
}

function parentDir(path: string): string {
  const parts = pathParts(path);
  return parts.length <= 1 ? "" : parts.slice(0, -1).join("/");
}

function parentParts(path: string): string[] {
  return pathParts(path).slice(0, -1);
}

function pathParts(path: string): string[] {
  return path.split("/").filter((part) => part && part !== ".");
}

function encodeRelativeLinkTarget(path: string): string {
  return encodeURI(path).replace(/\(/g, "%28").replace(/\)/g, "%29");
}

async function ensureDirDeep(
  storage: StorageAdapter,
  dir: string,
): Promise<void> {
  if (!dir) return;
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur = cur ? `${cur}/${part}` : part;
    if (!(await storage.exists(cur))) await storage.mkdir(cur);
  }
}
