import type { StorageAdapter } from "../core/adapters";
import type { Logger } from "../services/logger";
import type { ArxivSettings, OutputSettings } from "../settings/types";
import { formatArxivCategories } from "../settings/categories";
import type { DailyPaperWithContent } from "./summarizer";
import type { PaperIndexEntry } from "../services/paper-index";

export interface MarkdownWriterOpts {
  storage: StorageAdapter;
  logger: Logger;
  arxiv: ArxivSettings;
  output: OutputSettings;
}

export interface WriteDailyOptions {
  dateWindowNote?: string;
}

export class MarkdownWriter {
  constructor(private opts: MarkdownWriterOpts) {}

  dailyPath(dateStr: string): string {
    return this.opts.storage.normalizePath(
      `${this.opts.output.dailyDir}/${dateStr}.md`,
    );
  }

  paperDetailPath(id: string): string {
    return this.opts.storage.normalizePath(
      `${this.opts.output.papersDir}/${id}.md`,
    );
  }

  paperDetailLink(
    id: string,
    dateStr: string,
    paperPath?: string | null,
  ): string {
    if ((this.opts.output.linkStyle ?? "wikilink") === "relative") {
      const target = this.opts.storage.normalizePath(
        paperPath || this.paperDetailPath(id),
      );
      const relative = relativePath(this.dailyPath(dateStr), target);
      return `[${id}](${encodeRelativeLinkTarget(relative)})`;
    }
    return `[[${id}]]`;
  }

  async writeDaily(
    dateStr: string,
    summary: string,
    options: WriteDailyOptions = {},
  ): Promise<string> {
    const path = this.dailyPath(dateStr);
    await this.ensureDir(this.opts.output.dailyDir);
    if (await this.opts.storage.exists(path)) {
      throw new Error(`daily already exists: ${path}`);
    }
    const frontmatter =
      `---\n` +
      `date: ${dateStr}\n` +
      `weekday: ${weekdayName(dateStr)}\n` +
      `tags: [arxiv, daily]\n` +
      `---\n\n`;
    await this.opts.storage.writeText(
      path,
      frontmatter + dateWindowNote(options.dateWindowNote) + summary,
    );
    this.opts.logger.info(`wrote daily: ${path}`);
    return path;
  }

  async writePaperDetail(
    paper: DailyPaperWithContent,
    dateStr: string,
    summary: string,
    indexEntry?: PaperIndexEntry,
  ): Promise<string> {
    const path = this.paperDetailPath(paper.id);
    await this.ensureDir(this.opts.output.papersDir);
    if (await this.opts.storage.exists(path)) {
      throw new Error(`paper already exists: ${path}`);
    }
    const tags = this.tagsFor(paper);
    const fm = paperFrontmatter({
      title: paper.title,
      authors: paper.authors,
      arxivId: paper.id,
      primaryTopic: indexEntry?.primaryTopic ?? paper.category,
      dailyReport: dailyReportLink(this.dailyPath(dateStr), dateStr),
      tags,
    });
    await this.opts.storage.writeText(path, fm + summary);
    this.opts.logger.info(`wrote paper: ${path}`);
    return path;
  }

  async writePaperNote(entry: PaperIndexEntry, body?: string): Promise<string> {
    const path = entry.paperPath ?? this.paperDetailPath(entry.arxivId);
    await this.ensureDir(this.opts.output.papersDir);
    if (await this.opts.storage.exists(path)) {
      throw new Error(`paper already exists: ${path}`);
    }
    const topic = this.opts.arxiv.topics.find((t) => t.tag === entry.primaryTopic);
    const tags = ["arxiv", "paper", topic?.tag ?? entry.primaryTopic].filter(Boolean);
    const fm = paperFrontmatter({
      title: entry.title,
      authors: entry.authors.join(", "),
      arxivId: entry.arxivId,
      primaryTopic: entry.primaryTopic,
      dailyReport: latestDailyReportLink(entry.dailyReports),
      tags,
    });
    const noteBody =
      body ??
      `# ${entry.title}\n\n` +
        `- **arXiv**: [${entry.arxivId}](${entry.arxivUrl})\n` +
        `- **PDF**: [PDF](${entry.pdfUrl})\n\n` +
        `## Notes\n\n`;
    await this.opts.storage.writeText(path, fm + noteBody);
    this.opts.logger.info(`wrote paper note: ${path}`);
    return path;
  }

  async writeEmptyDaily(
    dateStr: string,
    options: WriteDailyOptions = {},
  ): Promise<string> {
    const summary = `# arXiv ${formatArxivCategories(this.opts.arxiv)} 每日追踪 ${dateStr}\n\n今日未发现相关论文。\n`;
    return this.writeDaily(dateStr, summary, options);
  }

  async dailyExists(dateStr: string): Promise<boolean> {
    const path = this.dailyPath(dateStr);
    return await this.opts.storage.exists(path);
  }

  async paperDetailExists(id: string): Promise<boolean> {
    const path = this.paperDetailPath(id);
    return await this.opts.storage.exists(path);
  }

  private tagsFor(paper: DailyPaperWithContent): string[] {
    const tags = ["arxiv", "paper"];
    const topic = this.opts.arxiv.topics.find((t) => t.tag === paper.category);
    if (topic) tags.push(topic.tag);
    return tags;
  }

  private async ensureDir(rel: string): Promise<void> {
    const norm = this.opts.storage.normalizePath(rel);
    if (!(await this.opts.storage.exists(norm))) {
      await this.opts.storage.mkdir(norm);
    }
  }

}

function dateWindowNote(note: string | undefined): string {
  return note ? `> ${note}\n\n` : "";
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
  const rel = [...up, ...down].join("/");
  return rel || toParts[toParts.length - 1] || ".";
}

function parentParts(path: string): string[] {
  const parts = pathParts(path);
  return parts.slice(0, -1);
}

function pathParts(path: string): string[] {
  return path.split("/").filter((part) => part && part !== ".");
}

function encodeRelativeLinkTarget(path: string): string {
  return encodeURI(path).replace(/\(/g, "%28").replace(/\)/g, "%29");
}

function escapeYaml(s: string): string {
  return s.replace(/"/g, '\\"');
}

function paperFrontmatter(meta: {
  title: string;
  authors: string;
  arxivId: string;
  primaryTopic: string;
  dailyReport?: string;
  tags: string[];
}): string {
  const dailyReport = meta.dailyReport
    ? `daily_report: "${escapeYaml(meta.dailyReport)}"\n`
    : "";
  return (
    `---\n` +
    `title: "${escapeYaml(meta.title)}"\n` +
    `authors: "${escapeYaml(meta.authors)}"\n` +
    `arxiv_id: "${meta.arxivId}"\n` +
    `primary_topic: ${meta.primaryTopic}\n` +
    dailyReport +
    `tags: [${meta.tags.join(", ")}]\n` +
    `---\n\n`
  );
}

function latestDailyReportLink(paths: string[]): string | undefined {
  const path = paths[paths.length - 1];
  return path ? dailyReportLink(path) : undefined;
}

function dailyReportLink(path: string, label?: string): string {
  const target = path.replace(/\.md$/i, "");
  const fallbackLabel = target.split("/").pop() || target;
  return `[[${target}|${label ?? fallbackLabel}]]`;
}

function weekdayName(dateStr: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateStr);
  if (!m) throw new Error(`Invalid date: ${dateStr}`);
  const date = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  return [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
  ][date.getUTCDay()];
}
