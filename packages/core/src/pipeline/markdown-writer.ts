import type { StorageAdapter } from "../core/adapters";
import type { Logger } from "../services/logger";
import type { ArxivSettings, OutputSettings } from "../settings/types";
import { formatArxivCategories } from "../settings/categories";
import {
  dailyHeader,
  noDailyPapersText,
} from "../settings/summary-language";
import type { DailyPaperWithContent } from "./summarizer";
import type { PaperIndexEntry } from "../services/paper-index";
import {
  appendGenerationMetrics,
  type GenerationMetrics,
} from "../metrics/generation";

export interface MarkdownWriterOpts {
  storage: StorageAdapter;
  logger: Logger;
  arxiv: ArxivSettings;
  output: OutputSettings;
}

export interface WriteDailyOptions {
  dateWindowNote?: string;
  metrics?: GenerationMetrics;
}

export interface WritePaperDetailOptions {
  metrics?: GenerationMetrics;
  replaceExisting?: boolean;
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
    await this.writeMarkdown(
      path,
      appendGenerationMetrics(
        frontmatter + dateWindowNote(options.dateWindowNote) + summary,
        options.metrics,
      ),
    );
    this.opts.logger.info(`wrote daily: ${path}`);
    return path;
  }

  async writePaperDetail(
    paper: DailyPaperWithContent,
    _dateStr: string,
    summary: string,
    indexEntry?: PaperIndexEntry,
    options: WritePaperDetailOptions = {},
  ): Promise<string> {
    const path = this.paperDetailPath(paper.id);
    await this.ensureDir(this.opts.output.papersDir);
    if ((await this.opts.storage.exists(path)) && !options.replaceExisting) {
      throw new Error(`paper already exists: ${path}`);
    }
    const tags = this.tagsFor(paper);
    const published = dateOnly(
      displayDateFromIndexEntry(indexEntry) ?? paper.published,
    );
    const publishedReport = await this.publishedReportLink(published);
    const fm = paperFrontmatter({
      title: paper.title,
      authors: paper.authors,
      arxivId: paper.id,
      primaryTopic: indexEntry?.primaryTopic ?? paper.category,
      published,
      publishedReport,
      tags,
    });
    await this.writeMarkdown(path, appendGenerationMetrics(fm + summary, options.metrics));
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
      published: dateOnly(displayDateFromIndexEntry(entry)),
      publishedReport: await this.publishedReportLink(
        dateOnly(displayDateFromIndexEntry(entry)),
      ),
      tags,
    });
    const noteBody =
      body ??
      `# ${entry.title}\n\n` +
        `- **arXiv**: [${entry.arxivId}](${entry.arxivUrl})\n` +
        `- **PDF**: [PDF](${entry.pdfUrl})\n\n` +
        `## Notes\n\n`;
    await this.writeMarkdown(path, fm + noteBody);
    this.opts.logger.info(`wrote paper note: ${path}`);
    return path;
  }

  async refreshPaperNoteFrontmatter(
    entry: PaperIndexEntry,
    paperPath?: string | null,
  ): Promise<string> {
    const path = this.opts.storage.normalizePath(
      paperPath || entry.paperPath || this.paperDetailPath(entry.arxivId),
    );
    const markdown = await this.opts.storage.readText(path);
    const body = stripFrontmatter(markdown).replace(/^\s+/, "");
    const fm = await this.paperFrontmatterForEntry(entry);
    await this.writeMarkdown(path, fm + body);
    this.opts.logger.info(`refreshed paper frontmatter: ${path}`);
    return path;
  }

  async writeEmptyDaily(
    dateStr: string,
    options: WriteDailyOptions = {},
  ): Promise<string> {
    const summary =
      `${dailyHeader(this.opts.output.summaryLanguage, formatArxivCategories(this.opts.arxiv), dateStr)}\n\n` +
      `${noDailyPapersText(this.opts.output.summaryLanguage)}\n`;
    return this.writeDaily(dateStr, summary, options);
  }

  async dailyExists(dateStr: string): Promise<boolean> {
    const path = this.dailyPath(dateStr);
    return await this.opts.storage.exists(path);
  }

  async readDaily(dateStr: string): Promise<string> {
    return this.opts.storage.readText(this.dailyPath(dateStr));
  }

  async paperDetailExists(id: string): Promise<boolean> {
    const path = this.paperDetailPath(id);
    return await this.opts.storage.exists(path);
  }

  async cleanupTemporaryFiles(): Promise<string[]> {
    const removed: string[] = [];
    for (const dir of [this.opts.output.dailyDir, this.opts.output.papersDir]) {
      const norm = this.opts.storage.normalizePath(dir);
      const entries = await this.opts.storage.list?.(norm).catch(() => []);
      for (const entry of entries ?? []) {
        if (entry.type !== "file" || !entry.path.endsWith(".tmp")) continue;
        await this.opts.storage.remove(entry.path);
        removed.push(entry.path);
      }
    }
    return removed.sort();
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

  private async writeMarkdown(path: string, content: string): Promise<void> {
    if (this.opts.storage.writeTextAtomic) {
      await this.opts.storage.writeTextAtomic(path, content);
      return;
    }
    const tmp = `${path}.tmp`;
    const bak = `${path}.bak`;
    if (await this.opts.storage.exists(tmp)) await this.opts.storage.remove(tmp);
    await this.opts.storage.writeText(tmp, content);
    if (!(await this.opts.storage.exists(path))) {
      await this.opts.storage.rename(tmp, path);
      return;
    }
    if (await this.opts.storage.exists(bak)) await this.opts.storage.remove(bak);
    await this.opts.storage.rename(path, bak);
    try {
      await this.opts.storage.rename(tmp, path);
      await this.opts.storage.remove(bak);
    } catch (e) {
      if (await this.opts.storage.exists(bak)) {
        await this.opts.storage.rename(bak, path);
      }
      throw e;
    }
  }

  private async publishedReportLink(
    published: string | undefined,
  ): Promise<string | undefined> {
    if (!published) return undefined;
    const path = this.dailyPath(published);
    return (await this.opts.storage.exists(path))
      ? dailyReportLink(path, published)
      : undefined;
  }

  private async paperFrontmatterForEntry(entry: PaperIndexEntry): Promise<string> {
    const topic = this.opts.arxiv.topics.find((t) => t.tag === entry.primaryTopic);
    const tags = ["arxiv", "paper", topic?.tag ?? entry.primaryTopic].filter(Boolean);
    const published = dateOnly(displayDateFromIndexEntry(entry));
    return paperFrontmatter({
      title: entry.title,
      authors: entry.authors.join(", "),
      arxivId: entry.arxivId,
      primaryTopic: entry.primaryTopic,
      published,
      publishedReport: await this.publishedReportLink(published),
      tags,
    });
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
  published?: string;
  publishedReport?: string;
  tags: string[];
}): string {
  const published = meta.publishedReport
    ? `published: "${escapeYaml(meta.publishedReport)}"\n`
    : meta.published
      ? `published: ${meta.published}\n`
    : "";
  return (
    `---\n` +
    `title: "${escapeYaml(meta.title)}"\n` +
    `authors: "${escapeYaml(meta.authors)}"\n` +
    `arxiv_id: "${meta.arxivId}"\n` +
    `primary_topic: ${meta.primaryTopic}\n` +
    published +
    `tags: [${meta.tags.join(", ")}]\n` +
    `---\n\n`
  );
}

function dailyReportLink(path: string, label?: string): string {
  const target = path.replace(/\.md$/i, "");
  const fallbackLabel = target.split("/").pop() || target;
  return `[[${target}|${label ?? fallbackLabel}]]`;
}

function displayDateFromIndexEntry(
  entry: Pick<PaperIndexEntry, "dailyReports" | "published"> | undefined,
): string | undefined {
  if (!entry) return undefined;
  return firstDailyReportDate(entry.dailyReports) ?? entry.published;
}

function firstDailyReportDate(paths: string[]): string | undefined {
  const dates = paths
    .map((path) => /(\d{4}-\d{2}-\d{2})\.md$/i.exec(path.trim())?.[1])
    .filter((date): date is string => Boolean(date))
    .sort();
  return dates[0];
}

function stripFrontmatter(markdown: string): string {
  const trimmedStart = markdown.trimStart();
  const leading = markdown.length - trimmedStart.length;
  const match = /^---\s*\n[\s\S]*?\n---\s*(?:\n|$)/.exec(trimmedStart);
  return match ? trimmedStart.slice(match[0].length) : markdown.slice(leading);
}

function dateOnly(value: string | undefined): string | undefined {
  const trimmed = value?.trim() ?? "";
  const match = /^(\d{4}-\d{2}-\d{2})/.exec(trimmed);
  return match?.[1] ?? (trimmed || undefined);
}

function weekdayName(dateStr: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateStr);
  if (!m) throw new Error(`Invalid date: ${dateStr}`);
  const [, year, month, day] = m;
  if (!year || !month || !day) throw new Error(`Invalid date: ${dateStr}`);
  const date = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day)));
  const weekdays = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
  ];
  return weekdays[date.getUTCDay()] ?? "Sunday";
}
