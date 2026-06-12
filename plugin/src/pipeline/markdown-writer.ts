import { type Vault, normalizePath } from "obsidian";
import type { Logger } from "../services/logger";
import type { ArxivSettings, OutputSettings } from "../settings/types";
import type { DailyPaperWithContent } from "./summarizer";
import type { PaperIndexEntry } from "../services/paper-index";

export interface MarkdownWriterOpts {
  vault: Vault;
  logger: Logger;
  arxiv: ArxivSettings;
  output: OutputSettings;
}

export interface DailyMissedPaper {
  id: string;
  title: string;
  authors: string;
}

export interface WriteDailyOptions {
  missedPapers?: DailyMissedPaper[];
}

export class MarkdownWriter {
  constructor(private opts: MarkdownWriterOpts) {}

  dailyPath(dateStr: string): string {
    return normalizePath(`${this.opts.output.dailyDir}/${dateStr}.md`);
  }

  paperDetailPath(id: string): string {
    return normalizePath(`${this.opts.output.papersDir}/${id}.md`);
  }

  async writeDaily(
    dateStr: string,
    summary: string,
    options: WriteDailyOptions = {},
  ): Promise<string> {
    const path = this.dailyPath(dateStr);
    await this.ensureDir(this.opts.output.dailyDir);
    if (await this.opts.vault.adapter.exists(path)) {
      throw new Error(`daily already exists: ${path}`);
    }
    const frontmatter =
      `---\n` +
      `date: ${dateStr}\n` +
      `weekday: ${weekdayName(dateStr)}\n` +
      `tags: [arxiv, daily]\n` +
      `---\n\n`;
    await this.opts.vault.adapter.write(
      path,
      frontmatter + appendMissedPapers(summary, options.missedPapers ?? []),
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
    if (await this.opts.vault.adapter.exists(path)) {
      throw new Error(`paper already exists: ${path}`);
    }
    const tags = this.tagsFor(paper);
    const fm = paperFrontmatter({
      title: paper.title,
      authors: paper.authors,
      arxivId: paper.id,
      date: dateStr,
      primaryTopic: indexEntry?.primaryTopic ?? paper.category,
      status: indexEntry?.status ?? "inbox",
      priority: indexEntry?.priority ?? "normal",
      seenDates: indexEntry?.seenDates ?? [dateStr],
      zoteroKey: indexEntry?.zoteroKey ?? "",
      citationKey: indexEntry?.citationKey ?? "",
      tags,
    });
    await this.opts.vault.adapter.write(path, fm + summary);
    this.opts.logger.info(`wrote paper: ${path}`);
    return path;
  }

  async writePaperNote(entry: PaperIndexEntry, body?: string): Promise<string> {
    const path = entry.paperPath ?? this.paperDetailPath(entry.arxivId);
    await this.ensureDir(this.opts.output.papersDir);
    if (await this.opts.vault.adapter.exists(path)) {
      throw new Error(`paper already exists: ${path}`);
    }
    const topic = this.opts.arxiv.topics.find((t) => t.tag === entry.primaryTopic);
    const tags = ["arxiv", "paper", topic?.tag ?? entry.primaryTopic].filter(Boolean);
    const fm = paperFrontmatter({
      title: entry.title,
      authors: entry.authors.join(", "),
      arxivId: entry.arxivId,
      date: entry.updated || entry.published,
      primaryTopic: entry.primaryTopic,
      status: entry.status,
      priority: entry.priority,
      seenDates: entry.seenDates,
      zoteroKey: entry.zoteroKey,
      citationKey: entry.citationKey,
      tags,
    });
    const noteBody =
      body ??
      `# ${entry.title}\n\n` +
        `- **arXiv**: [${entry.arxivId}](${entry.arxivUrl})\n` +
        `- **PDF**: [PDF](${entry.pdfUrl})\n\n` +
        `## Notes\n\n`;
    await this.opts.vault.adapter.write(path, fm + noteBody);
    this.opts.logger.info(`wrote paper note: ${path}`);
    return path;
  }

  async writeEmptyDaily(
    dateStr: string,
    options: WriteDailyOptions = {},
  ): Promise<string> {
    const summary = `# arXiv ${this.opts.arxiv.category} 每日追踪 ${dateStr}\n\n今日未发现相关论文。\n`;
    return this.writeDaily(dateStr, summary, options);
  }

  async dailyExists(dateStr: string): Promise<boolean> {
    const path = this.dailyPath(dateStr);
    return await this.opts.vault.adapter.exists(path);
  }

  async paperDetailExists(id: string): Promise<boolean> {
    const path = this.paperDetailPath(id);
    return await this.opts.vault.adapter.exists(path);
  }

  private tagsFor(paper: DailyPaperWithContent): string[] {
    const tags = ["arxiv", "paper"];
    const topic = this.opts.arxiv.topics.find((t) => t.tag === paper.category);
    if (topic) tags.push(topic.tag);
    return tags;
  }

  private async ensureDir(rel: string): Promise<void> {
    const norm = normalizePath(rel);
    if (!(await this.opts.vault.adapter.exists(norm))) {
      await this.opts.vault.adapter.mkdir(norm);
    }
  }

}

function appendMissedPapers(
  summary: string,
  missedPapers: DailyMissedPaper[],
): string {
  if (missedPapers.length === 0) return summary;
  const body = summary.trimEnd();
  return `${body}\n\n${renderMissedPapers(missedPapers)}\n`;
}

function renderMissedPapers(missedPapers: DailyMissedPaper[]): string {
  const lines = missedPapers.map((paper) => {
    const title = compactText(paper.title) || paper.id;
    const authors = compactText(paper.authors);
    const suffix = authors ? `（${authors}）` : "";
    return `- [${paper.id}](https://arxiv.org/abs/${paper.id}) — ${title}${suffix}`;
  });
  return [
    `<details>`,
    `<summary>未入选论文（可能漏报） · ${missedPapers.length} 篇</summary>`,
    "",
    ...lines,
    "",
    `</details>`,
  ].join("\n");
}

function compactText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function escapeYaml(s: string): string {
  return s.replace(/"/g, '\\"');
}

function paperFrontmatter(meta: {
  title: string;
  authors: string;
  arxivId: string;
  date: string;
  primaryTopic: string;
  status: string;
  priority: string;
  seenDates: string[];
  zoteroKey: string;
  citationKey: string;
  tags: string[];
}): string {
  return (
    `---\n` +
    `type: paper\n` +
    `source: arxiv\n` +
    `title: "${escapeYaml(meta.title)}"\n` +
    `authors: "${escapeYaml(meta.authors)}"\n` +
    `arxiv_id: "${meta.arxivId}"\n` +
    `arxiv: "${meta.arxivId}"\n` +
    `date: ${meta.date}\n` +
    `weekday: ${weekdayName(meta.date)}\n` +
    `status: ${meta.status}\n` +
    `priority: ${meta.priority}\n` +
    `primary_topic: ${meta.primaryTopic}\n` +
    `seen_dates:\n` +
    meta.seenDates.map((d) => `  - "${d}"\n`).join("") +
    `zotero_key: "${escapeYaml(meta.zoteroKey)}"\n` +
    `citation_key: "${escapeYaml(meta.citationKey)}"\n` +
    `tags: [${meta.tags.join(", ")}]\n` +
    `---\n\n`
  );
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
