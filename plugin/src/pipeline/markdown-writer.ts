import { type Vault, normalizePath } from "obsidian";
import type { Logger } from "../services/logger";
import type { ArxivSettings, OutputSettings } from "../settings/types";
import type { DailyPaperWithContent } from "./summarizer";

export interface MarkdownWriterOpts {
  vault: Vault;
  logger: Logger;
  arxiv: ArxivSettings;
  output: OutputSettings;
}

export class MarkdownWriter {
  constructor(private opts: MarkdownWriterOpts) {}

  async writeDaily(dateStr: string, summary: string): Promise<string> {
    const path = normalizePath(`${this.opts.output.dailyDir}/${dateStr}.md`);
    await this.ensureDir(this.opts.output.dailyDir);
    if (await this.opts.vault.adapter.exists(path)) {
      throw new Error(`daily already exists: ${path}`);
    }
    const frontmatter = `---\ndate: ${dateStr}\ntags: [arxiv, daily]\n---\n\n`;
    await this.opts.vault.adapter.write(path, frontmatter + summary);
    this.opts.logger.info(`wrote daily: ${path}`);
    return path;
  }

  async writePaperDetail(
    paper: DailyPaperWithContent,
    dateStr: string,
    summary: string,
  ): Promise<string> {
    const path = normalizePath(`${this.opts.output.papersDir}/${paper.id}.md`);
    await this.ensureDir(this.opts.output.papersDir);
    if (await this.opts.vault.adapter.exists(path)) {
      throw new Error(`paper already exists: ${path}`);
    }
    const tags = this.tagsFor(paper);
    const fm =
      `---\n` +
      `title: "${escapeYaml(paper.title)}"\n` +
      `authors: "${escapeYaml(paper.authors)}"\n` +
      `arxiv: "${paper.id}"\n` +
      `date: ${dateStr}\n` +
      `tags: [${tags.join(", ")}]\n` +
      `---\n\n`;
    await this.opts.vault.adapter.write(path, fm + summary);
    this.opts.logger.info(`wrote paper: ${path}`);
    return path;
  }

  async writeEmptyDaily(dateStr: string): Promise<string> {
    const summary = `# arXiv ${this.opts.arxiv.category} 每日追踪 ${dateStr}\n\n今日未发现相关论文。\n`;
    return this.writeDaily(dateStr, summary);
  }

  async dailyExists(dateStr: string): Promise<boolean> {
    const path = normalizePath(`${this.opts.output.dailyDir}/${dateStr}.md`);
    return await this.opts.vault.adapter.exists(path);
  }

  async paperDetailExists(id: string): Promise<boolean> {
    const path = normalizePath(`${this.opts.output.papersDir}/${id}.md`);
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

function escapeYaml(s: string): string {
  return s.replace(/"/g, '\\"');
}
