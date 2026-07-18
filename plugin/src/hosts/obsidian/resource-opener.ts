import type { App } from "obsidian";
import type {
  ResourceOpenOptions,
  ResourceOpener,
} from "@arxiv-daily/core";

export class ObsidianResourceOpener implements ResourceOpener {
  constructor(private app: App) {}

  async openNote(path: string, opts?: ResourceOpenOptions): Promise<void> {
    await this.openMarkdownPath(path, opts);
  }

  async openDailyReport(
    path: string,
    opts?: ResourceOpenOptions,
  ): Promise<void> {
    await this.openMarkdownPath(path, opts);
  }

  async openUrl(url: string): Promise<void> {
    window.activeWindow.open(url, "_blank", "noopener");
  }

  private async openMarkdownPath(
    path: string,
    opts?: ResourceOpenOptions,
  ): Promise<void> {
    await this.app.workspace.openLinkText(path, "", opts?.newLeaf ?? false);
  }
}
