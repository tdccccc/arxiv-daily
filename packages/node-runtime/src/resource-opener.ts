import type {
  ResourceOpenOptions,
  ResourceOpener,
} from "@arxiv-daily/core";
import type { WritableTextStream } from "./progress";

export class StreamResourceOpener implements ResourceOpener {
  constructor(private stream: WritableTextStream = process.stdout) {}

  async openNote(path: string, _opts?: ResourceOpenOptions): Promise<void> {
    this.write(`note ${path}`);
  }

  async openDailyReport(
    path: string,
    _opts?: ResourceOpenOptions,
  ): Promise<void> {
    this.write(`daily ${path}`);
  }

  async openUrl(url: string): Promise<void> {
    this.write(`url ${url}`);
  }

  private write(line: string): void {
    this.stream.write(`[arxiv-daily] ${line}\n`);
  }
}
