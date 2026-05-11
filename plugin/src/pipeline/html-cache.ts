import * as fs from "node:fs/promises";
import * as path from "node:path";
import { createHash } from "node:crypto";

export interface HtmlCacheOptions {
  rootDir: string;
  expiryDays: number;
}

export class HtmlCache {
  constructor(private opts: HtmlCacheOptions) {}

  async get(key: string, kind: "html" | "abs"): Promise<string | null> {
    const p = this.pathFor(key, kind);
    try {
      const stat = await fs.stat(p);
      const ageDays = (Date.now() - stat.mtimeMs) / 86_400_000;
      if (ageDays > this.opts.expiryDays) {
        await fs.unlink(p).catch(() => {});
        return null;
      }
      return await fs.readFile(p, "utf8");
    } catch {
      return null;
    }
  }

  async set(key: string, kind: "html" | "abs", content: string): Promise<void> {
    const p = this.pathFor(key, kind);
    await fs.mkdir(path.dirname(p), { recursive: true });
    await fs.writeFile(p, content, "utf8");
  }

  private pathFor(key: string, kind: "html" | "abs"): string {
    const safe = createHash("sha1").update(key).digest("hex").slice(0, 24);
    return path.join(this.opts.rootDir, kind, `${safe}.html`);
  }
}
