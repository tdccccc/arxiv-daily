import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { buildNodeHostAdapters } from "../src/hosts/node";
import { loadCliConfig } from "../src/cli/config";
import { buildCliRuntime } from "../src/cli/runtime";

const tempDirs: string[] = [];

afterEach(async () => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

async function makeTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "arxiv-daily-runtime-"));
  tempDirs.push(dir);
  return dir;
}

describe("CLI runtime", () => {
  it("builds pipeline dependencies on top of Node host adapters", async () => {
    const root = await makeTempDir();
    const config = await loadCliConfig({
      cwd: root,
      env: { ARXIV_DAILY_API_KEY: "test-key" },
      readText: async () => {
        const err = new Error("missing") as NodeJS.ErrnoException;
        err.code = "ENOENT";
        throw err;
      },
    });
    const host = buildNodeHostAdapters({
      rootDir: config.vaultRoot,
      fetch: async () => new Response("ok", { status: 200 }),
    });

    const runtime = buildCliRuntime(config, { host });

    expect(runtime.writer.dailyPath("2026-06-13")).toBe(
      "arxiv-daily/daily/2026-06-13.md",
    );
    await runtime.paperIndex.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "Runtime paper",
      authors: "A. Author",
      date: "2026-06-13",
      arxivCategory: "astro-ph",
      primaryTopic: "astro",
      detail: false,
    });

    expect(await host.storage.exists("arxiv-daily/.index/papers.json")).toBe(
      true,
    );
    const saved = JSON.parse(
      await host.storage.readText("arxiv-daily/.index/papers.json"),
    );
    expect(saved.papers["2606.12345"].title).toBe("Runtime paper");
  });
});
