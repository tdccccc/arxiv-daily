import { afterEach, describe, expect, it } from "vitest";
import { access, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { buildNodeHostAdapters } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../src/config";
import { buildCliRuntime } from "../src/runtime";

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

    const runtime = await buildCliRuntime(config, { host });

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
    expect(saved.papers["arxiv:2606.12345"].title).toBe("Runtime paper");

    await runtime.stateStore.setRunning("2026-06-13");
    await runtime.stateStore.setCompleted("2026-06-13", 2);
    expect(await host.storage.exists("arxiv-daily/.index/run-state.json")).toBe(
      true,
    );
  });

  it("passes configured detail selection policy to the pipeline", async () => {
    const root = await makeTempDir();
    const configuredPolicy = {
      profile: "custom" as const,
      normalThreshold: 81,
      exceptionalThreshold: 96,
      softLimit: 7,
    };
    const config = await loadCliConfig({
      cwd: root,
      env: { ARXIV_DAILY_API_KEY: "test-key" },
      readText: async () => JSON.stringify({ detailSelection: configuredPolicy }),
    });
    const host = buildNodeHostAdapters({ rootDir: config.vaultRoot });

    const runtime = await buildCliRuntime(config, { host });

    expect((runtime.pipeline as any).deps.detailSelection).toEqual(configuredPolicy);
    expect((runtime.pipeline as any).deps.detailSelection).toBe(
      config.settings.detailSelection,
    );
  });

  it("preserves legacy raw HTML cache files during runtime cleanup", async () => {
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
    const legacyDir = join(config.cacheDir, "html");
    const legacyPath = join(legacyDir, "d18a24abe03dd46c244455f5.html");
    await mkdir(legacyDir, { recursive: true });
    await writeFile(legacyPath, "legacy HTML");

    await buildCliRuntime(config, {
      host: buildNodeHostAdapters({ rootDir: config.vaultRoot }),
    });

    await expect(access(legacyPath)).resolves.toBeUndefined();
  });
});
