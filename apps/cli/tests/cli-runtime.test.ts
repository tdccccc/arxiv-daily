import { afterEach, describe, expect, it } from "vitest";
import { access, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DailySummaryCheckpointStore, Logger } from "@arxiv-daily/core";
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

function tomlForVault(vaultRoot: string, cacheDir: string): string {
  return `
schema_version = 1
vault_root = ${JSON.stringify(vaultRoot)}
cache_dir = ${JSON.stringify(cacheDir)}

[llm]
api_key = "test-key"
base_url = "https://api.example.com/v1"
model = "m"

[arxiv]
categories = ["astro-ph"]
timezone = "UTC"

[[arxiv.topics]]
name = "T"
tag = "t"
description = "topic"
detail = true

[output]
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
summary_language = "zh"
link_style = "wikilink"
`;
}

describe("CLI runtime", () => {
  it("builds pipeline dependencies on top of Node host adapters", async () => {
    const root = await makeTempDir();
    const cacheDir = join(root, ".cache");
    const config = await loadCliConfig({
      configPath: join(root, "config.toml"),
      readText: async () => tomlForVault(root, cacheDir),
    });
    const host = buildNodeHostAdapters({
      rootDir: config.vaultRoot,
      fetch: async () => new Response("ok", { status: 200 }),
    });
    const logger = new Logger("debug");

    const runtime = await buildCliRuntime(config, { host, logger });

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

    const checkpointStore = (runtime.pipeline as any).deps.checkpointStore;
    expect(checkpointStore).toBeInstanceOf(DailySummaryCheckpointStore);
    expect(checkpointStore.storage).toBe(host.storage);
    expect(checkpointStore.output).toBe(config.settings.output);
    checkpointStore.options.onWarning("checkpoint warning", new Error("store failed"));
    expect(logger.getBuffer().some((entry) =>
      entry.includes("checkpoint warning") && entry.includes("store failed")
    )).toBe(true);

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

  it("uses balanced detail selection (no profile surface in TOML)", async () => {
    const root = await makeTempDir();
    const cacheDir = join(root, ".cache");
    const config = await loadCliConfig({
      configPath: join(root, "config.toml"),
      readText: async () => tomlForVault(root, cacheDir),
    });
    const host = buildNodeHostAdapters({ rootDir: config.vaultRoot });
    const runtime = await buildCliRuntime(config, { host });

    expect(config.settings.detailSelection.profile).toBe("balanced");
    expect((runtime.pipeline as { deps: { detailSelection: unknown } }).deps.detailSelection).toEqual(
      config.settings.detailSelection,
    );
  });

  it("reuses persistent Atom metadata across CLI runtime instances", async () => {
    const root = await makeTempDir();
    const cacheDir = join(root, ".cache");
    const config = await loadCliConfig({
      configPath: join(root, "config.toml"),
      readText: async () => tomlForVault(root, cacheDir),
    });
    let requests = 0;
    const atom = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"><entry><id>http://arxiv.org/abs/2606.12345v1</id><title>Cached paper</title><author><name>A. Author</name></author><summary>Abstract.</summary><published>2026-06-13T00:00:00Z</published><updated>2026-06-14T00:00:00Z</updated><arxiv:primary_category term="astro-ph"/><category term="astro-ph"/></entry></feed>`;
    const buildHost = () => buildNodeHostAdapters({
      rootDir: config.vaultRoot,
      fetch: async () => { requests += 1; return new Response(atom, { status: 200 }); },
    });

    const first = await buildCliRuntime(config, { host: buildHost() });
    expect(await first.fetcher.fetchMetadataByIds(["2606.12345"])).toHaveProperty("size", 1);
    const second = await buildCliRuntime(config, { host: buildHost() });
    expect(await second.fetcher.fetchMetadataByIds(["2606.12345v2"])).toHaveProperty("size", 1);

    expect(requests).toBe(1);
    await expect(access(join(cacheDir, "atom-metadata", "2606.12345.json"))).resolves.toBeUndefined();
  });

  it("preserves legacy raw HTML cache files during runtime cleanup", async () => {
    const root = await makeTempDir();
    const cacheDir = join(root, ".cache");
    const config = await loadCliConfig({
      configPath: join(root, "config.toml"),
      readText: async () => tomlForVault(root, cacheDir),
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
