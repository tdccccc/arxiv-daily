import { afterEach, describe, expect, it } from "vitest";
import { access, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
  Logger,
  normalizeTopic,
  parseDailyReportTopicDirections,
} from "@arxiv-daily/core";
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
  it("generates a result-first daily report without rewriting saved reports", async () => {
    const root = await makeTempDir();
    const config = await loadCliConfig({
      configPath: join(root, "config.toml"),
      readText: async () => tomlForVault(root, join(root, ".cache")),
    });
    config.settings.arxiv.topics = [
      normalizeTopic({ name: "Empty topic", tag: "empty", description: "Unmatched research", detail: false }),
      normalizeTopic({ name: "Active topic", tag: "active", detail: false, directions: [
        { id: "d1", text: "Galaxy observations", origin: "manual" },
        { id: "d2", text: "Catalog comparisons", origin: "manual" },
      ] }),
      normalizeTopic({ name: "Limited topic", tag: "limited", description: "Other research", detail: false }),
    ];
    config.settings.output.maxDailyPapers = 1;
    const ids = ["2609.00001", "2609.00002"];
    // Public arXiv HTML/Atom and OpenAI-compatible SSE shapes. Only transport
    // is replaced; classification, content extraction, rendering and index writes run.
    const recent = `<html><body><dl id="articles"><h3>Tue, 8 Sep 2026</h3>${ids.map((id) =>
      `<dt><a title="Abstract">arXiv:${id}</a></dt><dd><div class="list-title">Title: Paper ${id}</div><div class="list-authors"><a>A. Author</a></div></dd>`,
    ).join("")}</dl></body></html>`;
    const atom = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${ids.map((id) =>
      `<entry><id>http://arxiv.org/abs/${id}v1</id><title>Paper ${id}</title><author><name>A. Author</name></author><summary>Galaxy observations.</summary><published>2026-09-08T00:00:00Z</published><updated>2026-09-08T00:00:00Z</updated><arxiv:primary_category term="astro-ph"/><category term="astro-ph"/></entry>`,
    ).join("")}</feed>`;
    const completions = [
      { papers: [
        { id: ids[0], category: "active", directions: ["active#1", "active#2"], relevanceScore: 95 },
        { id: ids[1], category: "limited", directions: ["limited#1"], relevanceScore: 70 },
      ] },
      { id: ids[0], coreProblem: "Research problem", keyMethod: "Measured method", mainResult: "Observed result", whyRelevant: "Research value", limitations: "Known limits" },
    ];
    const host = buildNodeHostAdapters({
      rootDir: root,
      progressStream: { write: () => {} },
      fetch: async (url, init) => {
        if (url.startsWith("https://arxiv.org/list/astro-ph/recent")) return new Response(recent);
        if (url.startsWith("https://export.arxiv.org/api/query?")) return new Response(atom);
        if (url === "https://arxiv.org/html/2609.00001") {
          return new Response('<html><body><div class="ltx_abstract">Galaxy observations.</div><h2>Results</h2><p>We observe a measured improvement.</p></body></html>');
        }
        if (url === "https://api.example.com/v1/chat/completions" && init?.method === "POST") {
          const completion = completions.shift();
          if (completion) return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: JSON.stringify(completion) }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`);
        }
        return new Response(`Unexpected fixture request: ${url}`, { status: 400 });
      },
    });
    const oldPath = "arxiv-daily/daily/2026-09-07.md";
    const oldMarkdown = "# Saved report\nKeep my annotations.\n";
    await host.storage.mkdir("arxiv-daily/daily");
    await host.storage.writeText(oldPath, oldMarkdown);
    const runtime = await buildCliRuntime(config, { host, logger: new Logger("error") });

    expect(await runtime.pipeline.runForDate("2026-09-08")).toMatchObject({ kind: "completed", papersWritten: 1 });
    const markdown = await host.storage.readText("arxiv-daily/daily/2026-09-08.md");
    expect(await runtime.pipeline.runForDate("2026-09-07")).toMatchObject({ kind: "completed" });
    expect(await host.storage.readText(oldPath)).toBe(oldMarkdown);
    expect((await runtime.paperIndex.get(ids[0]!))?.summary).toEqual({
      sourceSections: expect.stringContaining("Results"),
      coreProblem: "Research problem", keyMethod: "Measured method", mainResult: "Observed result",
      whyRelevant: "Research value", limitations: "Known limits",
    });
    expect(parseDailyReportTopicDirections(markdown, "2026-09-08")).toEqual({
      kind: "valid", occurrences: [{ arxivId: "2609.00001", hits: [
        { tag: "active", id: "d1", text: "Galaxy observations" },
        { tag: "active", id: "d2", text: "Catalog comparisons" },
      ] }],
    });
    expect.soft(markdown.match(/^## .+$/gm)).toEqual(["## Active topic", "## 其他关注主题"]);
    const otherTopics = markdown.split("## 其他关注主题")[1] ?? "";
    expect.soft(otherTopics).toMatch(/^[-*] .*Empty topic.*(?:未匹配|无相关论文).*$/m);
    expect.soft(otherTopics).toMatch(/^[-*] .*Limited topic.*上限.*1.*未展示.*$/m);
    expect.soft(markdown).toContain("> [!info]- 命中方向与信息来源");
    expect.soft(markdown).toContain("> - Galaxy observations\n> - Catalog comparisons");
    expect.soft(markdown).toContain("> [!abstract]- 研究背景、方法与边界");
    expect.soft(markdown).toContain("\n- **核心结果**: Observed result\n");
    expect.soft(markdown.match(/^- \*\*(?:研究问题|方法设计|核心结果|研究价值|适用边界)\*\*:/gm))
      .toEqual(["- **核心结果**:"]);
    const background = markdown.match(/^> \[!abstract\]-[^\n]*\n(?:>[^\n]*(?:\n|$))*/m)?.[0] ?? "";
    for (const line of [
      "> - **研究问题**: Research problem", "> - **方法设计**: Measured method",
      "> - **研究价值**: Research value", "> - **适用边界**: Known limits",
    ]) expect.soft(background).toContain(line);
  }, 15_000);

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

    const checkpointStores = (runtime.pipeline as any).deps.checkpointStores;
    expect(checkpointStores.filter).toBeInstanceOf(DailyFilterCheckpointStore);
    expect(checkpointStores.summary).toBeInstanceOf(DailySummaryCheckpointStore);
    for (const store of [checkpointStores.filter, checkpointStores.summary]) {
      expect(store.storage).toBe(host.storage);
      expect(store.output).toBe(config.settings.output);
      store.options.onWarning("checkpoint warning", new Error("store failed"));
    }
    expect(logger.getBuffer().filter((entry) =>
      entry.includes("checkpoint warning") && entry.includes("store failed")
    )).toHaveLength(2);

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
