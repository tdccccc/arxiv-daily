import { afterEach, describe, expect, it, vi } from "vitest";
import { access, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { parse, stringify } from "smol-toml";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
  LibraryWorkflow,
  Logger,
  authorizeLibraryConnection,
  createLibraryConnection,
  normalizeTopic,
  parseDailyReportPersonalNovelty,
  parseDailyReportTopicDirections,
  type ChatMessage,
  type DocumentParser,
  type EmbeddingModel,
} from "@arxiv-daily/core";
import {
  NodeStorageAdapter,
  buildNodeHostAdapters,
  openScopedLibrarySource,
  type FetchLike,
} from "@arxiv-daily/node-runtime";
import { createCliTopicSettings } from "../src/library-topic-settings";
import { loadCliConfig } from "../src/config";
import { buildCliRuntime, type CliRuntime } from "../src/runtime";

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
  it("does not run a daily pipeline while another host owns the Vault run lock", async () => {
    const root = await makeTempDir();
    const config = await loadCliConfig({
      configPath: join(root, "config.toml"),
      readText: async () => tomlForVault(root, join(root, ".cache")),
    });
    const host = buildNodeHostAdapters({ rootDir: root });
    const runtime = await buildCliRuntime(config, { host });
    const run = vi.spyOn(runtime.pipeline, "runForDate").mockResolvedValue({ kind: "completed", papersWritten: 1 });
    expect(host.storage.acquireLock).toBeTypeOf("function");
    const other = buildNodeHostAdapters({ rootDir: root });
    const lease = await other.storage.acquireLock!("daily-run");
    expect(lease).not.toBeNull();
    try {
      await runtime.scheduler.runForDateNow("2026-09-28");
      expect(run).not.toHaveBeenCalled();
    } finally {
      await lease!.release();
    }
    await runtime.scheduler.runForDateNow("2026-09-28");
    expect(run).toHaveBeenCalledOnce();
  });

  it("preserves another host's temporary Markdown during startup and cleans it after release", async () => {
    const root = await makeTempDir();
    const config = await loadCliConfig({
      configPath: join(root, "config.toml"),
      readText: async () => tomlForVault(root, join(root, ".cache")),
    });
    const active = buildNodeHostAdapters({ rootDir: root });
    const temp = "arxiv-daily/daily/2026-09-28.md.tmp";
    await active.storage.mkdir("arxiv-daily/daily");
    await active.storage.writeText(temp, "still writing");
    const lease = await active.storage.acquireLock!("daily-run");
    expect(lease).not.toBeNull();
    try {
      await buildCliRuntime(config);
      expect(await active.storage.readText(temp)).toBe("still writing");
    } finally {
      await lease!.release();
    }
    await buildCliRuntime(config);
    expect(await active.storage.exists(temp)).toBe(false);
  });

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

const NOVELTY_MARKER = "arxiv-daily-personal-novelty";
const NOVELTY_DATE = "2026-10-01";
const NOVELTY_DAILY_ID = "2610.10001";
const NOVELTY_MODEL_URL = "https://model.example/v1";
const NOVELTY_LIBRARY_IDS = ["2610.00001", "2610.00002", "2610.00003"];

function libraryMetadata(id: string): { title: string; abstract: string } {
  const title = id === NOVELTY_DAILY_ID
    ? "Reliable agent evaluation with controlled interventions"
    : `Agent evaluation baseline study ${id}`;
  return { title, abstract: `${title}. Agents are evaluated with experimental methods against reproducible baselines.` };
}

function libraryAtom(ids: string[]): string {
  return `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${ids.map((id) => {
    const paper = libraryMetadata(id);
    return `<entry><id>https://arxiv.org/abs/${id}v1</id><title>${paper.title}</title><author><name>A. Researcher</name></author><summary>${paper.abstract}</summary><published>2026-10-01T00:00:00Z</published><updated>2026-10-01T00:00:00Z</updated><arxiv:primary_category term="cs.AI"/><category term="cs.AI"/></entry>`;
  }).join("")}</feed>`;
}

/** Only network transport is controlled; the real pipeline, LlmClient, fetcher, stores and search run. */
function noveltyTransport() {
  const noveltyPayloads: Array<{ paper: { paperKey: string }; basis: Array<{ paperKey: string; title: string }> }> = [];
  const embeddingRequests: string[] = [];
  const fetch: FetchLike = async (input, init) => {
    const url = new URL(String(input));
    if (url.origin === "https://model.example") {
      const { messages } = JSON.parse(String(init?.body)) as { messages: ChatMessage[] };
      const system = messages[0]?.content ?? "";
      const user = messages[1]?.content ?? "";
      let content: string;
      if (user.startsWith("Compare this new paper")) {
        const body = user.split("<paper_data>\n")[1]!.split("\n</paper_data>")[0]!;
        const payload = JSON.parse(body) as (typeof noveltyPayloads)[number];
        noveltyPayloads.push(payload);
        content = JSON.stringify({
          differenceType: "new-method",
          comparisonBasis: [(payload.basis.find(({ paperKey }) => paperKey.startsWith("file:")) ?? payload.basis[0]!).paperKey],
          evidenceDepth: "metadata-and-abstract", explanation: "Adds controlled interventions the library papers do not use.",
        });
      } else if (system.includes("为所有命中论文评分")) {
        content = JSON.stringify({ papers: [{ id: NOVELTY_DAILY_ID, category: "astronomy", directions: ["astronomy#1"], relevanceScore: 90 }] });
      } else if (system.includes("严格 JSON 对象") || system.includes("strict JSON object")) {
        content = JSON.stringify({ id: NOVELTY_DAILY_ID, coreProblem: "Reliable measurement", keyMethod: "Controlled intervention", mainResult: "Improves the measured outcome", whyRelevant: "Supports the topic", limitations: "Small sample" });
      } else {
        throw new Error(`Unexpected model task: ${system.slice(0, 120)}`);
      }
      return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`, {
        status: 200, headers: { "content-type": "text/event-stream" },
      });
    }
    if (url.pathname.endsWith("/embeddings")) {
      embeddingRequests.push(url.toString());
      throw new Error("lexical novelty retrieval must not call an embedding endpoint");
    }
    if (url.hostname === "export.arxiv.org" && url.searchParams.has("id_list")) {
      return new Response(libraryAtom(url.searchParams.get("id_list")!.split(",")), { status: 200 });
    }
    if (url.hostname === "arxiv.org" && url.pathname === "/list/cs.AI/recent") {
      return new Response(`<html><body><dl id="articles"><h3>Thu, 1 Oct 2026 (showing 1 of 1 entries)</h3><dt><a title="Abstract" href="/abs/${NOVELTY_DAILY_ID}">arXiv:${NOVELTY_DAILY_ID}</a></dt><dd><div class="list-title">Title: ${libraryMetadata(NOVELTY_DAILY_ID).title}</div><div class="list-authors"><a>A. Researcher</a></div></dd></dl></body></html>`, { status: 200 });
    }
    if (url.hostname === "arxiv.org" && url.pathname.startsWith("/html/")) {
      const id = url.pathname.slice("/html/".length);
      return new Response(`<html><body><div class="ltx_abstract">${libraryMetadata(id).abstract}</div><h2>Results</h2><p>The method improves measured reliability.</p></body></html>`, { status: 200 });
    }
    throw new Error(`Unexpected network request: ${url}`);
  };
  return { fetch, noveltyPayloads, embeddingRequests };
}

/** An authorized (or not), fully indexed local library plus a daily paper that resembles it. */
async function noveltyFixture(authorized: boolean, extraFiles: Record<string, string> = {}) {
  const root = await makeTempDir();
  const selectedRoot = join(root, "library");
  const vaultRoot = join(root, "vault");
  await mkdir(selectedRoot);
  await mkdir(vaultRoot);
  await Promise.all(NOVELTY_LIBRARY_IDS.map((id) =>
    writeFile(join(selectedRoot, `${id}.pdf`), `agent evaluation experimental methods baselines ${id}. `.repeat(30))));
  await Promise.all(Object.entries(extraFiles).map(([name, text]) =>
    writeFile(join(selectedRoot, name), text)));
  const source = await openScopedLibrarySource(selectedRoot);
  const created = createLibraryConnection(source.canonicalRoot, source.rootIdentity);
  const connection = authorized ? authorizeLibraryConnection(created, { llmBaseUrl: NOVELTY_MODEL_URL }) : created;
  const configPath = join(root, "config.toml");
  await writeFile(configPath, stringify({
    schema_version: 1, vault_root: vaultRoot, cache_dir: join(root, "cache"),
    llm: { api_key: "fixture-key", base_url: NOVELTY_MODEL_URL, model: "controlled-model" },
    embedding: { mode: "local" },
    arxiv: { categories: ["cs.AI"], timezone: "UTC", topics: [{ name: "Astronomy", tag: "astronomy", description: "Agent evaluation", detail: false }] },
    output: { summary_language: "en" }, library: connection,
  }));
  const config = await loadCliConfig({ configPath });
  const network = noveltyTransport();
  const host = buildNodeHostAdapters({ rootDir: vaultRoot, fetch: network.fetch });
  host.storage = new NodeStorageAdapter(vaultRoot, { lockRoot: join(root, "locks") });
  const logger = new Logger("error");
  const seed = await buildCliRuntime(config, { host, logger });
  const parser: DocumentParser = {
    capabilities: ["page-text"], provenance: { id: "controlled-parser", version: "1" },
    parse: async (bytes) => ({ mediaType: "application/pdf", blocks: [{ kind: "page", text: new TextDecoder().decode(bytes), locator: { page: 1, block: 0 } }] }),
  };
  // The local embedding identity, so the generation built here is the one the daily search opens.
  const embedding: EmbeddingModel = {
    modelId: "multilingual-e5-small-q8", dimension: 384, prefixPolicy: "none",
    embed: async (texts) => texts.map(() => new Float32Array(384).fill(0.01)),
  };
  const workflow = new LibraryWorkflow({
    topicSettings: createCliTopicSettings(config),
    storage: host.storage, output: config.settings.output, connection, source, http: host.http,
    arxivFetcher: seed.fetcher, llm: seed.llm,
    createParser: async () => ({ parser }), createEmbedding: async () => embedding,
    assertCurrent: () => undefined, assertAuthorized: () => undefined,
  });
  await workflow.scan();
  expect(await workflow.index()).toMatchObject({ indexed: 3 + Object.keys(extraFiles).length, failed: 0 });
  seed.dispose?.();
  network.noveltyPayloads.length = 0;
  return { config, host, logger, network, configPath };
}

describe("CLI runtime library novelty", () => {
  const runtimes: CliRuntime[] = [];
  afterEach(() => {
    for (const runtime of runtimes.splice(0)) runtime.dispose?.();
  });

  it("annotates the daily report with library novelty when library processing is authorized", async () => {
    const f = await noveltyFixture(true);
    const runtime = await buildCliRuntime(f.config, { host: f.host, logger: f.logger });
    runtimes.push(runtime);

    expect(await runtime.pipeline.runForDate(NOVELTY_DATE), f.logger.getBuffer().join("\n"))
      .toMatchObject({ kind: "completed", papersWritten: 1 });

    const daily = await f.host.storage.readText(runtime.writer.dailyPath(NOVELTY_DATE));
    expect(daily).toContain(NOVELTY_MARKER);
    expect(daily).toContain("Personal novelty: new method vs. prior papers: Agent evaluation baseline study");
    expect(daily).toContain("Adds controlled interventions the library papers do not use");
    expect(f.network.noveltyPayloads).toHaveLength(1);
    const basisKeys = f.network.noveltyPayloads[0]!.basis.map(({ paperKey }) => paperKey);
    expect(basisKeys.length).toBeGreaterThan(0);
    for (const key of basisKeys) {
      expect(NOVELTY_LIBRARY_IDS.map((id) => `arxiv:${id}`)).toContain(key);
    }
    expect(f.network.embeddingRequests).toEqual([]);
  }, 30_000);

  it("compares against an indexed library PDF that has no arXiv identity, shown by title only", async () => {
    // An unrecognised filename: the index keys it file:sha256:... with an extracted title and abstract.
    const fileTitle = "Controlled interventions for reliable agent evaluation experiments";
    const f = await noveltyFixture(true, {
      "lecture-notes-final.pdf": `${fileTitle}\n\nAbstract\n${"Reliable agent evaluation uses controlled interventions against experimental baselines and reproducible methods. ".repeat(3)}\n\n1 Introduction\nBody text.`,
    });
    const runtime = await buildCliRuntime(f.config, { host: f.host, logger: f.logger });
    runtimes.push(runtime);

    expect(await runtime.pipeline.runForDate(NOVELTY_DATE), f.logger.getBuffer().join("\n"))
      .toMatchObject({ kind: "completed", papersWritten: 1 });

    const payload = f.network.noveltyPayloads[0]!;
    const fileBasis = payload.basis.find(({ paperKey }) => paperKey.startsWith("file:sha256:"));
    expect(fileBasis, JSON.stringify(payload.basis.map(({ paperKey }) => paperKey))).toBeDefined();
    expect(Object.keys(fileBasis!)).toEqual(["paperKey", "title", "abstract"]);
    expect(fileBasis!.title).toContain("reliable agent evaluation");
    const daily = await f.host.storage.readText(runtime.writer.dailyPath(NOVELTY_DATE));
    const parsed = parseDailyReportPersonalNovelty(daily, NOVELTY_DATE);
    expect(parsed.kind).toBe("valid");
    if (parsed.kind !== "valid") return;
    expect(parsed.occurrences).toHaveLength(1);
    expect(parsed.occurrences[0]!.novelty.comparisonBasis).toEqual([fileBasis!.paperKey]);
    expect(daily).toContain(`prior papers: ${fileBasis!.title.replace(/[.]/g, "\\.")}`);
    expect(daily).not.toContain("file:sha256");
  }, 30_000);

  it("writes the same report without novelty when library processing is not authorized", async () => {
    const f = await noveltyFixture(false);
    const runtime = await buildCliRuntime(f.config, { host: f.host, logger: f.logger });
    runtimes.push(runtime);

    expect(await runtime.pipeline.runForDate(NOVELTY_DATE), f.logger.getBuffer().join("\n"))
      .toMatchObject({ kind: "completed", papersWritten: 1 });

    const daily = await f.host.storage.readText(runtime.writer.dailyPath(NOVELTY_DATE));
    expect(daily).not.toContain(NOVELTY_MARKER);
    expect(daily).toContain(libraryMetadata(NOVELTY_DAILY_ID).title);
    expect(f.network.noveltyPayloads).toEqual([]);
  }, 30_000);
});
