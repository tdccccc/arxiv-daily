import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { stringify } from "smol-toml";
import {
  LibraryWorkflow,
  Logger,
  authorizeLibraryConnection,
  createLibraryConnection,
  parseDailyReportDiscoveryProvenance,
  parseDailyReportPersonalNovelty,
  revokeLibraryConnection,
  type ChatMessage,
  type DocumentParser,
  type EmbeddingModel,
} from "@arxiv-daily/core";
import { buildNodeHostAdapters, NodeStorageAdapter, openScopedLibrarySource, type FetchLike } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../src/config";
import { runCli } from "../src/main";
import { buildCliRuntime, type CliRuntime } from "../src/runtime";

const DATE = "2026-10-01";
const MANUAL_ID = "2610.10001";
const LIBRARY_ID = "2610.10002";
const DIRECTION_ID = "confirmed-agent-evaluation";
const MODEL_URL = "https://model.example/v1";
const NOVELTY = "Adds intervention-based evaluation absent from the representative abstracts.";
const roots: string[] = [];
const runtimes: CliRuntime[] = [];
afterEach(async () => {
  for (const runtime of runtimes.splice(0)) runtime.dispose?.();
  await Promise.all(roots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })));
});

function metadata(id: string) {
  const title = id === MANUAL_ID ? "Gravitational wave source populations"
    : id === LIBRARY_ID ? "Intervention-based agent evaluation"
      : `Library ${Number(id.slice(-5)) <= 3 ? "agents" : "vision"} research ${id}`;
  return { title, abstract: `${title}. We evaluate methods against reproducible experimental baselines.` };
}

function atom(ids: string[]): string {
  return `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${ids.map((id) => {
    const paper = metadata(id);
    return `<entry><id>https://arxiv.org/abs/${id}v1</id><title>${paper.title}</title><author><name>A. Researcher</name></author><summary>${paper.abstract}</summary><published>2026-10-01T00:00:00Z</published><updated>2026-10-01T00:00:00Z</updated><arxiv:primary_category term="cs.AI"/><category term="cs.AI"/></entry>`;
  }).join("")}</feed>`;
}

const recent = `<html><body><dl id="articles"><h3>Thu, 1 Oct 2026 (showing 2 of 2 entries)</h3>${[MANUAL_ID, LIBRARY_ID].map((id) =>
  `<dt><a title="Abstract" href="/abs/${id}">arXiv:${id}</a></dt><dd><div class="list-title">Title: ${metadata(id).title}</div><div class="list-authors"><a>A. Researcher</a></div></dd>`,
).join("")}</dl></body></html>`;

function paperHtml(id: string): string {
  return `<html><body><div class="ltx_abstract">${metadata(id).abstract}</div>
<h2>Introduction</h2><p>We study reliable evaluation of research methods.</p>
<h2>Methods</h2><p>We introduce controlled interventions and compare against a baseline.</p>
<h2>Results</h2><p>The method improves measured reliability on a bounded test set.</p>
<h2>Conclusions</h2><p>The evidence supports further evaluation under distribution shift.</p></body></html>`;
}

function fencedData(messages: ChatMessage[]): unknown {
  const user = messages.find(({ role }) => role === "user")?.content ?? "";
  const match = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(user);
  if (!match) throw new Error("Expected a bounded paper-data model request");
  return JSON.parse(match[1]!.replaceAll("&lt;/paper_data&gt;", "</paper_data>"));
}

/** Only the network transport is controlled; real LlmClient and ArxivFetcher decode every response. */
function transport() {
  const calls: Array<{ kind: string; messages: ChatMessage[] }> = [];
  const requests: string[] = [];
  const fetch: FetchLike = async (input, init) => {
    const url = new URL(String(input));
    requests.push(url.toString());
    if (url.origin === "https://model.example") {
      const { messages } = JSON.parse(String(init?.body)) as { messages: ChatMessage[] };
      const system = messages[0]?.content ?? "";
      let kind: string;
      let content: string;
      if (system.includes("You propose provisional research directions")) {
        kind = "propose";
        const papers = fencedData(messages) as Array<{ paperKey: string }>;
        content = JSON.stringify({ candidates: [{
          name: "Agent evaluation", description: "Reliable evaluation of autonomous research agents.",
          discoveryCues: ["agent evaluation", "reliability"],
          representativePaperKeys: papers.slice(0, 3).map(({ paperKey }) => paperKey),
        }] });
      } else if (system.includes("选择最匹配的主题")) {
        kind = "manual-filter";
        // The same deterministic model answer in both runs excludes the agent paper.
        content = JSON.stringify({ papers: [{ id: MANUAL_ID, category: "astronomy" }] });
      } else if (system.includes("You classify new arXiv papers against researcher-confirmed directions")) {
        kind = "library-filter";
        const payload = fencedData(messages) as { papers: Array<{ paperKey: string }>; directions: Array<{ id: string }> };
        expect(payload.directions.map(({ id }) => id)).toContain(DIRECTION_ID);
        content = JSON.stringify({ papers: payload.papers.map(({ paperKey }) => ({
          paperKey, directionIds: paperKey === `arxiv:${LIBRARY_ID}` ? [DIRECTION_ID] : [],
        })) });
      } else if (system.includes("compare one new arXiv paper against its representative prior papers")) {
        kind = "novelty";
        const payload = fencedData(messages) as { paper: { paperKey: string }; basis: Array<{ paperKey: string; abstract: string }> };
        content = JSON.stringify({ differenceType: "new-method", comparisonBasis: [payload.basis[0]!.paperKey], evidenceDepth: "metadata-and-abstract", explanation: NOVELTY });
      } else if (system.includes("严格 JSON 对象") || system.includes("strict JSON object")) {
        kind = "summary";
        const id = /ID: (\d{4}\.\d{4,5})/.exec(messages[1]?.content ?? "")?.[1];
        if (!id) throw new Error("Missing paper ID in daily summary prompt");
        content = JSON.stringify({ id, coreProblem: "Reliable measurement", keyMethod: "Controlled intervention", mainResult: "Improves the measured outcome", whyRelevant: "Supports the selected research topic", limitations: "Small experimental sample" });
      } else {
        throw new Error(`Unexpected model task: ${system.slice(0, 120)}`);
      }
      calls.push({ kind, messages });
      return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content }, finish_reason: "stop" }], usage: { prompt_tokens: 50, completion_tokens: 50 } })}\n\ndata: [DONE]\n\n`, {
        status: 200, headers: { "content-type": "text/event-stream" },
      });
    }
    if (url.hostname === "export.arxiv.org" && url.searchParams.has("id_list")) {
      return new Response(atom(url.searchParams.get("id_list")!.split(",")), { status: 200 });
    }
    if (url.hostname === "arxiv.org" && url.pathname === "/list/cs.AI/recent") return new Response(recent, { status: 200 });
    if (url.hostname === "arxiv.org" && url.pathname.startsWith("/html/")) return new Response(paperHtml(url.pathname.slice("/html/".length)), { status: 200 });
    throw new Error(`Unexpected network request: ${url}`);
  };
  return { fetch, calls, requests };
}

async function fixture(authorized: boolean) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-cli-personalized-workflow-"));
  roots.push(root);
  const selectedRoot = path.join(root, "library"), vaultRoot = path.join(root, "vault");
  await fs.mkdir(selectedRoot); await fs.mkdir(vaultRoot);
  const sourceBytes = new Map(Array.from({ length: 6 }, (_, index) => [
    `2610.${String(index + 1).padStart(5, "0")}.pdf`, `${index < 3 ? "agents" : "vision"} experimental methods. `.repeat(30),
  ]));
  await Promise.all([...sourceBytes].map(([name, bytes]) => fs.writeFile(path.join(selectedRoot, name), bytes)));
  const source = await openScopedLibrarySource(selectedRoot);
  const connection = authorizeLibraryConnection(createLibraryConnection(source.canonicalRoot, source.rootIdentity), { llmBaseUrl: MODEL_URL });
  const configPath = path.join(root, "config.toml");
  const document = {
    schema_version: 1, vault_root: vaultRoot, cache_dir: path.join(root, "cache"),
    llm: { api_key: "fixture-key", base_url: MODEL_URL, model: "controlled-model" },
    embedding: { mode: "local" },
    arxiv: { categories: ["cs.AI"], timezone: "UTC", topics: [{ name: "Astronomy", tag: "astronomy", description: "Gravitational-wave astronomy", detail: false }] },
    output: { summary_language: "zh" }, library: connection,
  };
  await fs.writeFile(configPath, stringify(document));
  const config = await loadCliConfig({ configPath });
  const network = transport();
  const host = buildNodeHostAdapters({ rootDir: vaultRoot, fetch: network.fetch });
  host.storage = new NodeStorageAdapter(vaultRoot, { lockRoot: path.join(root, "locks") });
  const logger = new Logger("error");
  const seedRuntime = await buildCliRuntime(config, { host, logger });
  runtimes.push(seedRuntime);
  const parser: DocumentParser = {
    capabilities: ["page-text"], provenance: { id: "controlled-parser", version: "1" },
    parse: async (bytes) => ({ mediaType: "application/pdf", blocks: [{ kind: "page", text: new TextDecoder().decode(bytes), locator: { page: 1, block: 0 } }] }),
  };
  const embedding: EmbeddingModel = {
    modelId: "controlled-embedding", dimension: 2, prefixPolicy: "none",
    embed: async (texts) => texts.map((text) => new Float32Array(text.includes("agents") ? [1, 0] : [0, 1])),
  };
  let id = 0;
  const workflow = new LibraryWorkflow({
    storage: host.storage, output: config.settings.output, connection, source, http: host.http,
    arxivFetcher: seedRuntime.fetcher, llm: seedRuntime.llm,
    createParser: async () => ({ parser }), createEmbedding: async () => embedding,
    assertCurrent: () => undefined, assertAuthorized: () => undefined, createId: () => `fixture-${++id}`,
  });
  const catalog = await workflow.scan();
  expect(await workflow.index()).toMatchObject({ indexed: 6, failed: 0 });
  const proposal = await workflow.propose();
  expect(proposal.candidates).toHaveLength(2);
  const candidate = proposal.candidates.find((entry) => entry.representatives.some(({ paperKey }) => paperKey === "arxiv:2610.00001"))!;
  const confirmed = await workflow.confirm({ candidateId: candidate.id, directionId: DIRECTION_ID, status: "active", draft: {
    name: candidate.name, description: candidate.description, discoveryCues: candidate.discoveryCues,
    representativePaperKeys: candidate.representatives.map(({ paperKey }) => paperKey),
  } });
  seedRuntime.dispose?.();
  if (!authorized) await fs.writeFile(configPath, stringify({ ...document, library: revokeLibraryConnection(connection) }));
  network.calls.length = 0;
  network.requests.length = 0;
  return { configPath, host, network, logger, sourceBytes, selectedRoot, catalog, profile: confirmed.profile };
}

describe("CLI personalized discovery through the complete daily workflow", () => {
  it.each([true, false])("writes a real daily and paper index with library authorization=%s", async (authorized) => {
    const f = await fixture(authorized);
    const stdout: string[] = [], stderr: string[] = [];
    let runtime: CliRuntime | undefined;
    const code = await runCli({
      argv: ["run", "--date", DATE], env: {},
      loadConfig: () => loadCliConfig({ configPath: f.configPath }),
      buildRuntime: async (config) => {
        runtime = await buildCliRuntime(config, { host: f.host, logger: f.logger });
        runtimes.push(runtime);
        return runtime;
      },
      io: { stdout: { write: (chunk) => { stdout.push(String(chunk)); } }, stderr: { write: (chunk) => { stderr.push(String(chunk)); } } },
    });
    expect(code, [...stderr, ...f.logger.getBuffer()].join("\n")).toBe(0);
    expect(stdout.join("\n")).toContain(`completed (${authorized ? 2 : 1} papers written)`);
    const dailyPath = runtime!.writer.dailyPath(DATE);
    const daily = await f.host.storage.readText(dailyPath);
    const inbox = await runtime!.paperIndex.load();
    expect(daily).toContain(metadata(MANUAL_ID).title);
    expect(inbox.papers[`arxiv:${MANUAL_ID}`]?.primaryTopic).toBe("astronomy");
    expect(f.network.calls.filter(({ kind }) => kind === "manual-filter")).toHaveLength(1);
    const provenance = parseDailyReportDiscoveryProvenance(daily, DATE);
    const novelty = parseDailyReportPersonalNovelty(daily, DATE);
    expect(provenance.kind).toBe("valid"); expect(novelty.kind).toBe("valid");
    if (provenance.kind !== "valid" || novelty.kind !== "valid") throw new Error("Invalid committed report markers");
    if (authorized) {
      expect(daily).toContain(metadata(LIBRARY_ID).title);
      expect(daily).toContain("> 个人新颖性：新方法");
      expect(daily).toContain("evaluation absent from the representative abstracts");
      expect(daily).toContain("证据深度：元数据与摘要");
      const entry = inbox.papers[`arxiv:${LIBRARY_ID}`]!;
      expect(entry.primaryTopic).toBe("personal-library");
      const occurrence = provenance.occurrences.find(({ arxivId }) => arxivId === LIBRARY_ID)!;
      expect(occurrence.provenance).toMatchObject({ manualTopicTags: [], directions: [{ id: DIRECTION_ID }] });
      expect(occurrence.provenance.directions[0]!.representatives.every(({ evidenceDepth }) => evidenceDepth === "metadata-and-abstract")).toBe(true);
      expect(entry.discoveryProvenanceByReport[dailyPath]).toEqual(occurrence.provenance);
      expect(novelty.occurrences).toEqual([{ arxivId: LIBRARY_ID, novelty: {
        differenceType: "new-method", comparisonBasis: ["arxiv:2610.00001"], evidenceDepth: "metadata-and-abstract", explanation: NOVELTY,
      } }]);
      expect(entry.noveltyByReport[dailyPath]).toEqual(novelty.occurrences[0]!.novelty);
      expect(f.network.calls.filter(({ kind }) => kind === "library-filter")).toHaveLength(1);
      expect(f.network.calls.filter(({ kind }) => kind === "novelty")).toHaveLength(1);
      const noveltyInput = fencedData(f.network.calls.find(({ kind }) => kind === "novelty")!.messages) as {
        paper: { paperKey: string }; basis: Array<Record<string, unknown>>;
      };
      expect(noveltyInput.paper.paperKey).toBe(`arxiv:${LIBRARY_ID}`);
      for (const basis of noveltyInput.basis) {
        expect(Object.keys(basis).sort()).toEqual(["abstract", "authors", "categories", "paperKey", "published", "title"]);
        expect(basis.abstract).toBe(f.catalog.papers[String(basis.paperKey)]!.abstract);
      }
    } else {
      expect(daily).not.toContain(metadata(LIBRARY_ID).title);
      expect(inbox.papers[`arxiv:${LIBRARY_ID}`]).toBeUndefined();
      expect(novelty.occurrences).toEqual([]);
      expect(f.network.calls.some(({ kind }) => kind === "library-filter" || kind === "novelty")).toBe(false);
    }
    expect(f.network.calls.filter(({ kind }) => kind === "summary")).toHaveLength(authorized ? 2 : 1);
    expect(runtime!.stateStore.snapshot()[DATE]?.status).toBe("completed");
    for (const [name, bytes] of f.sourceBytes) expect(await fs.readFile(path.join(f.selectedRoot, name), "utf8")).toBe(bytes);
  }, 45_000);
});
