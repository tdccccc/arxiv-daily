import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, realpath, rename, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parse as parseToml, stringify as stringifyToml } from "smol-toml";
import {
  authorizeLibraryConnection,
  createLibraryConnection,
  createPersonalLibraryScopeFingerprint,
  createPersonalLibraryIdentificationFingerprint,
  decodePersonalLibraryCatalog,
  decodePersonalLibraryDirectionProposal,
  decodeFullTextKnowledgeBaseManifest,
  PersonalLibraryCatalogStore,
  PersonalLibraryDirectionProposalStore,
  FullTextKnowledgeBaseFileStore,
  type DocumentParser,
  type EmbeddingModel,
  type HttpClient,
} from "@arxiv-daily/core";
import * as nodeRuntime from "@arxiv-daily/node-runtime";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../src/config";
import { runCliLibrary } from "../src/library-cmd";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
const endpoint = "https://library-fixture.invalid/v1";

// Valid source PDFs exercise the real filesystem and source identity boundary.
// Parsing and embeddings are controlled ports here; this is not an inference test.
function pdfBytes(id: string, theme: string): Uint8Array {
  const content = `BT /F1 12 Tf 72 720 Td (${theme} research evidence for ${id}.) Tj ET`;
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    `<< /Length ${Buffer.byteLength(content)} >>\nstream\n${content}\nendstream`,
    `<< /arXivID (${id}) >>`,
  ];
  let text = "%PDF-1.4\n";
  const offsets: number[] = [];
  for (const [index, object] of objects.entries()) {
    offsets.push(Buffer.byteLength(text));
    text += `${index + 1} 0 obj\n${object}\nendobj\n`;
  }
  const xref = Buffer.byteLength(text);
  text += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`
    + offsets.map(offset => `${String(offset).padStart(10, "0")} 00000 n \n`).join("")
    + `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R /Info 6 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return new TextEncoder().encode(text);
}

async function fixture({ authorized = true, remote = false } = {}) {
  const root = await mkdtemp(join(tmpdir(), "arxiv-cli-library-"));
  roots.push(root);
  const sourceRoot = join(root, "source papers");
  const vaultRoot = join(root, "research records");
  await mkdir(sourceRoot);
  await mkdir(vaultRoot);
  const canonical = await realpath(sourceRoot);
  const info = await stat(canonical);
  let connection = createLibraryConnection(canonical, `${info.dev}:${info.ino}`);
  if (authorized) connection = authorizeLibraryConnection(connection, {
    llmBaseUrl: endpoint,
    ...(remote ? { embeddingEndpoint: { baseUrl: endpoint } } : {}),
  });
  const sources = new Map<string, Uint8Array>();
  for (let index = 1; index <= 6; index++) {
    const id = `2610.${String(index).padStart(5, "0")}`;
    const bytes = pdfBytes(id, index <= 3 ? "agents" : "vision");
    sources.set(`${id}.pdf`, bytes);
    await writeFile(join(sourceRoot, `${id}.pdf`), bytes);
  }
  const configPath = join(root, "config.toml");
  await writeFile(configPath, stringifyToml({
    schema_version: 1, vault_root: vaultRoot, cache_dir: join(root, "cache"),
    llm: { provider: "openai", base_url: endpoint, api_key: "fixture-secret", model: "fixture-model", thinking_mode: false },
    arxiv: { categories: ["cs.AI"], timezone: "UTC", topics: [{ name: "Agent research", tag: "agents", description: "Agent evaluation", detail: false }] },
    output: { daily_dir: "arxiv-daily/daily", papers_dir: "arxiv-daily/papers", summary_language: "en", link_style: "relative" },
    embedding: { mode: remote ? "remote" : "local", base_url: endpoint, api_key: "embedding-fixture-secret", model: "fixture", dimension: 2 },
    advanced: { log_level: "error" }, library: connection,
  }));
  const config = await loadCliConfig({ configPath });
  expect(config.libraryConnection).toEqual(connection);
  const parser: DocumentParser = {
    capabilities: ["page-text"], provenance: { id: "controlled-test-parser", version: "1" },
    parse: vi.fn(async bytes => {
      expect(new TextDecoder().decode(bytes)).toMatch(/^%PDF-1\.4/);
      const body = new TextDecoder().decode(bytes).includes("agents") ? "agents" : "vision";
      return { mediaType: "application/pdf", blocks: [{ kind: "page", text: `${body} meaningful experimental research evidence. `.repeat(40), locator: { page: 1, block: 0 } }] };
    }),
  };
  const embedding: EmbeddingModel = {
    modelId: "controlled-test-vectors", dimension: 2, prefixPolicy: "none",
    embed: vi.fn(async texts => texts.map(text => new Float32Array(text.includes("agents") ? [1, 0] : [0, 1]))),
  };
  const modelRequests: string[] = [];
  const http: HttpClient = { request: vi.fn(async request => {
    const url = new URL(request.url);
    if (url.origin === "https://export.arxiv.org" && url.pathname === "/api/query" && url.searchParams.has("id_list")) {
      const ids = url.searchParams.get("id_list")!.split(",");
      const entries = ids.map(id => `<entry><id>http://arxiv.org/abs/${id}v1</id><title>Paper ${id}</title><author><name>Fixture Researcher</name></author><summary>Evidence for ${Number(id.slice(-5)) <= 3 ? "agents" : "vision"} research.</summary><published>2026-10-01T00:00:00Z</published><updated>2026-10-01T00:00:00Z</updated><arxiv:primary_category term="cs.AI"/><category term="cs.AI"/></entry>`).join("");
      return { status: 200, headers: {}, bodyText: `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${entries}</feed>` };
    }
    if (url.href === `${endpoint}/chat/completions`) {
      const body = JSON.parse(String(request.body)) as { messages: { role: string; content: string }[] };
      const user = body.messages.find(message => message.role === "user")!.content;
      modelRequests.push(user);
      const match = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(user);
      if (!match) throw new Error("Fixture model expected paper_data");
      const parsed = JSON.parse(match[1]!) as { groups: { id: string; papers: {paperKey:string}[] }[] };
      const answer = { topics: parsed.groups.map((group,index) => ({suggestedName:`Research theme ${modelRequests.length}-${index}`,directions:[{text:`Research direction ${index}`,discoveryCues:["methods","evaluation"],groupIds:[group.id],representativePaperKeys:group.papers.slice(0,3).map(paper=>paper.paperKey)}]})) };
      return { status: 200, headers: {}, bodyText: `data: ${JSON.stringify({ choices: [{ delta: { content: JSON.stringify(answer) } }] })}\n\ndata: [DONE]\n\n` };
    }
    throw new Error(`Unexpected fixture HTTP request: ${request.url}`);
  }) };
  const options = { http, createParser: vi.fn(async () => ({ parser })), createEmbedding: vi.fn(async () => embedding) };
  async function invoke(args: string[]) {
    const out: string[] = [], err: string[] = [];
    const currentConfig = await loadCliConfig({ configPath });
    const code = await runCliLibrary(currentConfig, args, { stdout: { write: text => out.push(text) }, stderr: { write: text => err.push(text) } }, options);
    const stdout = out.join(""), stderr = err.join("");
    expect(stdout + stderr).not.toMatch(/fixture-secret|embedding-fixture-secret/);
    return { code, stdout, stderr };
  }
  function stores() {
    const storage = new NodeStorageAdapter(vaultRoot);
    const scope = createPersonalLibraryScopeFingerprint(connection);
    const identification = createPersonalLibraryIdentificationFingerprint(connection.eligibleExtensions);
    return {
      storage, scope, identification,
      catalog: new PersonalLibraryCatalogStore(storage, config.settings.output),
      proposal: new PersonalLibraryDirectionProposalStore(storage, config.settings.output, scope, identification),
      knowledgeBase: new FullTextKnowledgeBaseFileStore(storage, config.settings.output, scope, identification),
    };
  }
  async function addPapers(serials: number[]) {
    for (const serial of serials) {
      const id = `2610.${String(serial).padStart(5, "0")}`;
      await writeFile(join(sourceRoot, `${id}.pdf`), pdfBytes(id, "agents"));
    }
    return serials.map(serial => `arxiv:2610.${String(serial).padStart(5, "0")}`);
  }
  async function assertSourcesUnchanged() {
    for (const [file, bytes] of sources) expect(new Uint8Array(await readFile(join(sourceRoot, file)))).toEqual(bytes);
  }
  return { root, sourceRoot, vaultRoot, sources, config, configPath, connection, parser, embedding, http, options, modelRequests, invoke, stores, addPapers, assertSourcesUnchanged };
}

describe("CLI shared library workflow", () => {
  it("scans, indexes, proposes and confirms through existing product stores across command invocations", async () => {
    const f = await fixture();
    const result = await f.invoke(["scan"]);
    expect(result.code, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({ paperCount: 6, lastScan: { papers: 6 } });
    const stores = f.stores();
    const catalogRaw = JSON.parse(await stores.storage.readText(stores.catalog.paths.documentPath));
    const catalog = decodePersonalLibraryCatalog(catalogRaw);
    expect(catalog).not.toBeNull();
    expect(Object.keys(catalog!.papers)).toHaveLength(6);
    expect(await stores.catalog.load(stores.scope, stores.identification)).toEqual(catalog);
    const indexed = await f.invoke(["index"]);
    expect(indexed.code, indexed.stderr).toBe(0);
    expect(JSON.parse(indexed.stdout)).toMatchObject({ indexed: 6, failed: 0, searchablePapers: 6 });
    expect(f.parser.parse).toHaveBeenCalledTimes(6);
    expect(f.embedding.embed).toHaveBeenCalled();
    const manifestRaw = JSON.parse(await stores.storage.readText(stores.knowledgeBase.paths.manifest.documentPath));
    const manifest = decodeFullTextKnowledgeBaseManifest(manifestRaw);
    expect(manifest).not.toBeNull();
    expect(Object.keys(manifest!.papers)).toHaveLength(6);
    expect(await stores.knowledgeBase.loadManifest()).toEqual(manifest);

    const proposed = await f.invoke(["propose"]);
    expect(proposed.code, proposed.stderr).toBe(0);
    const proposal = decodePersonalLibraryDirectionProposal(JSON.parse(proposed.stdout));
    expect(proposal).not.toBeNull();
    expect(proposal!.topics.flatMap(topic => topic.directions)).toHaveLength(2);
    expect(decodePersonalLibraryDirectionProposal(JSON.parse(await stores.storage.readText(stores.proposal.paths.documentPath))))
      .toEqual(proposal);
    const beforeConfirm = await f.invoke(["directions"]);
    expect(beforeConfirm.code, beforeConfirm.stderr).toBe(0);
    const review = JSON.parse(beforeConfirm.stdout);
    expect(review.acceptances).toEqual([]);
    expect(review.proposal).toEqual(proposal);
    const candidate = proposal!.topics.flatMap(topic => topic.directions)[0]!;
    const confirmArgs = ["confirm", "--candidate", candidate.id, "--proposal-revision", String(proposal!.revision)];
    const confirmed = await f.invoke(confirmArgs);
    expect(confirmed.code, confirmed.stderr).toBe(0);
    const confirmedReview = JSON.parse(confirmed.stdout);
    expect(confirmedReview.topics.flatMap((topic: {directions:{text:string}[]})=>topic.directions).some((direction:{text:string})=>direction.text===candidate.text)).toBe(true);
    expect(confirmedReview.acceptances).toHaveLength(1);
    const reloaded = await f.invoke(["directions"]);
    expect(reloaded.code, reloaded.stderr).toBe(0);
    expect(JSON.parse(reloaded.stdout).topics).toEqual(confirmedReview.topics);
    const acceptFile = join(f.root, "accept.json");
    await writeFile(acceptFile, JSON.stringify({ operation: "accept-topics", topicIds: [proposal!.topics[0]!.id], candidateIds: [candidate.id], expectedProposalRevision: confirmedReview.proposal.revision }));
    const repeatedAcceptance = await f.invoke(["review", "--input", acceptFile]);
    expect(repeatedAcceptance.code, repeatedAcceptance.stderr).toBe(0);
    expect(JSON.parse(repeatedAcceptance.stdout).topics).toEqual(confirmedReview.topics);
    expect(JSON.parse(repeatedAcceptance.stdout).acceptances).toEqual(confirmedReview.acceptances);
    const repeatedIndex = await f.invoke(["index"]);
    expect(repeatedIndex.code, repeatedIndex.stderr).toBe(0);
    expect(JSON.parse(repeatedIndex.stdout)).toMatchObject({ indexed: 0, reused: 6, failed: 0 });
    expect(f.parser.parse).toHaveBeenCalledTimes(6);
    const embeddingCallsBeforeSearch = vi.mocked(f.embedding.embed).mock.calls.length;
    const lexical = await f.invoke(["search", "--query", "agents", "--mode", "lexical", "--limit", "2"]);
    expect(lexical.code, lexical.stderr).toBe(0);
    const lexicalMatches = JSON.parse(lexical.stdout).matches;
    expect(lexicalMatches).toHaveLength(2);
    expect(lexicalMatches.every((match: { paperKey: string; rankingScoreKind: string }) =>
      ["arxiv:2610.00001", "arxiv:2610.00002", "arxiv:2610.00003"].includes(match.paperKey)
      && match.rankingScoreKind === "bm25")).toBe(true);
    expect(lexicalMatches[0].hits[0]).toMatchObject({ source: "lexical" });
    expect(lexicalMatches[0].hits[0].text).toContain("agents");
    // Core checks model identity even for lexical mode; the lazy model must not infer.
    expect(f.embedding.embed).toHaveBeenCalledTimes(embeddingCallsBeforeSearch);
    const hybrid = await f.invoke(["search", "--query", "agents", "--mode", "hybrid", "--limit", "2"]);
    expect(hybrid.code, hybrid.stderr).toBe(0);
    const hybridMatches = JSON.parse(hybrid.stdout).matches;
    expect(hybridMatches).toHaveLength(2);
    expect(hybridMatches[0].rankingScoreKind).toBe("rrf");
    expect(hybridMatches[0].hits.some((hit: { text: string }) => hit.text.includes("agents"))).toBe(true);
    // A mismatched proposal revision cannot append a second direction.
    const staleConfirmation = await f.invoke([
      "confirm", "--candidate", proposal!.topics.flatMap(topic => topic.directions)[1]!.id,
      "--proposal-revision", String(proposal!.revision + 1),
    ]);
    expect(staleConfirmation.code).not.toBe(0);
    expect(staleConfirmation.stderr).toMatch(/revision|stale|review/i);
    expect((await loadCliConfig({configPath:f.configPath})).settings.arxiv.topics).toEqual(confirmedReview.topics);
    await f.assertSourcesUnchanged();
  });

  it("explains retired profile operations without modifying topic settings", async () => {
    const f = await fixture();
    const before = await readFile(f.configPath, "utf8");
    const update = await f.invoke(["update"]);
    expect(update.code).toBe(2);
    expect(update.stderr).toMatch(/retired.*topic settings/);
    const file = join(f.root, "old-review.json");
    await writeFile(file, JSON.stringify({ operation: "lock", directionId: "old-profile" }));
    const review = await f.invoke(["review", "--input", file]);
    expect(review.code).toBe(2);
    expect(review.stderr).toMatch(/profiles are retired/);
    expect(await readFile(f.configPath, "utf8")).toBe(before);
  });

  it("indexes titles and abstracts without probing an enabled document sidecar", async () => {
    const f = await fixture();
    const document = parseToml(await readFile(f.configPath, "utf8"));
    await writeFile(f.configPath, stringifyToml({ ...document, pdf_parser_sidecar: {
      enabled: true, capabilities_url: "http://127.0.0.1:5001/v1/capabilities", parse_url: "http://127.0.0.1:5001/v1/parse",
    } }));
    const config = await loadCliConfig({ configPath: f.configPath });
    expect(config.settings.pdfParserSidecar.enabled).toBe(true);
    const scanned = await f.invoke(["scan"]);
    expect(scanned.code, scanned.stderr).toBe(0);
    const sidecarRequests: string[] = [];
    const http: HttpClient = { request: async request => {
      if (new URL(request.url).hostname === "127.0.0.1") {
        sidecarRequests.push(request.url);
        return { status: 200, headers: {}, bodyText: JSON.stringify({ protocolVersion: 1,
          parser: { id: "fixture-sidecar", version: "1" }, capabilities: ["page-text", "document-structure"],
          maxRequestBytes: 5_000_000, maxResponseBytes: 2_000_000,
        }) };
      }
      return f.http.request(request);
    } };
    const createParser = vi.spyOn(nodeRuntime, "createNodeLibraryDocumentParser").mockResolvedValue(f.parser);
    try {
      const out: string[] = [], err: string[] = [];
      const code = await runCliLibrary(config, ["index"], {
        stdout: { write: text => out.push(text) }, stderr: { write: text => err.push(text) },
      }, { http, createEmbedding: f.options.createEmbedding });
      expect(code, err.join("")).toBe(0);
      expect(JSON.parse(out.join(""))).toMatchObject({ indexed: 6, failed: 0, searchablePapers: 6 });
      expect(sidecarRequests).toEqual([]);
      await f.assertSourcesUnchanged();
    } finally { createParser.mockRestore(); }
  });

  it("allows local indexing without a model-processing grant but refuses direction inference", async () => {
    const f = await fixture({ authorized: false });
    expect((await f.invoke(["scan"])).code).toBe(0);
    const indexed = await f.invoke(["index"]);
    expect(indexed.code, indexed.stderr).toBe(0);
    expect(JSON.parse(indexed.stdout)).toMatchObject({ indexed: 6, failed: 0 });
    expect(f.modelRequests).toEqual([]);
    const proposed = await f.invoke(["propose"]);
    expect(proposed.code).not.toBe(0);
    expect(proposed.stderr).toMatch(/authoriz/i);
    expect(f.modelRequests).toEqual([]);
    expect(await f.stores().proposal.load()).toBeNull();
    const review = JSON.parse((await f.invoke(["directions"])).stdout);
    expect(review.proposal).toBeNull();
    expect(review.acceptances).toEqual([]);
    await f.assertSourcesUnchanged();
  });

  it("refuses unauthorized remote embedding before creating parsers or embedding clients", async () => {
    const f = await fixture({ authorized: false, remote: true });
    expect((await f.invoke(["scan"])).code).toBe(0);
    const result = await f.invoke(["index"]);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toMatch(/authoriz/i);
    expect(f.options.createParser).not.toHaveBeenCalled();
    expect(f.options.createEmbedding).not.toHaveBeenCalled();
    expect(f.modelRequests).toEqual([]);
    const manifest = await f.stores().knowledgeBase.loadManifest();
    expect(manifest.revision).toBe(0);
    expect(manifest.papers).toEqual({});
    await f.assertSourcesUnchanged();
  });
  it("reports papers added since the direction suggestions and nudges from three", async () => {
    const f = await fixture();
    await f.invoke(["scan"]);
    const beforeProposal = await f.invoke(["index"]);
    expect(JSON.parse(beforeProposal.stdout).newPapersSinceSuggestions).toBeNull();
    expect(beforeProposal.stderr).not.toMatch(/propose --new/);
    const proposal = decodePersonalLibraryDirectionProposal(JSON.parse((await f.invoke(["propose"])).stdout))!;
    await f.invoke(["confirm", "--candidate", proposal.topics[0]!.directions[0]!.id, "--proposal-revision", String(proposal.revision)]);
    const unchanged = await f.invoke(["index"]);
    expect(JSON.parse(unchanged.stdout).newPapersSinceSuggestions).toBe(0);
    expect(unchanged.stderr).not.toMatch(/propose --new/);

    await f.addPapers([7, 8]);
    await f.invoke(["scan"]);
    const two = await f.invoke(["index"]);
    expect(two.code, two.stderr).toBe(0);
    expect(JSON.parse(two.stdout).newPapersSinceSuggestions).toBe(2);
    expect(two.stderr).not.toMatch(/propose --new/);

    await f.addPapers([9]);
    await f.invoke(["scan"]);
    const three = await f.invoke(["index"]);
    expect(JSON.parse(three.stdout)).toMatchObject({ newPapersSinceSuggestions: 3, failed: 0 });
    expect(three.stderr).toMatch(/3 .*papers.*not covered.*library propose --new/s);
    expect((three.stderr.match(/propose --new/g) ?? [])).toHaveLength(1);

    const directions = await f.invoke(["directions"]);
    expect(JSON.parse(directions.stdout).newPapersSinceSuggestions).toBe(3);
    const status = await f.invoke(["status"]);
    expect(JSON.parse(status.stdout)).not.toHaveProperty("newPapersSinceSuggestions");
  }, 30_000);

  it("index alone notices PDFs added to the folder, asking arXiv only about them", async () => {
    const f = await fixture();
    await f.invoke(["scan"]); await f.invoke(["index"]);
    const first = decodePersonalLibraryDirectionProposal(JSON.parse((await f.invoke(["propose"])).stdout))!;
    await f.invoke(["confirm", "--candidate", first.topics[0]!.directions[0]!.id, "--proposal-revision", String(first.revision)]);
    const arxivLookups = () => vi.mocked(f.http.request).mock.calls
      .map(([request]) => new URL(request.url))
      .filter(url => url.origin === "https://export.arxiv.org" && url.searchParams.has("id_list"))
      .map(url => url.searchParams.get("id_list"));

    const unchanged = await f.invoke(["index"]);
    expect(unchanged.code, unchanged.stderr).toBe(0);
    expect(unchanged.stderr).not.toMatch(/folder changed/i);
    expect(JSON.parse(unchanged.stdout)).toMatchObject({ indexed: 0, reused: 6 });
    const lookupsBefore = arxivLookups().length;

    await f.addPapers([7, 8, 9]);
    const indexed = await f.invoke(["index"]);
    expect(indexed.code, indexed.stderr).toBe(0);
    expect(indexed.stderr).toMatch(/folder changed.*scanning/i);
    expect(JSON.parse(indexed.stdout)).toMatchObject({ indexed: 3, reused: 6, failed: 0, newPapersSinceSuggestions: 3 });
    expect(indexed.stderr).toMatch(/3 .*papers.*not covered.*library propose --new/s);
    expect(arxivLookups().slice(lookupsBefore)).toEqual(["2610.00007,2610.00008,2610.00009"]);
    expect(Object.keys((await f.stores().catalog.load(f.stores().scope, f.stores().identification)).papers)).toHaveLength(9);

    const again = await f.invoke(["index"]);
    expect(again.stderr).not.toMatch(/folder changed/i);
    expect(JSON.parse(again.stdout)).toMatchObject({ indexed: 0, reused: 9 });
    expect(arxivLookups().slice(lookupsBefore)).toHaveLength(1);

    await rm(join(f.sourceRoot, "2610.00009.pdf"));
    await rename(join(f.sourceRoot, "2610.00001.pdf"), join(f.sourceRoot, "moved.2610.00001.pdf"));
    const pruned = await f.invoke(["index"]);
    expect(pruned.code, pruned.stderr).toBe(0);
    expect(pruned.stderr).toMatch(/folder changed.*scanning/i);
    expect(arxivLookups().slice(lookupsBefore)).toHaveLength(1);
    const catalog = await f.stores().catalog.load(f.stores().scope, f.stores().identification);
    expect(Object.keys(catalog.papers)).toHaveLength(8);
    expect(catalog.files["moved.2610.00001.pdf"]?.status).toBe("ready");
    expect(catalog.files["2610.00001.pdf"]).toBeUndefined();
  }, 60_000);

  it("proposes --new sends only the new papers, then reports nothing new", async () => {
    const f = await fixture();
    await f.invoke(["scan"]); await f.invoke(["index"]);
    const first = decodePersonalLibraryDirectionProposal(JSON.parse((await f.invoke(["propose"])).stdout))!;
    await f.invoke(["confirm", "--candidate", first.topics[0]!.directions[0]!.id, "--proposal-revision", String(first.revision)]);
    const added = await f.addPapers([7, 8, 9]);
    await f.invoke(["scan"]); await f.invoke(["index"]);
    const requestsBefore = f.modelRequests.length;
    const updated = await f.invoke(["propose", "--new"]);
    expect(updated.code, updated.stderr).toBe(0);
    expect(f.modelRequests).toHaveLength(requestsBefore + 1);
    const sent = f.modelRequests[requestsBefore]!;
    for (const key of added) expect(sent).toContain(key);
    for (let serial = 1; serial <= 6; serial++) expect(sent).not.toContain(`arxiv:2610.${String(serial).padStart(5, "0")}`);
    const proposal = decodePersonalLibraryDirectionProposal(JSON.parse(updated.stdout))!;
    expect(proposal.catalogInputPapers.map(paper => paper.paperKey)).toEqual(added);
    expect(JSON.parse((await f.invoke(["directions"])).stdout).newPapersSinceSuggestions).toBe(0);

    const again = await f.invoke(["propose", "--new"]);
    expect(again.code, again.stderr).toBe(0);
    expect(f.modelRequests).toHaveLength(requestsBefore + 1);
    expect(again.stderr).toMatch(/no new papers/i);
    expect(JSON.parse(again.stdout)).toMatchObject({ updated: false, newPapersSinceSuggestions: 0 });
  }, 30_000);

  it("propose --new scans the whole library when there is nothing to compare with, and rejects other arguments", async () => {
    const f = await fixture();
    await f.invoke(["scan"]); await f.invoke(["index"]);
    const result = await f.invoke(["propose", "--new"]);
    expect(result.code, result.stderr).toBe(0);
    expect(result.stderr).toMatch(/whole library/i);
    expect(decodePersonalLibraryDirectionProposal(JSON.parse(result.stdout))!.catalogInputPapers).toHaveLength(6);
    const invalid = await f.invoke(["propose", "--old"]);
    expect(invalid.code).toBe(2);
    const bare = await f.invoke(["propose", "extra"]);
    expect(bare.code).toBe(2);
  });
});
