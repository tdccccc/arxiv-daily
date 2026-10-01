import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { stringify as stringifyToml } from "smol-toml";
import {
  authorizeLibraryConnection,
  createLibraryConnection,
  createPersonalLibraryScopeFingerprint,
  createPersonalLibraryIdentificationFingerprint,
  decodePersonalLibraryCatalog,
  decodePersonalLibraryDirectionProposal,
  decodePersonalLibraryInterestProfile,
  decodeFullTextKnowledgeBaseManifest,
  PersonalLibraryCatalogStore,
  PersonalLibraryDirectionProposalStore,
  PersonalLibraryInterestProfileStore,
  FullTextKnowledgeBaseFileStore,
  type DocumentParser,
  type EmbeddingModel,
  type HttpClient,
} from "@arxiv-daily/core";
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
      const parsed = JSON.parse(match[1]!) as { paperKey: string }[] | { clusters: { paperKeys: string[] }[] };
      const answer = Array.isArray(parsed)
        ? { candidates: [{ name: "Research direction", description: "A direction supported by this cluster.", discoveryCues: ["methods", "evaluation"], representativePaperKeys: parsed.slice(0, 3).map(paper => paper.paperKey) }] }
        : { suggestions: parsed.clusters.map(cluster => ({ kind: "new", paperKeys: cluster.paperKeys, reason: "New coherent research theme." })) };
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
      profile: new PersonalLibraryInterestProfileStore(storage, config.settings.output, scope, identification),
      knowledgeBase: new FullTextKnowledgeBaseFileStore(storage, config.settings.output, scope, identification),
    };
  }
  async function assertSourcesUnchanged() {
    for (const [file, bytes] of sources) expect(new Uint8Array(await readFile(join(sourceRoot, file)))).toEqual(bytes);
  }
  return { root, sourceRoot, vaultRoot, sources, config, configPath, connection, parser, embedding, http, options, modelRequests, invoke, stores, assertSourcesUnchanged };
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
    expect(proposal!.candidates).toHaveLength(2);
    expect(decodePersonalLibraryDirectionProposal(JSON.parse(await stores.storage.readText(stores.proposal.paths.documentPath))))
      .toEqual(proposal);
    const beforeConfirm = await f.invoke(["directions"]);
    expect(beforeConfirm.code, beforeConfirm.stderr).toBe(0);
    const review = JSON.parse(beforeConfirm.stdout);
    expect(review.profile.directions).toEqual([]);
    expect(review.proposal).toEqual(proposal);
    const candidate = proposal!.candidates[0]!;
    const confirmArgs = ["confirm", "--candidate", candidate.id, "--proposal-revision", String(proposal!.revision), "--profile-revision", String(review.profile.revision)];
    const confirmed = await f.invoke(confirmArgs);
    expect(confirmed.code, confirmed.stderr).toBe(0);
    const confirmedReview = JSON.parse(confirmed.stdout);
    const profile = decodePersonalLibraryInterestProfile(confirmedReview.profile);
    expect(profile).not.toBeNull();
    expect(profile!.directions).toHaveLength(1);
    expect(profile!.directions[0]).toMatchObject({ name: candidate.name, status: "active" });
    expect(decodePersonalLibraryInterestProfile(JSON.parse(await stores.storage.readText(stores.profile.paths.documentPath))))
      .toEqual(profile);
    // Fresh stores and config loads recover the same authoritative records.
    expect(await f.stores().profile.load()).toEqual(profile);
    const reloaded = await f.invoke(["directions"]);
    expect(reloaded.code, reloaded.stderr).toBe(0);
    expect(JSON.parse(reloaded.stdout).profile).toEqual(profile);
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
    expect(lexicalMatches[0].hits[0]).toMatchObject({ source: "lexical", page: 1 });
    expect(lexicalMatches[0].hits[0].text).toContain("agents");
    // Core checks model identity even for lexical mode; the lazy model must not infer.
    expect(f.embedding.embed).toHaveBeenCalledTimes(embeddingCallsBeforeSearch);
    const hybrid = await f.invoke(["search", "--query", "agents", "--mode", "hybrid", "--limit", "2"]);
    expect(hybrid.code, hybrid.stderr).toBe(0);
    const hybridMatches = JSON.parse(hybrid.stdout).matches;
    expect(hybridMatches).toHaveLength(2);
    expect(hybridMatches[0].rankingScoreKind).toBe("rrf");
    expect(hybridMatches[0].hits.some((hit: { text: string }) => hit.text.includes("agents"))).toBe(true);
    // Confirming removes the accepted candidate. Use the still-pending second
    // candidate with old revisions to exercise the revision gate itself.
    const staleConfirmation = await f.invoke([
      "confirm", "--candidate", proposal!.candidates[1]!.id,
      "--proposal-revision", String(proposal!.revision),
      "--profile-revision", String(review.profile.revision),
    ]);
    expect(staleConfirmation.code).not.toBe(0);
    expect(staleConfirmation.stderr).toMatch(/revision|stale|review/i);
    expect(await f.stores().profile.load()).toEqual(profile);
    await f.assertSourcesUnchanged();
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
    expect(review.suggestions.pendingAuthorization.bufferedPaperCount).toBe(6);
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
});
