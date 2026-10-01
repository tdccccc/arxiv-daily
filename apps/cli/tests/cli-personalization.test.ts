import { afterEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { parse, stringify } from "smol-toml";
import {
  ArxivPipeline,
  LibraryWorkflow,
  Logger,
  PersonalLibraryInterestProfileStore,
  authorizeLibraryConnection,
  createLibraryConnection,
  decodePersonalLibraryCatalog,
  decodePersonalLibraryInterestProfile,
  revokeLibraryConnection,
  type DocumentParser,
  type EmbeddingModel,
  type PipelineDeps,
} from "@arxiv-daily/core";
import { buildNodeHostAdapters, NodeStorageAdapter, openScopedLibrarySource, type FetchLike } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../src/config";
import { revokeCliLibrary } from "../src/library-connection-cmd";
import { buildCliRuntime, type CliRuntime } from "../src/runtime";

const roots: string[] = [];
const runtimes: CliRuntime[] = [];
afterEach(async () => {
  try {
    for (const runtime of runtimes.splice(0)) runtime.dispose?.();
  } finally {
    await Promise.all(roots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })));
  }
});

function trackRuntime(runtime: CliRuntime): CliRuntime {
  runtimes.push(runtime);
  return runtime;
}

async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-cli-personalization-"));
  roots.push(root);
  const selectedRoot = path.join(root, "library");
  const vaultRoot = path.join(root, "vault");
  await fs.mkdir(selectedRoot);
  await fs.mkdir(vaultRoot);
  await Promise.all(Array.from({ length: 6 }, (_, index) => fs.writeFile(
    path.join(selectedRoot, `2610.${String(index + 1).padStart(5, "0")}.pdf`),
    `${index < 3 ? "agents" : "vision"} methods and experimental evidence. `.repeat(30),
  )));
  const source = await openScopedLibrarySource(selectedRoot);
  const baseUrl = "https://model.example/v1";
  const connection = authorizeLibraryConnection(createLibraryConnection(source.canonicalRoot, source.rootIdentity), {
    llmBaseUrl: baseUrl,
  }, new Date("2026-10-01T00:00:00Z"));
  const configPath = path.join(root, "config.toml");
  await fs.writeFile(configPath, stringify({
    schema_version: 1, vault_root: vaultRoot, cache_dir: path.join(root, "cache"),
    llm: { api_key: "private-model-key", base_url: baseUrl, model: "test" },
    embedding: { mode: "local" },
    arxiv: { categories: ["cs.AI"], topics: [{ name: "Manual topic", tag: "manual", description: "Manual discovery remains available", detail: false }] },
    library: connection,
  }));
  const config = await loadCliConfig({ configPath });
  const fetch = vi.fn<FetchLike>(async () => { throw new Error("Unexpected network request"); });
  const host = buildNodeHostAdapters({ rootDir: vaultRoot, fetch });
  host.storage = new NodeStorageAdapter(vaultRoot, { lockRoot: path.join(root, "locks") });
  const parser: DocumentParser = {
    capabilities: ["page-text"], provenance: { id: "fixture", version: "1" },
    parse: async (bytes) => ({ mediaType: "application/pdf", blocks: [{
      kind: "page", text: new TextDecoder().decode(bytes), locator: { page: 1, block: 0 },
    }] }),
  };
  const embedding: EmbeddingModel = {
    modelId: "fixture", dimension: 2, prefixPolicy: "none",
    embed: async (texts) => texts.map((text) => new Float32Array(text.includes("agents") ? [1, 0] : [0, 1])),
  };
  let id = 0;
  const workflow = new LibraryWorkflow({
    storage: host.storage, output: config.settings.output, connection, source, http: host.http,
    arxivFetcher: { fetchMetadataByIds: async (ids) => new Map(ids.map((key) => [key, {
      id: key, title: `Library paper ${key}`, authors: "Researcher", authorNames: ["Researcher"],
      abstract: `Metadata evidence for ${key}`, published: "2026-10-01T00:00:00Z", updated: "2026-10-01T00:00:00Z",
      primaryCategory: "cs.AI", categories: ["cs.AI"],
    }])) },
    createParser: async () => ({ parser }), createEmbedding: async () => embedding,
    llm: { call: async (messages) => {
      const raw = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(messages.find((message) => message.role === "user")!.content)![1]!;
      const data = JSON.parse(raw) as { paperKey: string }[] | { clusters: { paperKeys: string[] }[] };
      if (!Array.isArray(data)) return JSON.stringify({ suggestions: [] });
      return JSON.stringify({ candidates: [{
        name: "Confirmed library theme", description: "A direction supported by the selected literature.",
        discoveryCues: ["methods", "evaluation"], representativePaperKeys: data.slice(0, 3).map((paper) => paper.paperKey),
      }] });
    } },
    assertCurrent: () => undefined, assertAuthorized: () => undefined,
    createId: () => `fixture-${++id}`, now: () => new Date("2026-10-01T12:00:00.000Z"),
  });
  const catalog = await workflow.scan();
  expect(decodePersonalLibraryCatalog(catalog)).not.toBeNull();
  expect(await workflow.index()).toMatchObject({ indexed: 6, failed: 0 });
  const proposal = await workflow.propose();
  expect(proposal.candidates).toHaveLength(2);
  const candidate = proposal.candidates[0]!;
  const confirmed = await workflow.confirm({ candidateId: candidate.id, directionId: "confirmed-direction", status: "active", draft: {
    name: candidate.name, description: candidate.description, discoveryCues: candidate.discoveryCues,
    representativePaperKeys: candidate.representatives.map((paper) => paper.paperKey),
  } });
  expect(decodePersonalLibraryInterestProfile(confirmed.profile)).not.toBeNull();
  const profileStore = new PersonalLibraryInterestProfileStore(
    host.storage, config.settings.output, catalog.scopeFingerprint, catalog.identificationFingerprint,
  );
  expect(await profileStore.load()).toEqual(confirmed.profile);
  return { root, selectedRoot, config, host, fetch, catalog, profile: confirmed.profile, profileStore };
}

function pipelineDeps(pipeline: ArxivPipeline): PipelineDeps {
  // The existing runtime tests inspect this composition boundary as well;
  // keep the actual pipeline and every shared store in the integration.
  return (pipeline as unknown as { deps: PipelineDeps }).deps;
}

describe("CLI personalized daily composition", () => {
  it("feeds confirmed directions and both novelty inputs from the existing stores into the real pipeline", async () => {
    const { config, host, fetch, profile, catalog } = await fixture();
    const runtime = trackRuntime(await buildCliRuntime(config, { host, logger: new Logger("debug") }));
    expect(runtime.pipeline).toBeInstanceOf(ArxivPipeline);
    const deps = pipelineDeps(runtime.pipeline);
    const direction = profile.directions[0]!;
    expect(deps.personalizedDiscovery?.directions).toEqual([{
      id: direction.id, name: direction.name, description: direction.description, discoveryCues: direction.discoveryCues,
      representatives: direction.representatives.map(({ paperKey }) => ({
        paperKey, title: catalog.papers[paperKey]!.title, evidenceDepth: "metadata-and-abstract",
      })),
    }]);
    const paperKeys = direction.representatives.map(({ paperKey }) => paperKey).sort();
    expect(deps.personalizedNoveltyRepresentatives?.representatives.map(({ paperKey }) => paperKey)).toEqual(paperKeys);
    expect(deps.personalizedNoveltyRepresentatives?.representatives[0]?.abstract).toBe(catalog.papers[paperKeys[0]!]!.abstract);
    expect(deps.personalizedNoveltyMatches).toEqual({ paperMatches: [], directionRepresentatives: [{
      directionId: direction.id, representativePaperKeys: paperKeys,
    }] });
    expect(Object.isFrozen(deps.personalizedDiscovery)).toBe(true);
    expect(Object.isFrozen(deps.personalizedNoveltyRepresentatives)).toBe(true);
    expect(Object.isFrozen(deps.personalizedNoveltyMatches)).toBe(true);
    expect(deps.arxiv.topics[0]?.tag).toBe("manual");
    const serialized = JSON.stringify([deps.personalizedDiscovery, deps.personalizedNoveltyRepresentatives, deps.personalizedNoveltyMatches]);
    for (const privateValue of ["private-model-key", "sha256:", config.libraryConnection!.selectedRoot, ...Object.keys(catalog.files)]) {
      expect(serialized).not.toContain(privateValue);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(["unauthorized", "corrupt-profile", "replaced-root"])("keeps the manual-topic pipeline usable for %s", async (reason) => {
    const { config, host, fetch, selectedRoot, profileStore } = await fixture();
    if (reason === "unauthorized") {
      config.libraryConnection = revokeLibraryConnection(config.libraryConnection!);
    } else if (reason === "corrupt-profile") {
      await host.storage.writeText(profileStore.paths.documentPath, "{broken-profile");
      await host.storage.writeText(profileStore.paths.backupPath, "{broken-backup");
    } else {
      await fs.rename(selectedRoot, `${selectedRoot}-original`);
      await fs.mkdir(selectedRoot);
    }
    const runtime = trackRuntime(await buildCliRuntime(config, { host, logger: new Logger("debug") }));
    const deps = pipelineDeps(runtime.pipeline);
    expect(deps.personalizedDiscovery).toBeUndefined();
    expect(deps.personalizedNoveltyRepresentatives).toBeUndefined();
    expect(deps.personalizedNoveltyMatches).toBeUndefined();
    expect(deps.arxiv.topics[0]?.tag).toBe("manual");
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(["revoke", "endpoint"])("cancels a running personalized pipeline when its config changes (%s)", async (change) => {
    const { config, host, fetch } = await fixture();
    const runtime = trackRuntime(await buildCliRuntime(config, { host, logger: new Logger("debug") }));
    const personalizedSignal = pipelineDeps(runtime.pipeline).personalizedDiscoverySignal!;
    expect(personalizedSignal).toBeDefined();
    expect(personalizedSignal.aborted).toBe(false);

    let transportSignal: AbortSignal | undefined;
    fetch.mockImplementation(async (_input, init) => {
      transportSignal = init?.signal ?? undefined;
      return new Promise<Response>((_resolve, reject) => {
        const abort = () => reject(transportSignal?.reason ?? new Error("Request aborted"));
        if (transportSignal?.aborted) abort();
        else transportSignal?.addEventListener("abort", abort, { once: true });
      });
    });
    const cleanupController = new AbortController();
    const outcome = runtime.pipeline.runForDate("2026-10-01", cleanupController.signal);
    // Attach a handler immediately; cleanup also settles a failed assertion's request.
    void outcome.catch(() => undefined);
    try {
      await vi.waitFor(() => expect(transportSignal).toBeDefined(), { timeout: 3_000 });
      if (change === "revoke") {
        await revokeCliLibrary(config);
      } else {
        const document = parse(await fs.readFile(config.configPath, "utf8"));
        await fs.writeFile(config.configPath, stringify({ ...document, llm: {
          ...(document.llm as Record<string, unknown>), base_url: "https://changed-model.example/v1",
        } }));
      }
      await vi.waitFor(() => expect(personalizedSignal.aborted).toBe(true), { timeout: 3_000 });
      expect(transportSignal?.aborted).toBe(true);
      await expect(outcome).resolves.toMatchObject({ kind: "cancelled" });
    } finally {
      cleanupController.abort();
      await outcome.catch(() => undefined);
    }
  }, 10_000);
});
