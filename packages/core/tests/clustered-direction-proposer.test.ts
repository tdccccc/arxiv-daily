import { describe, expect, it } from "vitest";
import {
  ClusteredDirectionsProposerError,
  PERSONAL_LIBRARY_CLUSTERED_DIRECTION_PROPOSER_VERSION,
  PERSONAL_LIBRARY_DIRECTION_MAX_COMPLETION_TOKENS,
  PERSONAL_LIBRARY_DIRECTION_MAX_OUTPUT_CODE_UNITS,
  PERSONAL_LIBRARY_DIRECTION_VALIDATION_ATTEMPTS,
  createPersonalLibraryClusteredDirectionGenerationContract,
  PERSONAL_LIBRARY_DIRECTION_SYNTHESIS_PROMPT_VERSION,
  proposeClusteredPersonalLibraryDirections,
  resolvePersonalLibraryClusteringOptions,
  type ProposeClusteredDirectionsOptions,
} from "../src/library/personal-library-direction-proposer";
import {
  clusterPaperVectors,
  type ClusteringOptions,
  type TwoLevelClusteringOptions,
} from "../src/library/clustering/clusterer";
import { buildClusteringInput } from "../src/library/clustering/paper-vector";
import {
  PERSONAL_LIBRARY_MAX_PROPOSAL_TOPICS,
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  decodePersonalLibraryDirectionProposal,
} from "../src/library/personal-library-interest-profile";
import type {
  PersonalLibraryCatalog,
  PersonalLibraryPaperRecord,
} from "../src/library/personal-library-catalog";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type FullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";
import type { ChatMessage, CallOptions } from "../src/llm/client";
import { RunCancelledError } from "../src/services/cancellation";

const DIMENSION = 8;
const scopeFingerprint = `sha256:${"a".repeat(64)}`;
const identificationFingerprint = `sha256:${"b".repeat(64)}`;
const timestamp = "2026-08-05T12:00:00.000Z";

const THEME_A = [1, 0, 0, 0, 0, 0, 0, 0];
const THEME_B = [0, 1, 0, 0, 0, 0, 0, 0];

function paper(index: number): PersonalLibraryPaperRecord {
  const externalId = `2608.${String(index).padStart(5, "0")}`;
  return {
    paperKey: `arxiv:${externalId}`,
    source: "arxiv",
    externalId,
    title: `Paper ${index}`,
    authors: ["A. Author"],
    abstract: `Abstract ${index}`,
    published: "2026-08-01T00:00:00.000Z",
    updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI",
    categories: ["cs.AI"],
    evidenceDepth: "metadata-and-abstract",
    filePaths: [`private/root/paper-${index}.pdf`],
  };
}

/** Deterministic pseudo-random perturbation around a base vector. */
function themeVector(base: number[], noise: number, seed: number): Float32Array {
  let state = seed;
  const rand = (): number => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  const out = new Float32Array(base.length);
  for (let index = 0; index < base.length; index += 1) {
    out[index] = (base[index] ?? 0) + noise * (rand() - 0.5);
  }
  return out;
}

function oneHot(dimension: number): Float32Array {
  const out = new Float32Array(DIMENSION);
  out[dimension] = 1;
  return out;
}

function catalog(records: PersonalLibraryPaperRecord[] = [1, 2, 3, 4, 5, 6, 7].map(paper)): PersonalLibraryCatalog {
  return {
    schemaVersion: 1,
    revision: 4,
    scopeFingerprint,
    identificationFingerprint,
    updatedAt: timestamp,
    lastScan: null,
    files: Object.fromEntries(records.map((entry, index) => [entry.filePaths[0]!, {
      path: entry.filePaths[0]!,
      status: "ready" as const,
      observationFingerprint: `sha256:${(index % 16).toString(16).repeat(64)}`,
      paperKey: entry.paperKey,
      arxivId: entry.externalId,
      updatedAt: timestamp,
    }])),
    papers: Object.fromEntries(records.map((entry) => [entry.paperKey, entry])),
  };
}

class MemoryKnowledgeBase implements FullTextKnowledgeBaseStore {
  readonly paths = {
    directory: "kb",
    manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/manifest.json.backup" },
    papersDirectory: "kb/papers",
  };
  constructor(
    private readonly manifestValue: FullTextKnowledgeBaseManifest,
    private readonly documents: ReadonlyMap<string, FullTextPaperDocument>,
  ) {}
  async loadManifest(): Promise<FullTextKnowledgeBaseManifest> { return this.manifestValue; }
  async replaceManifest(): Promise<FullTextKnowledgeBaseManifest> { throw new Error("not used"); }
  async loadPaper(paperKey: string): Promise<FullTextPaperDocument | null> {
    return this.documents.get(paperKey) ?? null;
  }
  async savePaper(): Promise<void> { throw new Error("not used"); }
  async removePaper(): Promise<void> {}
  async removeAll(): Promise<void> {}
}

function makeKnowledgeBase(
  entries: ReadonlyArray<readonly [paperIndex: number, vector: Float32Array]>,
  overrides: { scopeFingerprint?: string; identificationFingerprint?: string } = {},
): MemoryKnowledgeBase {
  const documents = new Map<string, FullTextPaperDocument>();
  const manifestPapers: Record<string, FullTextKnowledgeBaseManifest["papers"][string]> = {};
  for (const [index, vector] of entries) {
    const key = paper(index).paperKey;
    const textHash = `sha256:${String(index).padStart(64, "0")}`;
    const filePaths = [`kb/paper-${index}.pdf`];
    const observationFingerprints = [`sha256:${String(index + 100).padStart(64, "0")}`];
    documents.set(key, {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      paperKey: key,
      modelId: "fake",
      dimension: DIMENSION,
      textHash,
      filePaths,
      observationFingerprints,
      chunks: [{ index: 0, page: 1, text: `chunk ${index}` }],
      vectors: vector,
      updatedAt: "2026-08-05T00:00:00.000Z",
    });
    manifestPapers[key] = {
      paperKey: key,
      status: "ready",
      modelId: "fake",
      dimension: DIMENSION,
      textHash,
      filePaths,
      observationFingerprints,
      chunkCount: 1,
      updatedAt: "2026-08-05T00:00:00.000Z",
    };
  }
  const manifest: FullTextKnowledgeBaseManifest = {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: 1,
    scopeFingerprint: overrides.scopeFingerprint ?? scopeFingerprint,
    identificationFingerprint: overrides.identificationFingerprint ?? identificationFingerprint,
    modelId: "fake",
    dimension: DIMENSION,
    updatedAt: "2026-08-05T00:00:00.000Z",
    papers: manifestPapers,
  };
  return new MemoryKnowledgeBase(manifest, documents);
}

/** 2 theme clusters (papers 1-3, 4-6) plus 1 outlier (paper 7). */
function standardEntries(): Array<readonly [number, Float32Array]> {
  return [
    [1, themeVector(THEME_A, 0.1, 1)],
    [2, themeVector(THEME_A, 0.1, 2)],
    [3, themeVector(THEME_A, 0.1, 3)],
    [4, themeVector(THEME_B, 0.1, 4)],
    [5, themeVector(THEME_B, 0.1, 5)],
    [6, themeVector(THEME_B, 0.1, 6)],
    [7, themeVector([0.6, 0.6, 0.6, 0, 0, 0, 0, 0], 0, 7)],
  ];
}

type PaperDatum = { paperKey: string };
/** The synthesis stage is handed the provisional candidates, not papers. */
type SynthesisDatum = { candidates: unknown[] };
type StageDatum = PaperDatum[] | SynthesisDatum;

function paperData(messages: ChatMessage[]): StageDatum {
  const content = messages.find(({ role }) => role === "user")!.content;
  const match = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(content);
  if (!match) throw new Error("missing paper_data");
  return JSON.parse(match[1]!.replaceAll("&lt;/paper_data&gt;", "</paper_data>"));
}

/**
 * Extraction answers one candidate per cluster. The synthesis call is handed
 * those provisional candidates and, by default, hands them back unchanged, so
 * a test that is not about merging still sees exactly what the per-cluster
 * extractions produced.
 */
function defaultCandidate(data: StageDatum): string {
  // Naming sees only the finished direction lines.
  if (!Array.isArray(data) && "directions" in (data as any)) {
    return JSON.stringify({ name: "Theme topic" });
  }
  if (!Array.isArray(data)) return JSON.stringify({ candidates: (data as any).candidates });
  const key = data[0]!.paperKey;
  return JSON.stringify({ candidates: [{
    text: "Theme direction grounded in this cluster's evidence",
    discoveryCues: ["cluster evidence"],
    representativePaperKeys: [key],
  }] });
}

class ScriptedLlm implements PersonalLibraryDirectionLlmPort {
  calls: Array<{ messages: ChatMessage[]; options?: CallOptions }> = [];
  constructor(
    private readonly responder: (data: StageDatum, callIndex: number) => string = defaultCandidate,
  ) {}
  async call(messages: ChatMessage[], options?: CallOptions): Promise<string> {
    this.calls.push({ messages, options });
    return this.responder(paperData(messages), this.calls.length - 1);
  }
}

function ids(kind: "proposal" | "topic" | "candidate", ordinal: number): string {
  return `${kind}.${ordinal}`;
}

function proposeOptions(
  knowledgeBase: FullTextKnowledgeBaseStore,
  llm: PersonalLibraryDirectionLlmPort,
  extra: Partial<ProposeClusteredDirectionsOptions> = {},
): ProposeClusteredDirectionsOptions {
  return {
    catalog: catalog(),
    knowledgeBase,
    llm,
    now: () => new Date(timestamp),
    createId: ids,
    // The fine pass runs at the same ratio the single-level proposer used, so
    // a coarse cluster that does not sub-divide yields one direction and these
    // fixtures keep describing the clusters they were written against.
    clustering: { coarse: { relativeStopRatio: 0.65 }, fine: { relativeStopRatio: 0.65 } },
    ...extra,
  };
}

async function expectedClusters(store: FullTextKnowledgeBaseStore) {
  return clusterPaperVectors((await buildClusteringInput(store)).papers).clusters;
}

describe("proposeClusteredPersonalLibraryDirections", () => {
  it("runs extraction per fine cluster, then synthesis and naming per topic", async () => {
    const store = makeKnowledgeBase(standardEntries());
    const llm = new ScriptedLlm();
    const result = await proposeClusteredPersonalLibraryDirections(proposeOptions(store, llm));
    const clusters = await expectedClusters(store);
    expect(clusters).toHaveLength(2);
    // Each coarse cluster is one topic and runs its own three stages: one
    // extraction per fine cluster inside it (here the coarse cluster does not
    // sub-divide, so one), then synthesis and naming for the topic itself.
    expect(llm.calls).toHaveLength(clusters.length * 3);
    const extractionCalls = llm.calls.filter(({ messages }) => Array.isArray(paperData(messages)));
    expect(extractionCalls).toHaveLength(clusters.length);
    expect(llm.calls.filter(({ messages }) => "candidates" in (paperData(messages) as any)))
      .toHaveLength(clusters.length);
    expect(llm.calls.filter(({ messages }) => "directions" in (paperData(messages) as any)))
      .toHaveLength(clusters.length);
    const messageKeys = extractionCalls.map(({ messages }) =>
      (paperData(messages) as PaperDatum[]).map(({ paperKey }) => paperKey).sort(),
    );
    // Each extraction message is one fine cluster, and a fine cluster never
    // straddles two topics: every message's papers sit inside one coarse cluster.
    const coarseSets = clusters.map((cluster) => new Set(cluster.paperKeys));
    expect(messageKeys.every((keys) =>
      coarseSets.some((set) => keys.every((key) => set.has(key))))).toBe(true);
    // no member leaks into another cluster's extraction message
    const flattened = messageKeys.flat();
    expect(new Set(flattened).size).toBe(flattened.length);
    // every extraction/synthesis call uses the bounded settings of the shared stage
    expect(extractionCalls.every(({ options }) =>
      options?.temperature === 0
      && options.maxOutputCodeUnits === PERSONAL_LIBRARY_DIRECTION_MAX_OUTPUT_CODE_UNITS
      && options.maxCompletionTokens === PERSONAL_LIBRARY_DIRECTION_MAX_COMPLETION_TOKENS)).toBe(true);
    expect(result.topics).toHaveLength(2);
    expect(result.topics.every(({ directions }) => directions.length === 1)).toBe(true);
    expect(result.topics.map(({ suggestedName }) => suggestedName)).toEqual(["Theme topic", "Theme topic"]);
  });

  it("merges same-direction candidates within one topic, never across topics", async () => {
    const store = makeKnowledgeBase(standardEntries());
    // A coarse ratio low enough to hold the whole corpus as one topic, and the
    // usual fine ratio inside it, so the two fine clusters each name the same
    // direction and synthesis is the only stage able to see that.
    const surfaced: string[] = [];
    const llm = new ScriptedLlm((data) => {
      if (Array.isArray(data)) {
        const key = data[0]!.paperKey;
        surfaced.push(key);
        return JSON.stringify({ candidates: [{
          text: `One fine cluster's view of a shared direction (${key})`,
          discoveryCues: ["shared cue"],
          representativePaperKeys: [key],
        }] });
      }
      if ("directions" in (data as any)) return JSON.stringify({ name: "Shared topic" });
      // Synthesis merges only what this topic's own fine clusters surfaced.
      const keys = (data as any).candidates
        .flatMap((candidate: { representativePaperKeys: string[] }) => candidate.representativePaperKeys);
      return JSON.stringify({ candidates: [{
        text: "The single direction both fine clusters were describing",
        discoveryCues: ["shared cue"],
        representativePaperKeys: [...new Set<string>(keys)].sort(),
      }] });
    });
    const result = await proposeClusteredPersonalLibraryDirections(proposeOptions(store, llm, {
      // A fine ratio strict enough to split a coarse cluster: the topic stays
      // one topic, and synthesis is the only stage that can see its two fine
      // clusters describing the same direction.
      clustering: { coarse: { relativeStopRatio: 0.65 }, fine: { relativeStopRatio: 0.98 } },
    }));
    // Each topic collapses to the one direction its synthesis returned, and no
    // direction is shared between topics.
    expect(result.topics.every(({ directions }) => directions.length === 1)).toBe(true);
    expect(result.topics.every(({ suggestedName }) => suggestedName === "Shared topic")).toBe(true);
    expect(result.topics.flatMap(({ directions }) => directions).every(({ text }) =>
      text === "The single direction both fine clusters were describing")).toBe(true);
    const ids = result.topics.flatMap(({ directions }) => directions).map(({ id }) => id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("keeps the un-synthesized candidates when synthesis cannot be validated", async () => {
    const store = makeKnowledgeBase(standardEntries());
    // Extraction succeeds per cluster; every synthesis attempt comes back
    // unusable. The per-cluster work is already paid for, so a fragmented
    // proposal beats no proposal at all (ADR 0009 §2).
    const llm = new ScriptedLlm((data) => {
      if (Array.isArray(data)) return defaultCandidate(data);
      if ("directions" in (data as any)) return JSON.stringify({ name: "Theme topic" });
      return "not a JSON result";
    });
    const result = await proposeClusteredPersonalLibraryDirections(proposeOptions(store, llm));
    expect(result.topics).toHaveLength(2);
    expect(result.topics.every(({ directions }) => directions.length === 1)).toBe(true);
    expect(decodePersonalLibraryDirectionProposal(result)).toEqual(result);
    // the fallback candidates still carry their own cluster's members
    // Papers that fall out at the fine level back no direction — outliers have
    // never entered direction synthesis — but they stay in the input manifest,
    // so the review page can still show how much of the library is uncovered.
    const covered = new Set(result.topics.flatMap(({ directions }) => directions).flatMap((candidate) =>
      candidate.clusterMembers!.map(({ paperKey }) => paperKey)));
    const manifest = result.catalogInputPapers.map(({ paperKey }) => paperKey);
    expect(covered.size).toBeGreaterThan(0);
    expect([...covered].every((paperKey) => manifest.includes(paperKey))).toBe(true);
  });

  it("attaches every cluster member with its clustering confidence to each candidate", async () => {
    const store = makeKnowledgeBase(standardEntries());
    const llm = new ScriptedLlm();
    const result = await proposeClusteredPersonalLibraryDirections(proposeOptions(store, llm));
    const clusters = await expectedClusters(store);
    const coarseSets = clusters.map((cluster) => new Set(cluster.paperKeys));
    const members = result.topics.flatMap(({ directions }) => directions)
      .map((candidate) => [...(candidate.clusterMembers ?? [])]);
    // Members come from the fine pass inside one coarse cluster, so they are
    // bounded by that topic, canonically ordered, and in-range confidences.
    expect(members.every((entries) => entries.length > 0)).toBe(true);
    expect(members.every((entries) =>
      coarseSets.some((set) => entries.every(({ paperKey }) => set.has(paperKey))))).toBe(true);
    expect(members.every((entries) => entries.every(({ confidence }) =>
      confidence >= 0 && confidence <= 1))).toBe(true);
    expect(members.every((entries) => entries.every(({ paperKey }, index) =>
      index === 0 || entries[index - 1]!.paperKey < paperKey))).toBe(true);
    const covered = result.topics.flatMap(({ directions }) => directions).flatMap((candidate) =>
      candidate.clusterMembers!.map(({ paperKey }) => paperKey),
    );
    expect(new Set(covered).size).toBe(6);
    expect(decodePersonalLibraryDirectionProposal(result)).toEqual(result);
  });

  it("keeps the outlier pool in catalogInputPapers and out of every clusterMembers", async () => {
    const store = makeKnowledgeBase(standardEntries());
    const llm = new ScriptedLlm();
    const result = await proposeClusteredPersonalLibraryDirections(proposeOptions(store, llm));
    const outlier = paper(7).paperKey;
    const inputKeys = result.catalogInputPapers.map(({ paperKey }) => paperKey);
    expect(inputKeys).toContain(outlier);
    expect(inputKeys).toHaveLength(7);
    const covered = new Set(result.topics.flatMap(({ directions }) => directions).flatMap((candidate) =>
      candidate.clusterMembers!.map(({ paperKey }) => paperKey),
    ));
    expect(covered.has(outlier)).toBe(false);
    // the review interface derives the buffer pool by subtracting clusterMembers
    const bufferPool = inputKeys.filter((paperKey) => !covered.has(paperKey));
    expect(bufferPool).toEqual([outlier]);
    // envelope conventions match the unclustered proposer
    expect(result.schemaVersion).toBe(PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION);
    expect(result.revision).toBe(0);
    expect(result.proposalId).toBe("proposal.0");
    expect(result.scopeFingerprint).toBe(scopeFingerprint);
    expect(result.identificationFingerprint).toBe(identificationFingerprint);
    expect(result.generatedAt).toBe(timestamp);
  });

  it("fails with no-evidence when clustering leaves every paper in the outlier pool", async () => {
    const store = makeKnowledgeBase([
      [1, oneHot(0)],
      [2, oneHot(1)],
      [3, oneHot(2)],
    ]);
    const llm = new ScriptedLlm();
    const error = await proposeClusteredPersonalLibraryDirections(
      proposeOptions(store, llm, { clustering: { minClusterSize: 4 } }),
    ).catch((value) => value);
    expect(error).toBeInstanceOf(ClusteredDirectionsProposerError);
    expect(error).toMatchObject({ code: "no-evidence" });
    expect(llm.calls).toHaveLength(0);
  });

  it("fails with no-evidence pointing at full-text indexing when the knowledge base is empty", async () => {
    const store = makeKnowledgeBase([]);
    const llm = new ScriptedLlm();
    const error = await proposeClusteredPersonalLibraryDirections(proposeOptions(store, llm))
      .catch((value) => value);
    expect(error).toMatchObject({ code: "no-evidence" });
    expect((error as Error).message).toContain("full-text indexing");
    expect(llm.calls).toHaveLength(0);
  });

  it("rejects a knowledge base manifest whose scope or identification does not match the catalog", async () => {
    const scopeMismatch = makeKnowledgeBase(standardEntries(), { scopeFingerprint: `sha256:${"c".repeat(64)}` });
    const llm = new ScriptedLlm();
    const error = await proposeClusteredPersonalLibraryDirections(proposeOptions(scopeMismatch, llm))
      .catch((value) => value);
    expect(error).toMatchObject({ code: "catalog-invalid" });
    expect((error as Error).message).toContain("scope/identification fingerprints");
    expect(llm.calls).toHaveLength(0);

    const idMismatch = makeKnowledgeBase(standardEntries(), { identificationFingerprint: `sha256:${"d".repeat(64)}` });
    await expect(proposeClusteredPersonalLibraryDirections(proposeOptions(idMismatch, new ScriptedLlm())))
      .rejects.toMatchObject({ code: "catalog-invalid" });
  });

  it("bounds one topic's direction count by the proposal constant", async () => {
    const store = makeKnowledgeBase(standardEntries());
    // Synthesis returns one more direction than a topic may hold; the per-topic
    // bound is what stops it, since directions now live under a topic.
    const llm = new ScriptedLlm((data) => {
      if (Array.isArray(data)) return defaultCandidate(data);
      if ("directions" in (data as any)) return JSON.stringify({ name: "Theme topic" });
      const key = (data as any).candidates[0]!.representativePaperKeys[0]!;
      return JSON.stringify({ candidates:
        Array.from({ length: PERSONAL_LIBRARY_MAX_PROPOSAL_TOPICS }, (_, index) => ({
          text: `Direction ${index}`,
          discoveryCues: [`cue ${index}`],
          representativePaperKeys: [key],
        })),
      });
    });
    const result = await proposeClusteredPersonalLibraryDirections(proposeOptions(store, llm));
    // A topic filled exactly to the bound is accepted and still decodes; the
    // stage validator caps a single synthesis response at the same number, so
    // the proposer's own over-limit guard is defensive rather than reachable
    // through the model.
    expect(result.topics.every(({ directions }) =>
      directions.length === PERSONAL_LIBRARY_MAX_PROPOSAL_TOPICS)).toBe(true);
    expect(decodePersonalLibraryDirectionProposal(result)).toEqual(result);
    const overfull = structuredClone(result) as any;
    overfull.topics[0].directions.push({ ...overfull.topics[0].directions[0], id: "candidate.999" });
    expect(decodePersonalLibraryDirectionProposal(overfull)).toBeNull();
  });

  it("cancels between cluster extractions", async () => {
    const store = makeKnowledgeBase(standardEntries());
    const controller = new AbortController();
    const llm = new ScriptedLlm((data, callIndex) => {
      if (callIndex === 1) controller.abort("stop after first cluster");
      return defaultCandidate(data);
    });
    const error = await proposeClusteredPersonalLibraryDirections(
      proposeOptions(store, llm, { signal: controller.signal }),
    ).catch((value) => value);
    expect(error).toBeInstanceOf(RunCancelledError);
    expect(llm.calls).toHaveLength(2);
  });

  it("fails invalid catalogs before touching the knowledge base or the model", async () => {
    const llm = new ScriptedLlm();
    const error = await proposeClusteredPersonalLibraryDirections({
      catalog: {},
      knowledgeBase: makeKnowledgeBase(standardEntries()),
      llm,
      createId: ids,
    }).catch((value) => value);
    expect(error).toMatchObject({ code: "catalog-invalid" });
    expect(llm.calls).toHaveLength(0);
  });

  it("rejects knowledge base papers that are absent from the catalog", async () => {
    const store = makeKnowledgeBase(standardEntries());
    const llm = new ScriptedLlm();
    const error = await proposeClusteredPersonalLibraryDirections(
      proposeOptions(store, llm, { catalog: catalog([1, 2, 3, 4, 5, 6].map(paper)) }),
    ).catch((value) => value);
    expect(error).toMatchObject({ code: "catalog-invalid" });
    expect((error as Error).message).toContain("absent from the catalog");
    expect(llm.calls).toHaveLength(0);
  });

  it("stays robust on pathological identical-vector input (no oversized clusters)", async () => {
    // All papers with byte-identical theme chunks: a degenerate symmetric
    // input where every score ties and no mutual link exists. The engine must
    // not crash: it either finds no structure (no-evidence) or proposes
    // clusters that never exceed the schema's member bound.
    const entries: Array<readonly [number, Float32Array]> = [];
    for (let index = 1; index <= 600; index += 1) entries.push([index, themeVector(THEME_A, 0, index)]);
    const store = makeKnowledgeBase(entries);
    const llm = new ScriptedLlm();
    const result = await proposeClusteredPersonalLibraryDirections(
      proposeOptions(store, llm, { catalog: catalog(Array.from({ length: 600 }, (_, index) => paper(index + 1))) }),
    ).catch((value) => value);
    if (result instanceof Error) {
      // Degenerate symmetric input: no structure (no-evidence) or one giant
      // cluster beyond the schema bound (evidence-too-large) are both sane.
      expect(["no-evidence", "evidence-too-large"]).toContain((result as { code?: string }).code);
      expect(llm.calls).toHaveLength(0);
    } else {
      for (const candidate of result.topics.flatMap(({ directions }) => directions)) {
        expect(candidate.clusterMembers!.length).toBeLessThanOrEqual(512);
      }
      expect(llm.calls).toBeGreaterThan(0);
    }
  });
});

describe("clustered generation contract", () => {
  it("serializes the effective clustering parameters into the contract", () => {
    const defaults = createPersonalLibraryClusteredDirectionGenerationContract(
      resolvePersonalLibraryClusteringOptions(undefined),
    );
    expect(defaults.length).toBeLessThanOrEqual(4096);
    expect(defaults).toContain(PERSONAL_LIBRARY_CLUSTERED_DIRECTION_PROPOSER_VERSION);
    expect(defaults).toContain('"relativeStopRatio":0.65');
    expect(defaults).toContain('"minClusterSize":2');
    expect(defaults).toContain('"centerCorpus":true');
    const tuned = createPersonalLibraryClusteredDirectionGenerationContract(
      resolvePersonalLibraryClusteringOptions({ minClusterSize: 3, centerCorpus: false, relativeStopRatio: 0.8 }),
    );
    expect(tuned).not.toBe(defaults);
    expect(tuned).toContain('"relativeStopRatio":0.8');
    expect(tuned).toContain('"minClusterSize":3');
    expect(tuned).toContain('"centerCorpus":false');
  });

  /**
   * The contract's job is to make parameter drift detectable from a stored
   * proposal. It used to record `synthesisPrompt: "none"` and a
   * `...-no-synthesis` strategy while the proposer ran a synthesis stage, so
   * changing the synthesis prompt invalidated nothing.
   */
  it("records the synthesis stage the proposer actually runs", () => {
    const contract = createPersonalLibraryClusteredDirectionGenerationContract(
      resolvePersonalLibraryClusteringOptions(undefined),
    );
    expect(contract).toContain(PERSONAL_LIBRARY_DIRECTION_SYNTHESIS_PROMPT_VERSION);
    expect(contract).not.toContain('"synthesisPrompt":"none"');
    expect(contract).not.toContain("no-synthesis");
  });

  it("reflects clustering options in the proposal's generationContractFingerprint", async () => {
    const run = async (clustering: TwoLevelClusteringOptions): Promise<string> => {
      const llm = new ScriptedLlm();
      const result = await proposeClusteredPersonalLibraryDirections(
        proposeOptions(makeKnowledgeBase(standardEntries()), llm, { clustering }),
      );
      return result.generationContractFingerprint;
    };
    expect(await run({ coarse: { relativeStopRatio: 0.8 } }))
      .not.toBe(await run({ coarse: { relativeStopRatio: 0.65 } }));
  });
});
