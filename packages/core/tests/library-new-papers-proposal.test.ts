import { describe, expect, it } from "vitest";
import {
  ClusteredDirectionsProposerError,
  PERSONAL_LIBRARY_CLUSTERED_DIRECTION_PROPOSER_VERSION,
  PERSONAL_LIBRARY_DIRECTION_ORGANIZATION_PROMPT_VERSION,
  createPersonalLibraryClusteredDirectionGenerationContract,
  createPersonalLibraryNewPapersDirectionGenerationContract,
  proposeClusteredPersonalLibraryDirections,
  proposeNewPapersPersonalLibraryDirections,
  resolvePersonalLibraryClusteringOptions,
  type PersonalLibraryDirectionLlmPort,
} from "../src/library/personal-library-direction-proposer";
import { countNewPapersCoveredByExistingDirections, listUnaccountedLibraryPapers } from "../src/library/personal-library-change";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type FullTextPaperDocument,
  type FullTextPaperKnowledgeRecord,
} from "../src/library/fulltext/knowledge-base";
import type { PersonalLibraryCatalog, PersonalLibraryPaperRecord } from "../src/library/personal-library-catalog";
import {
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  accountedPaperKeysOf,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryGenerationContractFingerprint,
  decodePersonalLibraryDirectionProposal,
  type PersonalLibraryDirectionProposal,
} from "../src/library/personal-library-interest-profile";
import type { ChatMessage, CallOptions } from "../src/llm/client";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const evidence = `sha256:${"e".repeat(64)}`;
const time = "2026-10-15T12:00:00.000Z";

function paper(month: string, serial: number): PersonalLibraryPaperRecord {
  const externalId = `${month}.${String(serial).padStart(5, "0")}`;
  return {
    paperKey: `arxiv:${externalId}`, source: "arxiv", externalId,
    title: `Paper ${externalId}`, authors: ["A. Author"], abstract: `Abstract of ${externalId}`,
    published: "2026-10-01T00:00:00.000Z", updated: "2026-10-02T00:00:00.000Z",
    primaryCategory: "cs.AI", categories: ["cs.AI"], evidenceDepth: "metadata-and-abstract",
    filePaths: [`papers/${externalId}.pdf`],
  };
}

function catalogOf(papers: PersonalLibraryPaperRecord[], revision = 1): PersonalLibraryCatalog {
  return {
    schemaVersion: 1, revision, scopeFingerprint: scope, identificationFingerprint: identification,
    updatedAt: time, lastScan: null,
    files: Object.fromEntries(papers.map((entry, index) => [entry.filePaths[0]!, {
      path: entry.filePaths[0]!, status: "ready" as const,
      observationFingerprint: `sha256:${(index % 10).toString().repeat(64)}`,
      paperKey: entry.paperKey, arxivId: entry.externalId, updatedAt: time,
    }])),
    papers: Object.fromEntries(papers.map((entry) => [entry.paperKey, entry])),
  };
}

function record(paperKey: string, extra: Partial<FullTextPaperKnowledgeRecord> = {}): FullTextPaperKnowledgeRecord {
  return {
    paperKey, status: "ready", modelId: "fake", dimension: 8, filePaths: [`kb/${paperKey.replace(/\W/g, "_")}.pdf`],
    observationFingerprints: [evidence], chunkCount: 2, updatedAt: time, ...extra,
  };
}

/** A store that serves the manifest only; reading any paper (vector) fails the test. */
class ManifestOnlyKnowledgeBase implements FullTextKnowledgeBaseStore {
  readonly paths = {
    directory: "kb",
    manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/manifest.json.backup" },
    papersDirectory: "kb/papers",
  };
  paperReads: string[] = [];
  readonly manifestValue: FullTextKnowledgeBaseManifest;
  constructor(records: FullTextPaperKnowledgeRecord[]) {
    this.manifestValue = {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION, revision: 1,
      scopeFingerprint: scope, identificationFingerprint: identification,
      modelId: "fake", dimension: 8, updatedAt: time,
      papers: Object.fromEntries(records.map((entry) => [entry.paperKey, entry])),
    };
  }
  async loadManifest(): Promise<FullTextKnowledgeBaseManifest> { return this.manifestValue; }
  async replaceManifest(): Promise<FullTextKnowledgeBaseManifest> { throw new Error("not used"); }
  async loadPaper(paperKey: string): Promise<FullTextPaperDocument | null> {
    this.paperReads.push(paperKey);
    throw new Error("a new-papers update must not read paper vectors");
  }
  async savePaper(): Promise<void> { throw new Error("not used"); }
  async removePaper(): Promise<void> {}
  async removeAll(): Promise<void> {}
}

function previousProposal(
  inputKeys: string[],
  extra: Partial<PersonalLibraryDirectionProposal> = {},
): PersonalLibraryDirectionProposal {
  const catalogInputPapers = [...inputKeys].sort().map((paperKey) => ({ paperKey, evidenceFingerprint: evidence }));
  return {
    schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION, revision: 3, proposalId: "proposal.old",
    scopeFingerprint: scope, identificationFingerprint: identification,
    catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
      scopeFingerprint: scope, identificationFingerprint: identification, catalogInputPapers,
    }),
    catalogInputPapers,
    generationContractFingerprint: createPersonalLibraryGenerationContractFingerprint("old-contract"),
    generatedAt: "2026-10-01T00:00:00.000Z", topics: [], ...extra,
  };
}

type PaperDatum = { paperKey: string; title: string; abstract: string };
type GroupDatum = { id: string; paperCount: number; papers: PaperDatum[] };
type StageDatum = {
  existingTopics: Array<{ id: string; name: string; directions: Array<{ id: string; text: string }> }>;
  groups: GroupDatum[];
};

function paperData(messages: ChatMessage[]): StageDatum {
  const content = messages.find(({ role }) => role === "user")!.content;
  const match = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(content);
  if (!match) throw new Error("missing paper_data");
  return JSON.parse(match[1]!);
}

class ScriptedLlm implements PersonalLibraryDirectionLlmPort {
  calls: Array<{ messages: ChatMessage[]; data: StageDatum }> = [];
  constructor(private readonly responder: (data: StageDatum) => unknown) {}
  async call(messages: ChatMessage[], _options?: CallOptions): Promise<string> {
    const data = paperData(messages);
    this.calls.push({ messages, data });
    return JSON.stringify(this.responder(data));
  }
}

const existingTopics = [{
  id: "t-galaxies", name: "Galaxies",
  directions: [{ id: "d-surveys", text: "Galaxy formation and evolution from survey observations" }],
}, {
  id: "t-agents", name: "Agents",
  directions: [{ id: "d-tools", text: "Tool-using language model agents" }],
}] as const;

function ids(kind: "proposal" | "topic" | "candidate", ordinal: number): string {
  return `${kind}.${ordinal}`;
}

/** Four old papers a scan has seen and five fresh ones it has not. */
function scenario() {
  const old = [1, 2, 3, 4].map((serial) => paper("2608", serial));
  const fresh = [1, 2, 3, 4, 5].map((serial) => paper("2610", serial));
  const kb = new ManifestOnlyKnowledgeBase([...old, ...fresh].map((entry) => record(entry.paperKey)));
  return {
    old, fresh, kb, catalog: catalogOf([...old, ...fresh]),
    previous: previousProposal(old.map(({ paperKey }) => paperKey)),
  };
}

function run(
  setup: ReturnType<typeof scenario>,
  llm: PersonalLibraryDirectionLlmPort,
  extra: { existingTopics?: typeof existingTopics | []; previous?: PersonalLibraryDirectionProposal | null } = {},
) {
  return proposeNewPapersPersonalLibraryDirections({
    catalog: setup.catalog, knowledgeBase: setup.kb, llm,
    existingTopics: extra.existingTopics ?? existingTopics,
    previous: extra.previous === undefined ? setup.previous : extra.previous,
    now: () => new Date(time), createId: ids,
  });
}

/** Everything covered by the first existing direction. */
const coverAll = (data: StageDatum) => ({
  topics: [],
  coveredGroups: data.groups.map(({ id }) => ({ groupId: id, topicId: "t-galaxies", directionId: "d-surveys" })),
});

describe("proposeNewPapersPersonalLibraryDirections", () => {
  it("falls back to a full scan when there is no previous proposal or no existing topics", async () => {
    const setup = scenario();
    const llm = new ScriptedLlm(coverAll);
    await expect(run(setup, llm, { previous: null })).rejects.toMatchObject({ code: "no-baseline" });
    await expect(proposeNewPapersPersonalLibraryDirections({
      catalog: setup.catalog, knowledgeBase: setup.kb, llm, existingTopics, previous: undefined,
      now: () => new Date(time), createId: ids,
    })).rejects.toMatchObject({ code: "no-baseline" });
    await expect(run(setup, llm, { existingTopics: [] })).rejects.toMatchObject({ code: "no-baseline" });
    const otherLibrary = { ...setup.previous, scopeFingerprint: `sha256:${"9".repeat(64)}` };
    await expect(run(setup, llm, { previous: otherLibrary })).rejects.toBeInstanceOf(ClusteredDirectionsProposerError);
    expect(llm.calls).toHaveLength(0);
  });

  it("reports nothing-new, without calling the model, when every paper is accounted for", async () => {
    const setup = scenario();
    const all = [...setup.old, ...setup.fresh].map(({ paperKey }) => paperKey);
    const llm = new ScriptedLlm(coverAll);
    await expect(run(setup, llm, { previous: previousProposal(all) })).rejects.toMatchObject({ code: "nothing-new" });
    expect(llm.calls).toHaveLength(0);
  });

  it("sends only the unaccounted papers, each as its own group, with the existing direction text", async () => {
    const setup = scenario();
    const llm = new ScriptedLlm(coverAll);
    await run(setup, llm);
    expect(llm.calls).toHaveLength(1);
    const { data, messages } = llm.calls[0]!;
    expect(data.existingTopics).toEqual(existingTopics);
    expect(data.groups).toHaveLength(setup.fresh.length);
    expect(data.groups.every((group) => group.paperCount === 1 && group.papers.length === 1)).toBe(true);
    expect(data.groups.flatMap(({ papers }) => papers.map(({ paperKey }) => paperKey)).sort())
      .toEqual(setup.fresh.map(({ paperKey }) => paperKey).sort());
    const sent = messages.map(({ content }) => content).join("\n");
    for (const oldPaper of setup.old) {
      expect(sent).not.toContain(oldPaper.paperKey);
      expect(sent).not.toContain(oldPaper.abstract);
    }
    expect(sent).toContain("Galaxy formation and evolution from survey observations");
    expect(sent).toContain("a single paper");
    expect(setup.kb.paperReads).toEqual([]);
  });

  it("treats all-covered as a valid, empty addition", async () => {
    const setup = scenario();
    const result = await run(setup, new ScriptedLlm(coverAll));
    expect(result.topics).toEqual([]);
    expect(result.coveredPaperKeys).toEqual(setup.fresh.map(({ paperKey }) => paperKey).sort());
    expect(result.coverageEvidence).toEqual([{
      topicId: "t-galaxies", directionId: "d-surveys",
      directionText: existingTopics[0].directions[0].text,
      paperKeys: setup.fresh.map(({ paperKey }) => paperKey).sort(),
    }]);
    expect(result.catalogInputPapers.map(({ paperKey }) => paperKey)).toEqual(setup.fresh.map(({ paperKey }) => paperKey).sort());
    expect(decodePersonalLibraryDirectionProposal(result)).toEqual(result);
  });

  it("accepts a proposed direction in an existing topic and a proposed new topic", async () => {
    const setup = scenario();
    const [a, b, c, d, e] = setup.fresh.map(({ paperKey }) => paperKey) as [string, string, string, string, string];
    const byKey = (data: StageDatum, key: string) => data.groups.find((group) => group.papers[0]!.paperKey === key)!.id;
    const llm = new ScriptedLlm((data) => ({
      topics: [{
        suggestedName: "Agents", targetTopicId: "t-agents",
        directions: [{
          text: "Agent benchmarks and evaluation of tool use",
          discoveryCues: ["benchmark"], groupIds: [byKey(data, a), byKey(data, b)], representativePaperKeys: [a],
        }],
      }, {
        suggestedName: "Cosmic dust",
        directions: [{
          text: "Dust grains and their role in galaxy evolution",
          discoveryCues: ["dust"], groupIds: [byKey(data, c)], representativePaperKeys: [c],
        }],
      }],
      coveredGroups: [d, e].map((key) => ({ groupId: byKey(data, key), topicId: "t-galaxies", directionId: "d-surveys" })),
    }));
    const result = await run(setup, llm);
    expect(result.topics.map(({ suggestedName, targetTopicId }) => [suggestedName, targetTopicId])).toEqual([
      ["Agents", "t-agents"], ["Cosmic dust", undefined],
    ]);
    // Two single-paper groups became one direction; both are its members.
    expect(result.topics[0]!.directions[0]!.clusterMembers!.map(({ paperKey }) => paperKey)).toEqual([a, b]);
    expect(result.coveredPaperKeys).toEqual([d, e]);
    expect(decodePersonalLibraryDirectionProposal(result)).toEqual(result);
  });

  it("does not need a cluster: five single-paper groups never raise no-evidence", async () => {
    const setup = scenario();
    const llm = new ScriptedLlm((data) => ({
      topics: [{
        suggestedName: "New field",
        directions: [{
          text: "A field that all five new papers belong to",
          discoveryCues: ["new field"], groupIds: data.groups.map(({ id }) => id),
          representativePaperKeys: [data.groups[0]!.papers[0]!.paperKey],
        }],
      }],
    }));
    const result = await run(setup, llm);
    expect(result.topics[0]!.directions[0]!.clusterMembers).toHaveLength(5);
    expect(llm.calls).toHaveLength(1);
  });

  it("rejects an invalid model answer through the strict decode", async () => {
    const setup = scenario();
    const llm = new ScriptedLlm((data) => ({
      topics: [], coveredGroups: data.groups.slice(1).map(({ id }) => ({ groupId: id, topicId: "t-galaxies", directionId: "d-surveys" })),
    }));
    await expect(run(setup, llm)).rejects.toMatchObject({ name: "PersonalLibraryDirectionValidationError" });
    expect(llm.calls).toHaveLength(3);
  });

  it("records the inherited accounted set so the rest of the library does not look new again", async () => {
    const setup = scenario();
    const result = await run(setup, new ScriptedLlm(coverAll));
    const all = [...setup.old, ...setup.fresh].map(({ paperKey }) => paperKey).sort();
    expect(result.accountedPaperKeys).toEqual(all);
    expect(listUnaccountedLibraryPapers({ manifest: setup.kb.manifestValue, proposal: result })).toEqual([]);
    expect(listUnaccountedLibraryPapers({ manifest: setup.kb.manifestValue, proposal: setup.previous }))
      .toHaveLength(setup.fresh.length);
    // The new proposal analysed only the new papers, yet the old ones stay accounted.
    expect(result.catalogInputPapers).toHaveLength(setup.fresh.length);
  });

  it("keeps papers an earlier rescan accounted for beyond the previous input, and drops removed ones", async () => {
    const setup = scenario();
    const removed = "arxiv:2501.99999";
    const earlier = `file:sha256:${"1".repeat(64)}`;
    const kb = new ManifestOnlyKnowledgeBase([
      ...[...setup.old, ...setup.fresh].map((entry) => record(entry.paperKey)),
      record(earlier, { title: "An earlier file" }),
    ]);
    const previous = previousProposal(setup.old.map(({ paperKey }) => paperKey), {
      accountedPaperKeys: [...setup.old.map(({ paperKey }) => paperKey), removed, earlier].sort(),
    });
    const catalog = catalogOf([...setup.old, ...setup.fresh]);
    const result = await proposeNewPapersPersonalLibraryDirections({
      catalog, knowledgeBase: kb, llm: new ScriptedLlm(coverAll), existingTopics, previous,
      now: () => new Date(time), createId: ids,
    });
    expect(accountedPaperKeysOf(result)).toContain(earlier);
    expect(accountedPaperKeysOf(result)).not.toContain(removed);
    expect(accountedPaperKeysOf(result)).toEqual(expect.arrayContaining(setup.fresh.map(({ paperKey }) => paperKey)));
  });

  it("describes itself as a new-papers generation in the contract fingerprint", async () => {
    const setup = scenario();
    const result = await run(setup, new ScriptedLlm(coverAll));
    const contract = createPersonalLibraryNewPapersDirectionGenerationContract();
    expect(JSON.parse(contract)).toMatchObject({ scope: "new-papers", clustering: "none" });
    expect(contract.length).toBeLessThanOrEqual(4096);
    expect(result.scope).toBe("new-papers");
    expect(result.generationContractFingerprint).toBe(createPersonalLibraryGenerationContractFingerprint(contract));
    const full = createPersonalLibraryClusteredDirectionGenerationContract(resolvePersonalLibraryClusteringOptions());
    expect(JSON.parse(full)).toMatchObject({ scope: "full-library" });
    expect(createPersonalLibraryGenerationContractFingerprint(full)).not.toBe(result.generationContractFingerprint);
  });

  it("bumps the proposer and prompt versions for the changed prompt and contract", () => {
    expect(PERSONAL_LIBRARY_DIRECTION_ORGANIZATION_PROMPT_VERSION).toBe("personal-library-topic-organization-v4");
    expect(PERSONAL_LIBRARY_CLUSTERED_DIRECTION_PROPOSER_VERSION).toBe("personal-library-clustered-direction-proposer-v5");
  });

  it("falls back to an indexed title for non-arXiv files and leaves out a paper a scan could not use", async () => {
    const setup = scenario();
    const local = `file:sha256:${"2".repeat(64)}`;
    const untitled = `file:sha256:${"3".repeat(64)}`;
    const kb = new ManifestOnlyKnowledgeBase([
      ...[...setup.old, ...setup.fresh].map((entry) => record(entry.paperKey)),
      record(local, { title: "A local report on dust", abstract: "Local abstract" }),
      record(untitled),
    ]);
    const llm = new ScriptedLlm(coverAll);
    const result = await proposeNewPapersPersonalLibraryDirections({
      catalog: setup.catalog, knowledgeBase: kb, llm, existingTopics, previous: setup.previous,
      now: () => new Date(time), createId: ids,
    });
    const sent = llm.calls[0]!.data.groups.flatMap(({ papers }) => papers.map(({ paperKey }) => paperKey));
    expect(sent).toContain(local);
    expect(sent).not.toContain(untitled);
    // Not counted, so not sent, not accounted: the update never fails on it.
    expect(accountedPaperKeysOf(result)).not.toContain(untitled);
    expect(listUnaccountedLibraryPapers({ manifest: kb.manifestValue, proposal: result, catalog: setup.catalog })).toEqual([]);
  });

  it("reports nothing-new when the only unaccounted paper is one a scan could not use", async () => {
    const setup = scenario();
    const all = [...setup.old, ...setup.fresh].map(({ paperKey }) => paperKey);
    const kb = new ManifestOnlyKnowledgeBase([
      ...all.map((key) => record(key)), record(`file:sha256:${"3".repeat(64)}`),
    ]);
    const llm = new ScriptedLlm(coverAll);
    await expect(proposeNewPapersPersonalLibraryDirections({
      catalog: setup.catalog, knowledgeBase: kb, llm, existingTopics, previous: previousProposal(all),
      now: () => new Date(time), createId: ids,
    })).rejects.toMatchObject({ code: "nothing-new" });
    expect(llm.calls).toHaveLength(0);
  });
});


describe("the full scan records and carries the accounted set", () => {
  /** Real vectors are needed here: the full scan clusters. */
  function vectorsKnowledgeBase(records: FullTextPaperKnowledgeRecord[], vectors: Map<string, Float32Array>) {
    const base = new ManifestOnlyKnowledgeBase(records);
    base.loadPaper = async (paperKey: string) => {
      const record = base.manifestValue.papers[paperKey]!;
      const vector = vectors.get(paperKey);
      if (!vector) return null;
      return {
        schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION, paperKey, modelId: "fake", dimension: 8,
        textHash: `sha256:${"1".repeat(64)}`, ...(record.title ? { title: record.title, abstract: record.abstract } : {}),
        filePaths: [...record.filePaths], observationFingerprints: [...record.observationFingerprints],
        chunks: [{ index: 0, page: 1, text: "chunk" }], vectors: vector, updatedAt: time,
      } as FullTextPaperDocument;
    };
    return base;
  }

  const unit = (index: number, noise: number) => {
    const v = new Float32Array(8);
    v[index] = 1;
    v[(index + 1) % 8] = noise;
    return v;
  };

  it("includes the ungrouped paper and the same-title duplicate it merged away, and keeps inherited papers", async () => {
    const papers = [1, 2, 3, 4, 5, 6].map((serial) => paper("2610", serial));
    const vectors = new Map<string, Float32Array>();
    papers.slice(0, 3).forEach((entry, i) => vectors.set(entry.paperKey, unit(0, 0.01 * i)));
    papers.slice(3, 6).forEach((entry, i) => vectors.set(entry.paperKey, unit(1, 0.01 * i)));
    const title = "A Sufficiently Long Identifying Title For Duplicate Merge";
    const keep = `file:sha256:${"4".repeat(64)}`;
    const drop = `file:sha256:${"5".repeat(64)}`;
    vectors.set(keep, unit(0, 0.02));
    vectors.set(drop, unit(0, 0.02));
    const outlier = `file:sha256:${"6".repeat(64)}`;
    vectors.set(outlier, unit(5, 0));
    const inherited = `file:sha256:${"7".repeat(64)}`;
    const kb = vectorsKnowledgeBase([
      ...papers.map((entry) => record(entry.paperKey)),
      record(keep, { title, abstract: "kept abstract" }), record(drop, { title, abstract: "dropped abstract" }),
      record(outlier, { title: "Outlier local file", abstract: "outlier abstract" }),
      record(inherited, { title: "Inherited local file", abstract: "inherited abstract" }),
    ], vectors);
    const removed = "arxiv:2501.99999";
    // `inherited` was seen by an earlier scan but has no vectors now, so this run's
    // clustering input does not include it; it must stay accounted for regardless.
    const previous = previousProposal(["arxiv:2610.00001"], {
      accountedPaperKeys: ["arxiv:2610.00001", inherited, removed].sort(),
    });
    const result = await proposeClusteredPersonalLibraryDirections({
      catalog: catalogOf(papers), knowledgeBase: kb, llm: new ScriptedLlm(coverAll), previous, existingTopics,
      now: () => new Date(time), createId: ids, clustering: { similarityQuantile: 0.5, centerCorpus: false },
    });
    const accounted = accountedPaperKeysOf(result);
    expect(result.scope).toBe("full");
    // The outlier is in no group (ungrouped) yet is still in the scan input and accounted.
    expect(result.catalogInputPapers.map(({ paperKey }) => paperKey)).toContain(outlier);
    expect(accounted).toContain(outlier);
    expect(result.catalogInputPapers.map(({ paperKey }) => paperKey)).not.toContain(drop);
    expect(accounted).toContain(drop);
    expect(accounted).toContain(inherited);
    expect(accounted).not.toContain(removed);
    expect(accounted.length).toBeGreaterThan(result.catalogInputPapers.length);
    // The regression: a full scan whose accounted set outgrew its input is
    // still not an update of new papers.
    expect(countNewPapersCoveredByExistingDirections(result)).toBe(0);
    for (const key of result.catalogInputPapers.map(({ paperKey }) => paperKey)) expect(accounted).toContain(key);
    expect(accounted).toEqual([...accounted].sort());
    // The merged duplicate is not a new paper to the change count either.
    expect(listUnaccountedLibraryPapers({ manifest: kb.manifestValue, proposal: result })).toEqual([]);
  });

  it("writes the accounted set even without a previous proposal", async () => {
    const papers = [1, 2, 3].map((serial) => paper("2610", serial));
    const vectors = new Map(papers.map((entry, i) => [entry.paperKey, unit(0, 0.01 * i)] as const));
    const kb = vectorsKnowledgeBase(papers.map((entry) => record(entry.paperKey)), vectors);
    const result = await proposeClusteredPersonalLibraryDirections({
      catalog: catalogOf(papers), knowledgeBase: kb, llm: new ScriptedLlm(coverAll), existingTopics,
      now: () => new Date(time), createId: ids, clustering: { similarityQuantile: 0.5, centerCorpus: false },
    });
    expect(result.accountedPaperKeys).toEqual(papers.map(({ paperKey }) => paperKey).sort());
    expect(result.scope).toBe("full");
  });
});
