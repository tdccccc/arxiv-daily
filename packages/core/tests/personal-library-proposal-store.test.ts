import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  decodePersonalLibraryDirectionProposal,
  createEmptyPersonalLibraryInterestProfile,
  createPersonalLibraryCatalogInputFingerprint,
  createPersonalLibraryCatalogInputManifest,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  isEmptyPersonalLibraryInterestProfile,
  type PersonalLibraryDirectionProposal,
} from "../src/library/personal-library-interest-profile";
import {
  PersonalLibraryDirectionProposalStore,
  PersonalLibraryDirectionProposalStoreError,
  PersonalLibraryInterestProfileStoreError,
  confirmPersonalLibraryDirectionWithStores,
  confirmPersonalLibraryDirectionsWithStores,
  derivePersonalLibraryInterestProfileStorePaths,
} from "../src/library/personal-library-proposal-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { PersonalLibraryCatalog, PersonalLibraryPaperRecord } from "../src/library/personal-library-catalog";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const otherScope = `sha256:${"c".repeat(64)}`;
const otherIdentification = `sha256:${"d".repeat(64)}`;
const evidence = createPersonalLibraryPaperEvidenceFingerprint({
  paperKey: "arxiv:2608.00001", source: "arxiv", externalId: "2608.00001",
  title: "Reliable agents", authors: ["A. Researcher"], abstract: "Reliable agents.",
  published: "2026-08-01T00:00:00.000Z", updated: "2026-08-02T00:00:00.000Z",
  primaryCategory: "cs.AI", categories: ["cs.AI"], evidenceDepth: "metadata-and-abstract",
  filePaths: ["papers/2608.00001.pdf"],
});
const firstTime = new Date("2026-08-03T12:00:00.000Z");
const secondTime = new Date("2026-08-03T13:00:00.000Z");
const directory = `arxiv-daily/.index/personal-library-profiles/${"a".repeat(64)}/${"b".repeat(64)}`;
const proposalPath = `${directory}/direction-proposal.json`;
const proposalBackupPath = `${proposalPath}.backup`;

function makeStorage(atomic = true) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  let atomicImplementation: ((path: string, content: string) => Promise<void>) | null = null;
  const normalizePath = vi.fn((path: string) => path.replace(/\\/g, "/")
    .replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""));
  const writeTextAtomic = vi.fn(async (path: string, content: string) => {
    if (atomicImplementation) return await atomicImplementation(path, content);
    files[path] = content;
  });
  const storage: StorageAdapter = {
    normalizePath,
    readText: vi.fn(async (path) => {
      if (!(path in files)) throw new Error(`unreadable ${path}`);
      return files[path]!;
    }),
    writeText: vi.fn(async (path, content) => { files[path] = content; }),
    ...(atomic ? { writeTextAtomic } : {}),
    exists: vi.fn(async (path) => path in files || dirs.has(path)),
    mkdir: vi.fn(async (path) => { dirs.add(path); }),
    remove: vi.fn(async (path) => { delete files[path]; dirs.delete(path); }),
    rename: vi.fn(async (from, to) => { files[to] = files[from]!; delete files[from]; }),
  };
  return {
    files, storage, writeTextAtomic,
    setAtomicImplementation(value: typeof atomicImplementation) { atomicImplementation = value; },
  };
}

function catalogPaper(): PersonalLibraryPaperRecord {
  return {
    paperKey: "arxiv:2608.00001", source: "arxiv", externalId: "2608.00001",
    title: "Reliable agents", authors: ["A. Researcher"], abstract: "Reliable agents.",
    published: "2026-08-01T00:00:00.000Z", updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI", categories: ["cs.AI"], evidenceDepth: "metadata-and-abstract",
    filePaths: ["papers/2608.00001.pdf"],
  };
}

function confirmationCatalog(): PersonalLibraryCatalog {
  const paper = catalogPaper();
  return {
    schemaVersion: 1, revision: 1, scopeFingerprint: scope, identificationFingerprint: identification,
    updatedAt: firstTime.toISOString(), lastScan: null,
    files: { [paper.filePaths[0]!]: {
      path: paper.filePaths[0]!, status: "ready", observationFingerprint: `sha256:${"2".repeat(64)}`,
      paperKey: paper.paperKey, arxivId: paper.externalId, updatedAt: firstTime.toISOString(),
    } },
    papers: { [paper.paperKey]: paper },
  };
}

function proposal(overrides: Partial<PersonalLibraryDirectionProposal> = {}): PersonalLibraryDirectionProposal {
  const paper = catalogPaper();
  const representatives = [{
    paperKey: paper.paperKey, evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(paper),
  }];
  return {
    schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
    revision: 99,
    proposalId: "proposal-1",
    scopeFingerprint: scope,
    identificationFingerprint: identification,
    catalogInputFingerprint: createPersonalLibraryCatalogInputFingerprint({
      scopeFingerprint: scope, identificationFingerprint: identification,
      papers: Object.values(confirmationCatalog().papers),
    }),
    catalogInputPapers: createPersonalLibraryCatalogInputManifest(Object.values(confirmationCatalog().papers)),
    generationContractFingerprint: `sha256:${"1".repeat(64)}`,
    generatedAt: firstTime.toISOString(),
    topics: [{
      id: "topic-1",
      suggestedName: "Reliable agents",
      directions: [{
        id: "candidate-1", text: "Reliability of long-running research agents",
        discoveryCues: ["agent reliability"], representatives,
        representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
        lineage: { candidateIds: ["candidate-1"] },
      }],
    }],
    ...overrides,
  };
}

function stores(storage: StorageAdapter, now = () => secondTime) {
  return {
    proposals: new PersonalLibraryDirectionProposalStore(
      storage, DEFAULT_SETTINGS.output, scope, identification,
    ),
  };
}

function parse<T>(raw: string | undefined): T {
  if (!raw) throw new Error("missing document");
  return JSON.parse(raw) as T;
}

function codeOf(caught: unknown): string | undefined {
  return (caught as { code?: string }).code;
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function flatLegacyProposal(schemaVersion: 2 | 3 = 3) {
  const { topics, ...fields } = proposal();
  return {
    ...fields,
    schemaVersion,
    candidates: topics.flatMap(({ directions }) => directions.map(({ text, ...direction }) => ({
      ...direction,
      name: "Reviewed legacy name",
      description: `${text}\nPreserved reviewed detail`,
      clusterMembers: [{ paperKey: "arxiv:2608.00001", confidence: 0.9 }],
    }))),
  };
}

describe("explicit regeneration of retired proposals", () => {
  it.each([2, 3] as const)("archives both v%i files without overwriting older archives", async (version) => {
    const memory = makeStorage();
    const store = new PersonalLibraryDirectionProposalStore(memory.storage, DEFAULT_SETTINGS.output, scope, identification, {
      now: () => secondTime,
    });
    const primaryRaw = JSON.stringify(flatLegacyProposal(version), null, 4);
    const backupRaw = JSON.stringify({ ...flatLegacyProposal(version), revision: 1 });
    memory.files[proposalPath] = primaryRaw;
    memory.files[proposalBackupPath] = backupRaw;
    const oldArchive = `${proposalPath}.retired-20260803T130000000Z-0`;
    memory.files[oldArchive] = "Do not overwrite this archive";
    const fresh = await store.replace(proposal(), null, { regenerateRetired: true });
    expect(fresh).toMatchObject({ schemaVersion: 6, revision: 0 });
    expect(memory.files[oldArchive]).toBe("Do not overwrite this archive");
    expect(memory.files[`${proposalPath}.retired-20260803T130000000Z-1`]).toBe(primaryRaw);
    expect(memory.files[`${proposalBackupPath}.retired-20260803T130000000Z-1`]).toBe(backupRaw);
    await expect(store.load()).resolves.toEqual(fresh);
  });

  it.each(["current primary", "current backup", "corrupt", "foreign scope", "missing"])(
    "refuses explicit regeneration of %s",
    async (kind) => {
      const memory = makeStorage();
      const legacy = flatLegacyProposal();
      if (kind !== "missing") memory.files[proposalPath] = JSON.stringify(legacy);
      if (kind === "current primary") memory.files[proposalPath] = JSON.stringify(proposal());
      if (kind === "current backup") memory.files[proposalBackupPath] = JSON.stringify(proposal());
      if (kind === "corrupt") memory.files[proposalPath] = "corrupt JSON";
      if (kind === "foreign scope") memory.files[proposalPath] = JSON.stringify({
        ...legacy, scopeFingerprint: otherScope,
        catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
          scopeFingerprint: otherScope, identificationFingerprint: identification, catalogInputPapers: legacy.catalogInputPapers,
        }),
      });
      const originals = { ...memory.files };
      await expect(stores(memory.storage).proposals.replace(proposal(), null, { regenerateRetired: true })).rejects.toMatchObject({
        code: kind === "foreign scope" ? "incompatible" : kind === "corrupt" ? "corrupt-or-unreadable" : "stale",
      });
      expect(memory.files).toEqual(originals);
      expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    },
  );

  it("preserves the original files if archiving fails", async () => {
    const memory = makeStorage();
    const raw = JSON.stringify(flatLegacyProposal());
    memory.files[proposalPath] = raw;
    memory.files[proposalBackupPath] = raw;
    memory.setAtomicImplementation(async () => { throw new Error("archive write failed"); });
    await expect(stores(memory.storage).proposals.replace(proposal(), null, { regenerateRetired: true }))
      .rejects.toMatchObject({ code: "save-failed" });
    expect(memory.files).toEqual({ [proposalPath]: raw, [proposalBackupPath]: raw });
  });

  it("retains both archived originals when promotion of the generated proposal fails", async () => {
    const memory = makeStorage();
    const primaryRaw = JSON.stringify({ ...flatLegacyProposal(), revision: 2 });
    const backupRaw = JSON.stringify({ ...flatLegacyProposal(), revision: 1 });
    memory.files[proposalPath] = primaryRaw;
    memory.files[proposalBackupPath] = backupRaw;
    memory.setAtomicImplementation(async (path, content) => {
      if (path === proposalPath) throw new Error("promotion failed");
      memory.files[path] = content;
    });
    await expect(stores(memory.storage).proposals.replace(proposal(), null, { regenerateRetired: true }))
      .rejects.toMatchObject({ code: "save-failed" });
    const archives = Object.entries(memory.files).filter(([path]) => path.includes(".retired-"));
    expect(archives.map(([, raw]) => raw)).toEqual([primaryRaw, backupRaw]);
    expect(memory.files[proposalPath]).toBe(primaryRaw);
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "regeneration-required" });
  });

  it("archives and replaces a valid retired backup when there is no primary", async () => {
    const memory = makeStorage();
    const raw = JSON.stringify(flatLegacyProposal());
    memory.files[proposalBackupPath] = raw;
    const fresh = await stores(memory.storage).proposals.replace(proposal(), null, { regenerateRetired: true });
    expect(Object.entries(memory.files).filter(([path]) => path.includes(".retired-")))
      .toEqual([[expect.stringContaining(`${proposalBackupPath}.retired-`), raw]]);
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(fresh);
  });

  it("rechecks saved files after archiving and refuses a newly saved current draft", async () => {
    const memory = makeStorage();
    memory.files[proposalPath] = JSON.stringify(flatLegacyProposal());
    const otherDraft = JSON.stringify(proposal({ proposalId: "other-reviewed-draft" }));
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path.includes(".retired-")) memory.files[proposalPath] = otherDraft;
    });
    await expect(stores(memory.storage).proposals.replace(proposal(), null, { regenerateRetired: true }))
      .rejects.toMatchObject({ code: "stale" });
    expect(memory.files[proposalPath]).toBe(otherDraft);
    expect(memory.files[proposalBackupPath]).toBeUndefined();
  });
});

describe("scope-bound paths and construction", () => {
  it("validates bound fingerprints before path normalization or I/O", () => {
    const { storage } = makeStorage();
    expect(() => new PersonalLibraryDirectionProposalStore(
      storage, DEFAULT_SETTINGS.output, "bad", identification,
    )).toThrow(expect.objectContaining({ code: "invalid" }));
    expect(storage.normalizePath).not.toHaveBeenCalled();
  });

});

describe("proposal lifecycle", () => {
  it.each([2, 3] as const)("identifies valid flat v%i proposals without replacing either saved file", async (version) => {
    const memory = makeStorage();
    const legacy = flatLegacyProposal(version);
    const primaryRaw = JSON.stringify({ ...legacy, revision: 2 });
    const backupRaw = JSON.stringify({ ...legacy, revision: 1 });
    memory.files[proposalPath] = primaryRaw;
    memory.files[proposalBackupPath] = backupRaw;
    const store = stores(memory.storage).proposals;
    await expect(store.load()).rejects.toMatchObject({ code: "regeneration-required" });
    await expect(store.replace(proposal(), null)).rejects.toMatchObject({ code: "regeneration-required" });
    expect(memory.files[proposalPath]).toBe(primaryRaw);
    expect(memory.files[proposalBackupPath]).toBe(backupRaw);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("recognizes a valid legacy backup when the primary is missing", async () => {
    const memory = makeStorage();
    memory.files[proposalBackupPath] = JSON.stringify(flatLegacyProposal());
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "regeneration-required" });
    expect(memory.files[proposalPath]).toBeUndefined();
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it.each(["missing fields", "candidate shape", "fingerprint", "current corruption", "unreadable primary"])(
    "does not treat %s as a regeneratable flat legacy proposal",
    async (kind) => {
      const memory = makeStorage();
      const legacy = flatLegacyProposal();
      if (kind === "missing fields") memory.files[proposalPath] = JSON.stringify({ schemaVersion: 3 });
      if (kind === "candidate shape") memory.files[proposalPath] = JSON.stringify({
        ...legacy, candidates: [{ ...legacy.candidates[0], name: "" }],
      });
      if (kind === "fingerprint") memory.files[proposalPath] = JSON.stringify({ ...legacy, catalogInputFingerprint: otherScope });
      if (kind === "current corruption") memory.files[proposalPath] = JSON.stringify({ ...proposal(), topics: null });
      if (kind === "unreadable primary") {
        memory.files[proposalPath] = JSON.stringify(legacy);
        vi.mocked(memory.storage.readText).mockRejectedValueOnce(new Error("permission denied"));
      }
      memory.files[proposalBackupPath] = JSON.stringify(legacy);
      await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
      expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    },
  );

  it("refuses a valid legacy generation belonging to another scope", async () => {
    const memory = makeStorage();
    const legacy = flatLegacyProposal();
    memory.files[proposalPath] = JSON.stringify({
      ...legacy,
      scopeFingerprint: otherScope,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: otherScope, identificationFingerprint: identification, catalogInputPapers: legacy.catalogInputPapers,
      }),
    });
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "incompatible" });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("keeps a usable current proposal even when an older backup remains", async () => {
    const memory = makeStorage();
    const current = proposal();
    memory.files[proposalPath] = JSON.stringify(current);
    memory.files[proposalBackupPath] = JSON.stringify(flatLegacyProposal());
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(current);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("reloads coverage evidence without losing the direction comparison basis", async () => {
    const memory = makeStorage();
    const original = proposal();
    const paperKeys = original.catalogInputPapers.map(({ paperKey }) => paperKey);
    const coverageEvidence = [{ topicId: "existing", directionId: "followed", directionText: "Reliable agents", paperKeys }];
    await stores(memory.storage).proposals.replace({ ...original, topics: [], coveredPaperKeys: paperKeys, coverageEvidence }, null);
    const reloaded = await stores(memory.storage).proposals.load();
    expect(reloaded?.coverageEvidence).toEqual(coverageEvidence);
  });

  it.each([4, 5])("retires a v%i proposal on load and allows a fresh generation with existing coverage", async (schemaVersion) => {
    const memory = makeStorage();
    const legacy = { ...proposal(), schemaVersion };
    memory.files[proposalPath] = JSON.stringify(legacy);
    memory.files[proposalBackupPath] = JSON.stringify(legacy);
    expect(decodePersonalLibraryDirectionProposal(legacy)).toBeNull();
    const store = stores(memory.storage).proposals;
    await expect(store.load()).rejects.toMatchObject({ code: "regeneration-required" });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    const fresh = await store.replace(proposal(), null, { regenerateRetired: true });
    expect(fresh.schemaVersion).toBe(6);
    await expect(store.load()).resolves.toEqual(fresh);
  });

  it("persists covered empty generations and explicit target selections on reload", async () => {
    const memory = makeStorage();
    const covered = proposal({ topics: [], coveredPaperKeys: ["arxiv:2608.00001"] });
    const first = await stores(memory.storage).proposals.replace(covered, null);
    expect(first).toMatchObject({ topics: [], coveredPaperKeys: ["arxiv:2608.00001"], revision: 0 });
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(first);
    const extension = proposal();
    extension.topics[0]!.targetTopicId = "existing-topic";
    const second = await stores(memory.storage).proposals.replace(extension, first.revision);
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(second);
    expect(second.topics[0]!.targetTopicId).toBe("existing-topic");
    expect(parse(memory.files[proposalBackupPath])).toEqual(first);
  });

  it("does not treat malformed or wrong-scope v5 data as a retired valid generation", async () => {
    const memory = makeStorage();
    memory.files[proposalPath] = JSON.stringify({ ...proposal(), schemaVersion: 5, coveredPaperKeys: ["arxiv:2608.00099"] });
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    const legacy = { ...proposal(), schemaVersion: 5 };
    memory.files[proposalPath] = JSON.stringify({ ...legacy,
      scopeFingerprint: otherScope,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: otherScope, identificationFingerprint: identification, catalogInputPapers: legacy.catalogInputPapers,
      }),
    });
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "incompatible" });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it.each([4, 5])("retires valid v%i proposals that used the former per-topic candidate bounds", async (schemaVersion) => {
    const memory = makeStorage();
    const legacy = proposal();
    const template = legacy.topics[0]!.directions[0]!;
    legacy.topics = [0, 1].map((topicIndex) => ({
      id: `topic.${topicIndex}`, suggestedName: `Topic ${topicIndex}`,
      directions: Array.from({ length: 7 }, (_, index) => {
        const id = `candidate.${topicIndex}.${index}`;
        return { ...template, id, lineage: { candidateIds: [id] } };
      }),
    }));
    const raw = JSON.stringify({ ...legacy, schemaVersion });
    memory.files[proposalPath] = raw;
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "regeneration-required" });
    expect(memory.files[proposalPath]).toBe(raw);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    await expect(stores(memory.storage).proposals.replace(proposal(), null, { regenerateRetired: true }))
      .resolves.toMatchObject({ schemaVersion: 6 });
  });

  it("does not retire a wrong-scope or corrupt proposal as if it were empty", async () => {
    const memory = makeStorage();
    const legacy = { ...proposal(), schemaVersion: 4 };
    memory.files[proposalBackupPath] = JSON.stringify(legacy);
    memory.files[proposalPath] = "corrupt current proposal";
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    const catalogInputPapers = legacy.catalogInputPapers;
    memory.files[proposalPath] = JSON.stringify({ ...legacy,
      scopeFingerprint: otherScope,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: otherScope, identificationFingerprint: identification, catalogInputPapers,
      }),
    });
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "incompatible" });
    memory.files[proposalBackupPath] = JSON.stringify(proposal());
    await expect(stores(memory.storage).proposals.load()).rejects.toMatchObject({ code: "incompatible" });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("fails closed with regeneration-required for legacy v1 primary or backup", async () => {
    const primary = makeStorage();
    const legacy = proposal() as unknown as Record<string, any>;
    legacy.schemaVersion = 1;
    delete legacy.catalogInputPapers;
    primary.files[proposalPath] = JSON.stringify(legacy);
    await expect(stores(primary.storage).proposals.load())
      .rejects.toMatchObject({ code: "regeneration-required" });

    const backup = makeStorage();
    backup.files[proposalPath] = "corrupt";
    backup.files[proposalBackupPath] = JSON.stringify(legacy);
    await expect(stores(backup.storage).proposals.load())
      .rejects.toMatchObject({ code: "regeneration-required" });
  });

  it("distinguishes missing null from a durable empty proposal generation", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).proposals;
    await expect(store.load()).resolves.toBeNull();
    const saved = await store.replace(proposal({ topics: [] }), null);
    expect(saved).toMatchObject({ revision: 0, topics: [] });
    await expect(store.load()).resolves.toEqual(saved);
  });

  it("seeds backup before first primary and rotates prior primary", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).proposals;
    const first = await store.replace(proposal(), null);
    expect(memory.writeTextAtomic.mock.calls.map(([path]) => path)).toEqual([
      proposalBackupPath, proposalPath,
    ]);
    const second = await store.replace({ ...first, generatedAt: secondTime.toISOString() }, 0);
    expect(second.revision).toBe(1);
    expect(parse(memory.files[proposalBackupPath])).toEqual(first);
  });

  it("accepts stale equal replay idempotently but rejects changed stale CAS", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).proposals;
    const first = await store.replace(proposal(), null);
    memory.writeTextAtomic.mockClear();
    await expect(store.replace({ ...first, revision: 999 }, null)).resolves.toEqual(first);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    const caught = await store.replace({ ...first, generatedAt: secondTime.toISOString() }, null)
      .catch((error) => error);
    expect(caught).toMatchObject({ code: "stale", expectedRevision: null, currentRevision: 0 });
  });

  it("makes first-generation committed-then-thrown retry idempotent", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).proposals;
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === proposalPath) throw new Error("response lost");
    });
    await expect(store.replace(proposal(), null)).rejects.toMatchObject({ code: "save-failed" });
    memory.setAtomicImplementation(null);
    await expect(store.replace(proposal(), null)).resolves.toMatchObject({ revision: 0 });
  });

  it("makes second-generation committed-then-thrown retry idempotent", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).proposals;
    const first = await store.replace(proposal(), null);
    const requested = { ...first, generatedAt: secondTime.toISOString() };
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === proposalPath) throw new Error("response lost");
    });
    await expect(store.replace(requested, first.revision))
      .rejects.toMatchObject({ code: "save-failed" });
    memory.setAtomicImplementation(null);
    memory.writeTextAtomic.mockClear();
    const committed = await store.replace(requested, first.revision);
    expect(committed).toMatchObject({ revision: 1, generatedAt: secondTime.toISOString() });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    await expect(store.replace({ ...requested, generatedAt: "2026-08-03T14:00:00.000Z" },
      first.revision)).rejects.toMatchObject({
      code: "stale", expectedRevision: 0, currentRevision: 1,
    });
  });

  it("validates next identity and only rejects exhaustion for changed content", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).proposals;
    await expect(store.replace(proposal({ scopeFingerprint: otherScope }), null))
      .rejects.toMatchObject({ code: "invalid" });
    const exhausted = proposal({ revision: Number.MAX_SAFE_INTEGER });
    memory.files[proposalPath] = JSON.stringify(exhausted);
    await expect(store.replace(proposal(), 1)).resolves.toEqual(exhausted);
    await expect(store.replace({ ...proposal(), generatedAt: secondTime.toISOString() },
      Number.MAX_SAFE_INTEGER)).rejects.toMatchObject({ code: "invalid" });
  });
});

describe("recovery, errors, and serialization", () => {
  it("repairs compatible backup and fails valid incompatible primary without resurrection", async () => {
    const memory = makeStorage();
    const saved = proposal({ revision: 3 });
    memory.files[proposalBackupPath] = JSON.stringify(saved);
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(saved);
    expect(parse(memory.files[proposalPath])).toEqual(saved);

    const incompatibleManifest = createPersonalLibraryCatalogInputManifest(Object.values(confirmationCatalog().papers));
    memory.files[proposalPath] = JSON.stringify(proposal({
      scopeFingerprint: otherScope,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: otherScope, identificationFingerprint: identification,
        catalogInputPapers: incompatibleManifest,
      }),
    }));
    memory.files[proposalBackupPath] = JSON.stringify(saved);
    const caught = await stores(memory.storage).proposals.load().catch((error) => error);
    expect(codeOf(caught)).toBe("incompatible");
  });

  it("preserves backup/primary invariants on promotion and backup failures", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).proposals;
    const first = await store.replace(proposal(), null);
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === proposalPath) throw new Error("ambiguous");
    });
    await expect(store.replace({ ...first, generatedAt: secondTime.toISOString() }, 0))
      .rejects.toMatchObject({ code: "save-failed" });
    expect(parse<PersonalLibraryDirectionProposal>(memory.files[proposalPath]).generatedAt)
      .toBe(secondTime.toISOString());
    expect(parse(memory.files[proposalBackupPath])).toEqual(first);

    memory.setAtomicImplementation(async (path, content) => {
      if (path === proposalBackupPath) throw new Error("backup failed");
      memory.files[path] = content;
    });
    const committed = parse<PersonalLibraryDirectionProposal>(memory.files[proposalPath]);
    await expect(store.replace({ ...committed, generatedAt: "2026-08-03T14:00:00.000Z" }, 1))
      .rejects.toMatchObject({ code: "save-failed" });
    expect(parse(memory.files[proposalPath])).toEqual(committed);
  });

});
