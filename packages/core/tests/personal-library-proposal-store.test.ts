import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
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
    candidates: [{
      id: "candidate-1", name: "Reliable agents", description: "Reliable research agents.",
      discoveryCues: ["agent reliability"], representatives,
      representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
      lineage: { candidateIds: ["candidate-1"] },
    }],
    ...overrides,
  };
}

function legacyProfile(overrides: Partial<PersonalLibraryInterestProfile> = {}): Record<string, any> {
  const current = profile(overrides) as unknown as Record<string, any>;
  current.schemaVersion = 1;
  current.directions = current.directions.map((direction: Record<string, any>) => ({
    ...direction,
    lineage: {
      proposalId: direction.lineage.proposalIds[0],
      candidateIds: direction.lineage.candidateIds,
      directionIds: direction.lineage.directionIds,
    },
  }));
  return current;
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
    const saved = await store.replace(proposal({ candidates: [] }), null);
    expect(saved).toMatchObject({ revision: 0, candidates: [] });
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
