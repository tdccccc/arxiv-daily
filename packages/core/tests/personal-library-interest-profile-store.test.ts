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
  type PersonalLibraryInterestProfile,
} from "../src/library/personal-library-interest-profile";
import {
  PersonalLibraryDirectionProposalStore,
  PersonalLibraryDirectionProposalStoreError,
  PersonalLibraryInterestProfileStore,
  PersonalLibraryInterestProfileStoreError,
  confirmPersonalLibraryDirectionWithStores,
  derivePersonalLibraryInterestProfileStorePaths,
} from "../src/library/personal-library-interest-profile-store";
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
const profilePath = `${directory}/interest-profile.json`;
const profileBackupPath = `${profilePath}.backup`;

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

function profile(overrides: Partial<PersonalLibraryInterestProfile> = {}): PersonalLibraryInterestProfile {
  const representatives = [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: evidence }];
  return {
    schemaVersion: PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
    revision: 99,
    scopeFingerprint: scope,
    identificationFingerprint: identification,
    updatedAt: firstTime.toISOString(),
    directions: [{
      id: "direction-1", status: "active", name: "Reliable agents",
      description: "Reliable research agents.", discoveryCues: ["agent reliability"],
      representatives,
      representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
      lineage: { proposalIds: ["proposal-1"], candidateIds: ["candidate-1"], directionIds: [] },
      createdAt: firstTime.toISOString(), updatedAt: firstTime.toISOString(),
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
    profiles: new PersonalLibraryInterestProfileStore(
      storage, DEFAULT_SETTINGS.output, scope, identification, { now },
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
  it("derives exact normalized opaque identity-isolated paths", () => {
    const { storage } = makeStorage();
    expect(derivePersonalLibraryInterestProfileStorePaths(
      storage, DEFAULT_SETTINGS.output, scope, identification,
    )).toEqual({
      directory,
      proposal: { directory, documentPath: proposalPath, backupPath: proposalBackupPath },
      profile: { directory, documentPath: profilePath, backupPath: profileBackupPath },
    });
    expect(storage.normalizePath(directory)).toBe(directory);
    expect(proposalPath).not.toContain("sha256:");
  });

  it("validates bound fingerprints before path normalization or I/O", () => {
    const { storage } = makeStorage();
    expect(() => new PersonalLibraryDirectionProposalStore(
      storage, DEFAULT_SETTINGS.output, "bad", identification,
    )).toThrow(expect.objectContaining({ code: "invalid" }));
    expect(storage.normalizePath).not.toHaveBeenCalled();
  });

  it("keeps A→B→A identities and proposal/profile documents isolated", async () => {
    const memory = makeStorage();
    const a = stores(memory.storage);
    const bProposal = new PersonalLibraryDirectionProposalStore(
      memory.storage, DEFAULT_SETTINGS.output, otherScope, otherIdentification,
    );
    const savedA = await a.proposals.replace(proposal(), null);
    const otherManifest = createPersonalLibraryCatalogInputManifest(Object.values(confirmationCatalog().papers));
    const b = proposal({ scopeFingerprint: otherScope, identificationFingerprint: otherIdentification,
      proposalId: "proposal-b", catalogInputPapers: otherManifest,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: otherScope, identificationFingerprint: otherIdentification,
        catalogInputPapers: otherManifest,
      }) });
    await bProposal.replace(b, null);
    await a.profiles.replace(profile(), 0);
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(savedA);
    expect(Object.keys(memory.files).filter((path) => path.endsWith("direction-proposal.json"))).toHaveLength(2);
    expect(memory.files[profilePath]).toBeDefined();
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

describe("profile lifecycle", () => {
  it("rejects persisted v2 chronology where a direction is newer than the document", async () => {
    const memory = makeStorage();
    const malformed = profile({ revision: 4 });
    malformed.directions[0]!.updatedAt = secondTime.toISOString();
    memory.files[profilePath] = JSON.stringify(malformed);
    await expect(stores(memory.storage).profiles.load())
      .rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("migrates a valid v1 primary atomically and reloads strict v2 without semantic changes", async () => {
    const memory = makeStorage();
    memory.files[profilePath] = JSON.stringify(legacyProfile({ revision: 6 }));
    const loaded = await stores(memory.storage).profiles.load();
    expect(loaded).toEqual(profile({ revision: 6 }));
    expect(parse<PersonalLibraryInterestProfile>(memory.files[profilePath])).toEqual(loaded);
    expect(memory.writeTextAtomic).toHaveBeenCalledWith(
      profilePath, `${JSON.stringify(loaded, null, 2)}\n`,
    );
    memory.writeTextAtomic.mockClear();
    await expect(stores(memory.storage).profiles.load()).resolves.toEqual(loaded);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("recovers and migrates a valid v1 backup when primary is corrupt", async () => {
    const memory = makeStorage();
    memory.files[profilePath] = "corrupt";
    memory.files[profileBackupPath] = JSON.stringify(legacyProfile({ revision: 5 }));
    const loaded = await stores(memory.storage).profiles.load();
    expect(loaded).toEqual(profile({ revision: 5 }));
    expect(parse<PersonalLibraryInterestProfile>(memory.files[profilePath])).toEqual(loaded);
  });

  it("returns a strict unpersisted empty profile and first content is revision one", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).profiles;
    const empty = await store.load();
    expect(isEmptyPersonalLibraryInterestProfile(empty)).toBe(true);
    expect(memory.files).toEqual({});
    await expect(store.replace(empty, 0)).resolves.toEqual(empty);
    const saved = await store.replace(profile(), 0);
    expect(saved).toMatchObject({ revision: 1, updatedAt: secondTime.toISOString() });
    expect(memory.writeTextAtomic.mock.calls.map(([path]) => path)).toEqual([
      profileBackupPath, profilePath,
    ]);
  });

  it("accepts stale equal replay but rejects changed stale state", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).profiles;
    const first = await store.replace(profile(), 0);
    memory.writeTextAtomic.mockClear();
    await expect(store.replace({ ...first, revision: 0, updatedAt: firstTime.toISOString() }, 0))
      .resolves.toEqual(first);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    const changed = profile();
    changed.directions[0]!.name = "Changed";
    await expect(store.replace(changed, 0)).rejects.toMatchObject({
      code: "stale", expectedRevision: 0, currentRevision: 1,
    });
  });

  it("makes first profile committed-then-thrown retry idempotent", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).profiles;
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === profilePath) throw new Error("response lost");
    });
    await expect(store.replace(profile(), 0)).rejects.toMatchObject({ code: "save-failed" });
    memory.setAtomicImplementation(null);
    await expect(store.replace(profile(), 0)).resolves.toMatchObject({ revision: 1 });
  });

  it("makes second-generation profile committed-then-thrown retry idempotent", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).profiles;
    const first = await store.replace(profile(), 0);
    const requested = profile();
    requested.directions[0]!.name = "Second generation";
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === profilePath) throw new Error("response lost");
    });
    await expect(store.replace(requested, first.revision))
      .rejects.toMatchObject({ code: "save-failed" });
    memory.setAtomicImplementation(null);
    memory.writeTextAtomic.mockClear();
    const committed = await store.replace(requested, first.revision);
    expect(committed).toMatchObject({
      revision: 2,
      directions: [expect.objectContaining({ name: "Second generation" })],
    });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    const different = profile();
    different.directions[0]!.name = "Different stale update";
    await expect(store.replace(different, first.revision)).rejects.toMatchObject({
      code: "stale", expectedRevision: 1, currentRevision: 2,
    });
  });

  it("keeps updatedAt monotonic across backward clocks and direction timestamps", async () => {
    const memory = makeStorage();
    let now = secondTime;
    const store = stores(memory.storage, () => now).profiles;
    const first = await store.replace(profile(), 0);
    now = new Date("2020-01-01T00:00:00.000Z");
    const changed = profile();
    changed.directions[0]!.name = "Backward clock";
    const second = await store.replace(changed, first.revision);
    expect(second.updatedAt).toBe(first.updatedAt);

    const future = profile();
    future.directions[0]!.name = "Future direction";
    future.directions[0]!.updatedAt = "2030-01-01T00:00:00.000Z";
    future.updatedAt = future.directions[0]!.updatedAt;
    const third = await store.replace(future, second.revision);
    expect(third.updatedAt).toBe("2030-01-01T00:00:00.000Z");
  });

  it("rejects an invalid clock with typed invalid error and no writes", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage, () => new Date(Number.NaN)).profiles;
    await expect(store.replace(profile(), 0)).rejects.toMatchObject({ code: "invalid" });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("returns defensive clones from replace and load", async () => {
    const memory = makeStorage();
    const store = stores(memory.storage).profiles;
    const saved = await store.replace(profile(), 0);
    saved.directions[0]!.name = "mutated";
    const loaded = await store.load();
    expect(loaded.directions[0]!.name).toBe("Reliable agents");
    loaded.directions[0]!.name = "mutated again";
    await expect(store.load()).resolves.toMatchObject({
      directions: [expect.objectContaining({ name: "Reliable agents" })],
    });
    delete memory.files[profilePath];
    const recovered = await store.load();
    recovered.directions[0]!.name = "mutated recovery";
    await expect(store.load()).resolves.toMatchObject({
      directions: [expect.objectContaining({ name: "Reliable agents" })],
    });
  });
});

describe("profile-first confirmation coordinator", () => {
  function confirmationInput(memory: ReturnType<typeof makeStorage>) {
    const bound = stores(memory.storage);
    return {
      proposalStore: bound.proposals, profileStore: bound.profiles,
      proposal: proposal({ revision: 0 }), profile: createEmptyPersonalLibraryInterestProfile(scope, identification, firstTime),
      catalog: confirmationCatalog(), candidateId: "candidate-1", directionId: "direction-1",
      status: "active" as const,
      draft: { name: "Reliable agents", description: "Reliable research agents.",
        discoveryCues: ["agent reliability"], representativePaperKeys: ["arxiv:2608.00001"] },
      now: secondTime, expectedProposalRevision: 0, expectedProfileRevision: 0,
    };
  }

  it("persists profile before proposal and returns actual saved documents", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    const initial = await input.proposalStore.replace(input.proposal, null);
    input.proposal = initial;
    memory.writeTextAtomic.mockClear();
    const result = await confirmPersonalLibraryDirectionWithStores(input);
    const primaryWrites = memory.writeTextAtomic.mock.calls.map(([path]) => path)
      .filter((path) => path === profilePath || path === proposalPath);
    expect(primaryWrites).toEqual([profilePath, proposalPath]);
    expect(result.profile).toMatchObject({
      revision: 1, updatedAt: secondTime.toISOString(),
      directions: [expect.objectContaining({ id: "direction-1", updatedAt: secondTime.toISOString() })],
    });
    expect(result.proposal).toMatchObject({ revision: 1, candidates: [] });
    await expect(input.profileStore.load()).resolves.toEqual(result.profile);
    await expect(input.proposalStore.load()).resolves.toEqual(result.proposal);
  });

  it("leaves proposal untouched when profile persistence fails", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    const before = await input.proposalStore.load();
    memory.setAtomicImplementation(async (path, content) => {
      if (path === profilePath) throw new Error("profile failed");
      memory.files[path] = content;
    });
    await expect(confirmPersonalLibraryDirectionWithStores(input)).rejects.toMatchObject({
      code: "partial-confirmation-conflict",
      details: expect.objectContaining({ stage: "profile-state-unreadable" }),
    });
    await expect(input.proposalStore.load()).resolves.toEqual(before);
  });

  it("continues after profile committed-then-thrown and consumes proposal once", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    let profilePrimaryWrites = 0;
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === profilePath && profilePrimaryWrites++ === 0) throw new Error("profile response lost");
    });
    const result = await confirmPersonalLibraryDirectionWithStores(input);
    expect(result.profile.directions).toHaveLength(1);
    expect(result.proposal.candidates).toEqual([]);
    expect(memory.writeTextAtomic.mock.calls.filter(([path]) => path === proposalPath)).toHaveLength(2);
  });

  it("does not touch proposal when ambiguous profile state diverged", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    const originalProfileReplace = input.profileStore.replace.bind(input.profileStore);
    vi.spyOn(input.profileStore, "replace").mockImplementationOnce(async () => {
      const divergent = profile();
      divergent.directions[0]!.name = "Concurrent profile edit";
      await originalProfileReplace(divergent, 0);
      throw new PersonalLibraryInterestProfileStoreError("ambiguous", "save-failed");
    });
    const proposalWrites = memory.writeTextAtomic.mock.calls.filter(([path]) => path === proposalPath).length;
    await expect(confirmPersonalLibraryDirectionWithStores(input)).rejects.toMatchObject({
      code: "partial-confirmation-conflict",
      details: expect.objectContaining({ stage: "profile-divergent" }),
    });
    expect(memory.writeTextAtomic.mock.calls.filter(([path]) => path === proposalPath)).toHaveLength(proposalWrites);
    await expect(input.proposalStore.load()).resolves.toEqual(input.proposal);
  });

  it("reconciles proposal failure and retry without duplicate direction", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    let failed = false;
    memory.setAtomicImplementation(async (path, content) => {
      if (path === proposalPath && !failed) { failed = true; throw new Error("proposal failed"); }
      memory.files[path] = content;
    });
    const result = await confirmPersonalLibraryDirectionWithStores(input);
    expect(result.profile.directions.map(({ id }) => id)).toEqual(["direction-1"]);
    expect(result.proposal.candidates).toEqual([]);
    await expect(confirmPersonalLibraryDirectionWithStores(input)).resolves.toMatchObject({
      profile: expect.objectContaining({ directions: [expect.objectContaining({ id: "direction-1" })] }),
      proposal: expect.objectContaining({ candidates: [] }),
    });
  });

  it("wraps proposal stale with partial-state semantics without a second replace", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    const replace = vi.spyOn(input.proposalStore, "replace").mockRejectedValueOnce(
      new PersonalLibraryDirectionProposalStoreError("stale injected", "stale"),
    );
    const caught = await confirmPersonalLibraryDirectionWithStores(input).catch((error) => error);
    expect(caught).toMatchObject({
      code: "partial-confirmation-conflict",
      details: expect.objectContaining({ stage: "proposal-write-failed-after-profile-commit" }),
      cause: expect.objectContaining({ code: "stale" }),
    });
    expect(replace).toHaveBeenCalledTimes(1);
  });

  it("accepts a second proposal retry that committed then threw after one final read", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    const originalReplace = input.proposalStore.replace.bind(input.proposalStore);
    let call = 0;
    vi.spyOn(input.proposalStore, "replace").mockImplementation(async (next, revision) => {
      call += 1;
      if (call === 1) throw new PersonalLibraryDirectionProposalStoreError("first precommit", "save-failed");
      const saved = await originalReplace(next, revision);
      throw new PersonalLibraryDirectionProposalStoreError(`second committed ${saved.revision}`, "save-failed");
    });
    const result = await confirmPersonalLibraryDirectionWithStores(input);
    expect(result.proposal.candidates).toEqual([]);
    expect(call).toBe(2);
  });

  it("wraps second proposal retry stale and preserves original proposal", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    let call = 0;
    const replace = vi.spyOn(input.proposalStore, "replace").mockImplementation(async () => {
      call += 1;
      if (call === 1) throw new PersonalLibraryDirectionProposalStoreError("first precommit", "save-failed");
      throw new PersonalLibraryDirectionProposalStoreError("retry stale", "stale");
    });
    const caught = await confirmPersonalLibraryDirectionWithStores(input).catch((error) => error);
    expect(caught).toMatchObject({
      code: "partial-confirmation-conflict",
      details: expect.objectContaining({ stage: "proposal-retry-failed-after-profile-commit" }),
      cause: expect.objectContaining({ code: "stale" }),
    });
    expect(replace).toHaveBeenCalledTimes(2);
    await expect(input.proposalStore.load()).resolves.toEqual(input.proposal);
  });

  it("preserves a concurrent proposal edit and reports partial confirmation conflict", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    const originalReplace = input.proposalStore.replace.bind(input.proposalStore);
    let first = true;
    const replace = vi.spyOn(input.proposalStore, "replace").mockImplementation(async (next, revision) => {
      if (!first) return await originalReplace(next, revision);
      first = false;
      const edited = { ...input.proposal, candidates: input.proposal.candidates.map((candidate) => ({
        ...candidate, name: "Concurrent researcher edit",
      })) };
      await originalReplace(edited, input.proposal.revision);
      throw new PersonalLibraryDirectionProposalStoreError("ambiguous", "save-failed");
    });
    await expect(confirmPersonalLibraryDirectionWithStores(input)).rejects.toMatchObject({
      code: "partial-confirmation-conflict",
    });
    expect(replace).toHaveBeenCalledTimes(1);
    await expect(input.proposalStore.load()).resolves.toMatchObject({
      candidates: [expect.objectContaining({ name: "Concurrent researcher edit" })],
    });
  });

  it("rejects concurrent stale profile confirmation before touching proposal", async () => {
    const memory = makeStorage();
    const input = confirmationInput(memory);
    input.proposal = await input.proposalStore.replace(input.proposal, null);
    const winner = await confirmPersonalLibraryDirectionWithStores(input);
    const proposalWrites = memory.writeTextAtomic.mock.calls.filter(([path]) => path === proposalPath).length;
    const stale = confirmationInput(memory);
    stale.proposal = input.proposal;
    stale.directionId = "direction-2";
    await expect(confirmPersonalLibraryDirectionWithStores(stale)).rejects.toMatchObject({ code: "stale" });
    expect(memory.writeTextAtomic.mock.calls.filter(([path]) => path === proposalPath)).toHaveLength(proposalWrites);
    expect(winner.profile.directions).toHaveLength(1);
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

  it("uses stable corrupt, atomic unsupported, repair, and save codes", async () => {
    const corrupt = makeStorage();
    corrupt.files[profilePath] = "bad";
    corrupt.files[profileBackupPath] = "also bad";
    await expect(stores(corrupt.storage).profiles.load())
      .rejects.toMatchObject({ code: "corrupt-or-unreadable" });

    const unsupported = makeStorage(false);
    await expect(stores(unsupported.storage).proposals.replace(proposal(), null))
      .rejects.toMatchObject({ code: "atomic-write-unsupported" });
    unsupported.files[profileBackupPath] = JSON.stringify(profile({ revision: 1 }));
    await expect(stores(unsupported.storage).profiles.load())
      .rejects.toMatchObject({ code: "atomic-write-unsupported" });

    const repair = makeStorage();
    repair.files[profileBackupPath] = JSON.stringify(profile({ revision: 1 }));
    repair.setAtomicImplementation(async () => { throw new Error("repair failed"); });
    await expect(stores(repair.storage).profiles.load())
      .rejects.toMatchObject({ code: "repair-failed" });
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

  it("serializes load and CAS across instances and rejected queues recover", async () => {
    const memory = makeStorage();
    const gate = deferred();
    let block = true;
    memory.setAtomicImplementation(async (path, content) => {
      if (path === profilePath && block) { block = false; await gate.promise; }
      memory.files[path] = content;
    });
    const first = stores(memory.storage).profiles;
    const second = stores(memory.storage).profiles;
    const saving = first.replace(profile(), 0);
    const loading = second.load();
    await Promise.resolve();
    gate.resolve();
    const [saved, loaded] = await Promise.all([saving, loading]);
    expect(loaded).toEqual(saved);

    const changedA = profile(); changedA.directions[0]!.name = "A";
    const changedB = profile(); changedB.directions[0]!.name = "B";
    const results = await Promise.allSettled([
      first.replace(changedA, saved.revision), second.replace(changedB, saved.revision),
    ]);
    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected")).toHaveLength(1);
    await expect(first.load()).resolves.toBeDefined();
  });
});
