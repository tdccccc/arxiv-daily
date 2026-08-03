import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  createEmptyPersonalLibraryInterestProfile,
  createPersonalLibraryRepresentativeSetFingerprint,
  isEmptyPersonalLibraryInterestProfile,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryInterestProfile,
} from "../src/library/personal-library-interest-profile";
import {
  PersonalLibraryDirectionProposalStore,
  PersonalLibraryInterestProfileStore,
  derivePersonalLibraryInterestProfileStorePaths,
} from "../src/library/personal-library-interest-profile-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const otherScope = `sha256:${"c".repeat(64)}`;
const otherIdentification = `sha256:${"d".repeat(64)}`;
const evidence = `sha256:${"e".repeat(64)}`;
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

function proposal(overrides: Partial<PersonalLibraryDirectionProposal> = {}): PersonalLibraryDirectionProposal {
  const representatives = [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: evidence }];
  return {
    schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
    revision: 99,
    proposalId: "proposal-1",
    scopeFingerprint: scope,
    identificationFingerprint: identification,
    catalogInputFingerprint: `sha256:${"f".repeat(64)}`,
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
      lineage: { proposalId: "proposal-1", candidateIds: ["candidate-1"], directionIds: [] },
      createdAt: firstTime.toISOString(), updatedAt: firstTime.toISOString(),
    }],
    ...overrides,
  };
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
    const b = proposal({ scopeFingerprint: otherScope, identificationFingerprint: otherIdentification,
      proposalId: "proposal-b" });
    await bProposal.replace(b, null);
    await a.profiles.replace(profile(), 0);
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(savedA);
    expect(Object.keys(memory.files).filter((path) => path.endsWith("direction-proposal.json"))).toHaveLength(2);
    expect(memory.files[profilePath]).toBeDefined();
  });
});

describe("proposal lifecycle", () => {
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

describe("recovery, errors, and serialization", () => {
  it("repairs compatible backup and fails valid incompatible primary without resurrection", async () => {
    const memory = makeStorage();
    const saved = proposal({ revision: 3 });
    memory.files[proposalBackupPath] = JSON.stringify(saved);
    await expect(stores(memory.storage).proposals.load()).resolves.toEqual(saved);
    expect(parse(memory.files[proposalPath])).toEqual(saved);

    memory.files[proposalPath] = JSON.stringify(proposal({ scopeFingerprint: otherScope }));
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
