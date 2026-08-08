import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  applyAttachSuggestion,
  applyMergeSuggestion,
  applySplitSuggestion,
  buildNewDirectionDraft,
  SPLIT_DERIVED_PROPOSAL_MARKER,
  SUGGESTION_MEMBER_CONFIDENCE,
  type NewDirectionDraft,
} from "../src/library/incremental/apply";
import type { DirectionDiffSuggestion } from "../src/library/incremental/diff-suggestions";
import {
  INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION,
  IncrementalSuggestionsStore,
  IncrementalSuggestionsStoreError,
  createEmptyIncrementalSuggestionsDocument,
  decodeIncrementalSuggestion,
  decodeIncrementalSuggestionsDocument,
  deriveIncrementalSuggestionsPaths,
  type IncrementalSuggestionsDocument,
} from "../src/library/incremental/suggestions-store";
import {
  createEmptyPersonalLibraryInterestProfile,
  createPersonalLibraryRepresentativeSetFingerprint,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryInterestProfile,
  type PersonalLibraryRepresentativeEvidence,
} from "../src/library/personal-library-interest-profile";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { sha256Hex } from "../src/utils/digest";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const otherScope = `sha256:${"c".repeat(64)}`;
const firstTime = new Date("2026-08-05T09:00:00.000Z");
const secondTime = new Date("2026-08-05T10:00:00.000Z");
const t0 = "2026-08-04T08:00:00.000Z";
const directory = `arxiv-daily/.index/personal-library-incremental-suggestions/${"a".repeat(64)}/${"b".repeat(64)}`;
const documentPath = `${directory}/incremental-suggestions.json`;
const backupPath = `${documentPath}.backup`;

function key(index: number): string {
  return `arxiv:2608.${String(index).padStart(5, "0")}`;
}

function representative(paperKey: string, hex = "e"): PersonalLibraryRepresentativeEvidence {
  return { paperKey, evidenceFingerprint: `sha256:${hex.repeat(64)}` };
}

function attach(directionId: string, paperKeys: string[], reason: string): DirectionDiffSuggestion {
  return { kind: "attach", directionId, paperKeys, reason };
}

function newSuggestion(paperKeys: string[], reason: string): DirectionDiffSuggestion {
  return { kind: "new", paperKeys, reason };
}

function split(directionId: string, paperKeys: string[], reason: string): DirectionDiffSuggestion {
  return { kind: "split", directionId, paperKeys, reason };
}

function merge(directionIds: [string, string], reason: string): DirectionDiffSuggestion {
  return { kind: "merge", directionIds, reason };
}

function direction(
  id: string,
  options: {
    members?: string[];
    representatives?: PersonalLibraryRepresentativeEvidence[];
    locked?: boolean;
    mergedIntoDirectionId?: string;
    createdAt?: string;
  } = {},
): PersonalLibraryConfirmedDirection {
  const createdAt = options.createdAt ?? t0;
  const representatives = options.representatives ?? [representative(key(1))];
  const members = (options.members ?? []).map((paperKey) => ({ paperKey, confidence: 0.9 }));
  const base = {
    id,
    status: "active" as const,
    name: `Direction ${id}`,
    description: "Confirmed direction.",
    discoveryCues: ["cue"],
    representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    clusterMembers: members,
    timeline: [{ kind: "created", at: createdAt }],
    lineage: { proposalIds: [`proposal.${id}`], candidateIds: [`candidate.${id}`], directionIds: [] },
    createdAt,
    updatedAt: createdAt,
    ...(options.locked ? { lockedAt: createdAt } : {}),
  };
  if (options.mergedIntoDirectionId !== undefined) {
    return { ...base, status: "merged" as const, mergedIntoDirectionId: options.mergedIntoDirectionId };
  }
  return base;
}

function profile(directions: PersonalLibraryConfirmedDirection[] = []): PersonalLibraryInterestProfile {
  return {
    ...createEmptyPersonalLibraryInterestProfile(scope, identification, new Date(t0)),
    revision: 4,
    directions: [...directions].sort((left, right) => left.id.localeCompare(right.id)),
  };
}

function frozen<T>(value: T): T {
  const visit = (item: any): any => {
    if (item && typeof item === "object") {
      Object.values(item).forEach(visit);
      Object.freeze(item);
    }
    return item;
  };
  return visit(value);
}

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

function store(
  storage: StorageAdapter,
  now = () => secondTime,
  onWarning?: (message: string, error?: unknown) => void,
) {
  return new IncrementalSuggestionsStore(
    storage, DEFAULT_SETTINGS.output, scope, identification, { now, onWarning },
  );
}

function document(overrides: Partial<IncrementalSuggestionsDocument> = {}): IncrementalSuggestionsDocument {
  return {
    schemaVersion: INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION,
    revision: 99,
    scopeFingerprint: scope,
    identificationFingerprint: identification,
    updatedAt: firstTime.toISOString(),
    suggestions: [attach("d-a", [key(3)], "Papers continue the d-a theme.")],
    ...overrides,
  };
}

function parse<T>(raw: string | undefined): T {
  if (!raw) throw new Error("missing document");
  return JSON.parse(raw) as T;
}

describe("suggestions document decoding", () => {
  it("round-trips a canonical document through JSON", () => {
    const value = document({
      suggestions: [
        attach("d-a", [key(3)], "Papers continue the d-a theme."),
        merge(["d-a", "d-b"], "Both directions cover the same method."),
        newSuggestion([key(20)], "Fresh theme with no covering direction."),
        split("d-c", [key(10), key(11)], "Cluster overlaps d-c members but forms a distinct theme."),
      ],
    });
    expect(decodeIncrementalSuggestionsDocument(JSON.parse(JSON.stringify(value)))).toEqual(value);
  });

  it("accepts an empty suggestions array", () => {
    const decoded = decodeIncrementalSuggestionsDocument(document({ suggestions: [] }));
    expect(decoded).toMatchObject({ suggestions: [] });
    expect(decoded!.suggestions).toEqual([]);
  });

  it("round-trips the optional pending-authorization note", () => {
    const withPending = document({
      suggestions: [],
      pendingAuthorization: {
        bufferedPaperCount: 4,
        updatedAt: "2026-08-07T00:00:00.000Z",
      },
    });
    expect(decodeIncrementalSuggestionsDocument(
      JSON.parse(JSON.stringify(withPending)),
    )).toEqual(withPending);
  });

  it("rejects malformed pending-authorization notes", () => {
    const invalid: unknown[] = [
      { ...document(), pendingAuthorization: { bufferedPaperCount: -1, updatedAt: "2026-08-07T00:00:00.000Z" } },
      { ...document(), pendingAuthorization: { bufferedPaperCount: 1.5, updatedAt: "2026-08-07T00:00:00.000Z" } },
      { ...document(), pendingAuthorization: { bufferedPaperCount: 1, updatedAt: "2026-08-07" } },
      { ...document(), pendingAuthorization: { bufferedPaperCount: 1 } },
      { ...document(), pendingAuthorization: { bufferedPaperCount: 1, updatedAt: "2026-08-07T00:00:00.000Z", extra: true } },
      { ...document(), pendingAuthorization: "pending" },
    ];
    for (const value of invalid) {
      expect(decodeIncrementalSuggestionsDocument(value)).toBeNull();
    }
  });

  it("rejects documents with invalid schema, identity, or revision fields", () => {
    const invalid = [
      { ...document(), schemaVersion: 2 },
      { ...document(), revision: -1 },
      { ...document(), scopeFingerprint: "sha256:not-hex" },
      { ...document(), identificationFingerprint: "other" },
      { ...document(), updatedAt: "2026-08-05" },
      { ...document(), extra: true },
    ];
    for (const value of invalid) {
      expect(decodeIncrementalSuggestionsDocument(value)).toBeNull();
    }
  });

  it("rejects invalid suggestion kinds and malformed shapes", () => {
    const invalid: unknown[] = [
      { ...document(), suggestions: [{ kind: "explode", reason: "Unknown kind." }] },
      { ...document(), suggestions: [{ kind: "attach", directionId: "d-a", paperKeys: [key(1)], reason: "R", extra: true }] },
      { ...document(), suggestions: [{ kind: "attach", directionId: "d-a", paperKeys: [key(1)] }] },
      { ...document(), suggestions: ["not an object"] },
      { ...document(), suggestions: [{ kind: "new", paperKeys: [key(1)], reason: "R", clusterId: "c-1" }] },
    ];
    for (const value of invalid) {
      expect(decodeIncrementalSuggestionsDocument(value)).toBeNull();
    }
    expect(decodeIncrementalSuggestion(null)).toBeNull();
    expect(decodeIncrementalSuggestion([1, 2])).toBeNull();
  });

  it("rejects empty, duplicated, or unsorted paperKeys", () => {
    const invalid: unknown[] = [
      { ...document(), suggestions: [{ kind: "attach", directionId: "d-a", paperKeys: [], reason: "Empty keys." }] },
      { ...document(), suggestions: [{ kind: "new", paperKeys: [key(1), key(1)], reason: "Duplicated key." }] },
      { ...document(), suggestions: [{ kind: "split", directionId: "d-a", paperKeys: [key(2), key(1)], reason: "Unsorted keys." }] },
    ];
    for (const value of invalid) {
      expect(decodeIncrementalSuggestionsDocument(value)).toBeNull();
    }
  });

  it("rejects overlong, control-character, untrimmed, and empty reasons", () => {
    const invalid: unknown[] = [
      { ...document(), suggestions: [{ kind: "new", paperKeys: [key(1)], reason: "" }] },
      { ...document(), suggestions: [{ kind: "new", paperKeys: [key(1)], reason: "x".repeat(501) }] },
      { ...document(), suggestions: [{ kind: "new", paperKeys: [key(1)], reason: "line\nbreak" }] },
      { ...document(), suggestions: [{ kind: "new", paperKeys: [key(1)], reason: " leading space" }] },
    ];
    for (const value of invalid) {
      expect(decodeIncrementalSuggestionsDocument(value)).toBeNull();
    }
  });

  it("rejects malformed merge directionIds and non-opaque direction ids", () => {
    const invalid: unknown[] = [
      { ...document(), suggestions: [{ kind: "merge", directionIds: ["d-a"], reason: "One id." }] },
      { ...document(), suggestions: [{ kind: "merge", directionIds: ["d-a", "d-a"], reason: "Duplicate id." }] },
      { ...document(), suggestions: [{ kind: "merge", directionIds: ["d-b", "d-a"], reason: "Unsorted ids." }] },
      { ...document(), suggestions: [{ kind: "attach", directionId: "not a valid id!", paperKeys: [key(1)], reason: "Bad id." }] },
      { ...document(), suggestions: [{ kind: "split", directionId: "x".repeat(200), paperKeys: [key(1)], reason: "Overlong id." }] },
    ];
    for (const value of invalid) {
      expect(decodeIncrementalSuggestionsDocument(value)).toBeNull();
    }
  });

  it("rejects cross-suggestion conflicts like the T3 validation rules", () => {
    const sharedPaper = document({ suggestions: [
      attach("d-a", [key(10)], "First suggestion."),
      newSuggestion([key(10), key(11)], "Second suggestion."),
    ] });
    const splitMerge = document({ suggestions: [
      split("d-a", [key(10)], "Split out of d-a."),
      merge(["d-a", "d-b"], "Merge d-a and d-b."),
    ] });
    expect(decodeIncrementalSuggestionsDocument(sharedPaper)).toBeNull();
    expect(decodeIncrementalSuggestionsDocument(splitMerge)).toBeNull();
  });

  it("rejects suggestions stored out of canonical order", () => {
    const value = document({ suggestions: [
      merge(["d-a", "d-b"], "Merge first."),
      attach("d-a", [key(3)], "Attach second."),
    ] });
    expect(decodeIncrementalSuggestionsDocument(value)).toBeNull();
    // Canonical kind order: attach < merge < new < split.
    const canonical = document({ suggestions: [
      attach("d-a", [key(3)], "Attach first."),
      merge(["d-a", "d-b"], "Merge second."),
    ] });
    expect(decodeIncrementalSuggestionsDocument(canonical)).not.toBeNull();
  });
});

describe("incremental suggestions store", () => {
  it("validates bound fingerprints before path normalization or I/O", () => {
    const memory = makeStorage();
    expect(() => new IncrementalSuggestionsStore(
      memory.storage, DEFAULT_SETTINGS.output, "bad", identification,
    )).toThrow(expect.objectContaining({ code: "invalid", name: "IncrementalSuggestionsStoreError" }));
    expect(() => new IncrementalSuggestionsStore(
      memory.storage, DEFAULT_SETTINGS.output, scope, "sha256:not-hex",
    )).toThrow(expect.objectContaining({ code: "invalid" }));
    expect(memory.storage.normalizePath).not.toHaveBeenCalled();
  });

  it("exposes the sharded paths bound to the constructed identity", () => {
    const memory = makeStorage();
    expect(store(memory.storage).paths).toEqual({
      directory, documentPath, backupPath,
    });
    expect(deriveIncrementalSuggestionsPaths(memory.storage, DEFAULT_SETTINGS.output, scope, identification))
      .toEqual({ directory, documentPath, backupPath });
  });

  it("loads an unpersisted empty document and first replace writes revision one", async () => {
    const memory = makeStorage();
    const suggestions = store(memory.storage);
    const empty = await suggestions.load();
    expect(empty).toEqual({
      schemaVersion: INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION,
      revision: 0,
      scopeFingerprint: scope,
      identificationFingerprint: identification,
      updatedAt: secondTime.toISOString(),
      suggestions: [],
    });
    expect(memory.files).toEqual({});
    const saved = await suggestions.replace(document(), 0);
    expect(saved).toMatchObject({ revision: 1, suggestions: document().suggestions });
    expect(memory.writeTextAtomic.mock.calls.map(([path]) => path)).toEqual([backupPath, documentPath]);
    await expect(suggestions.load()).resolves.toEqual(saved);
  });

  it("returns the empty in-memory document when replacing with empty semantics", async () => {
    const memory = makeStorage();
    const suggestions = store(memory.storage);
    const saved = await suggestions.replace(
      createEmptyIncrementalSuggestionsDocument(scope, identification, firstTime), 0,
    );
    expect(saved).toMatchObject({ revision: 0, suggestions: [] });
    expect(memory.files).toEqual({});
  });

  it("rejects stale replacements with currentRevision and succeeds after correction", async () => {
    const memory = makeStorage();
    const suggestions = store(memory.storage);
    await suggestions.replace(document(), 0);
    const next = document({ suggestions: [
      attach("d-a", [key(3)], "Papers continue the d-a theme."),
      attach("d-b", [key(4)], "Papers continue the d-b theme."),
    ] });
    const caught = await suggestions.replace(next, 0).catch((value) => value);
    expect(caught).toBeInstanceOf(IncrementalSuggestionsStoreError);
    expect(caught).toMatchObject({
      code: "stale", expectedRevision: 0, currentRevision: 1, name: "IncrementalSuggestionsStoreError",
    });
    const saved = await suggestions.replace(next, 1);
    expect(saved).toMatchObject({ revision: 2 });
    await expect(suggestions.load()).resolves.toEqual(saved);
  });

  it("accepts stale equal replay idempotently without incrementing or writing", async () => {
    const memory = makeStorage();
    const suggestions = store(memory.storage);
    const first = await suggestions.replace(document(), 0);
    memory.writeTextAtomic.mockClear();
    const replayed = await suggestions.replace(
      { ...first, revision: 0, updatedAt: "2000-01-01T00:00:00.000Z" }, 0,
    );
    expect(replayed).toEqual(first);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    await expect(suggestions.load()).resolves.toEqual(first);
  });

  it("makes a committed-then-thrown first replace retry idempotent", async () => {
    const memory = makeStorage();
    const suggestions = store(memory.storage);
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === documentPath) throw new Error("response lost");
    });
    await expect(suggestions.replace(document(), 0)).rejects.toMatchObject({ code: "save-failed" });
    memory.setAtomicImplementation(null);
    memory.writeTextAtomic.mockClear();
    await expect(suggestions.replace(document(), 0)).resolves.toMatchObject({ revision: 1 });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("rotates the prior primary into the backup and keeps updatedAt monotonic", async () => {
    const memory = makeStorage();
    let now = secondTime;
    const suggestions = store(memory.storage, () => now);
    const first = await suggestions.replace(document(), 0);
    now = new Date("2020-01-01T00:00:00.000Z");
    const second = await suggestions.replace(document({ suggestions: [] }), 1);
    expect(second.updatedAt).toBe(first.updatedAt);
    expect(parse<IncrementalSuggestionsDocument>(memory.files[backupPath])).toEqual(first);
    expect(parse<IncrementalSuggestionsDocument>(memory.files[documentPath])).toEqual(second);
  });

  it("repairs a corrupt primary from a valid backup and warns", async () => {
    const memory = makeStorage();
    const saved = { ...document(), revision: 3 };
    memory.files[documentPath] = "corrupt";
    memory.files[backupPath] = `${JSON.stringify(saved, null, 2)}\n`;
    const onWarning = vi.fn();
    const loaded = await store(memory.storage, () => secondTime, onWarning).load();
    expect(loaded).toEqual(saved);
    expect(parse<IncrementalSuggestionsDocument>(memory.files[documentPath])).toEqual(saved);
    expect(onWarning).toHaveBeenCalledTimes(1);
    expect(onWarning.mock.calls[0]![0]).toContain("recovered from backup");
  });

  it("treats corrupt primary and backup as corrupt-or-unreadable", async () => {
    const memory = makeStorage();
    memory.files[documentPath] = "bad";
    memory.files[backupPath] = "also bad";
    await expect(store(memory.storage).load())
      .rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("rejects a valid incompatible backup and a persisted foreign identity", async () => {
    const incompatible = makeStorage();
    incompatible.files[documentPath] = "corrupt";
    incompatible.files[backupPath] = `${JSON.stringify({ ...document(), scopeFingerprint: otherScope }, null, 2)}\n`;
    await expect(store(incompatible.storage).load())
      .rejects.toMatchObject({ code: "incompatible" });

    const foreign = makeStorage();
    foreign.files[documentPath] = `${JSON.stringify({ ...document(), identificationFingerprint: otherScope }, null, 2)}\n`;
    await expect(store(foreign.storage).load())
      .rejects.toMatchObject({ code: "incompatible" });

    const memory = makeStorage();
    await expect(store(memory.storage).replace({ ...document(), scopeFingerprint: otherScope }, 0))
      .rejects.toMatchObject({ code: "invalid" });
    expect(memory.files).toEqual({});
  });

  it("rejects invalid documents and expected revisions without writing", async () => {
    const memory = makeStorage();
    const suggestions = store(memory.storage);
    await expect(suggestions.replace(document({ suggestions: [
      newSuggestion([key(1)], "First."),
      attach("d-a", [key(1)], "Second."),
    ] }), 0)).rejects.toMatchObject({ code: "invalid" });
    await expect(suggestions.replace(document(), -1)).rejects.toMatchObject({ code: "invalid" });
    expect(memory.files).toEqual({});
  });

  it("fails closed without atomic write support", async () => {
    const memory = makeStorage(false);
    await expect(store(memory.storage).replace(document(), 0))
      .rejects.toMatchObject({ code: "atomic-write-unsupported" });
  });
});

describe("applyAttachSuggestion", () => {
  it("appends members at fixed confidence with a members-updated event and monotonic time", () => {
    const original = frozen(profile([
      direction("d-a", { members: [key(1), key(2)] }),
      direction("d-b", { members: [key(3)] }),
    ]));
    const result = applyAttachSuggestion({
      profile: original,
      suggestion: attach("d-a", [key(4), key(5)], "Papers continue the d-a theme."),
      now: secondTime,
    });
    expect(result).not.toBe(original);
    expect(original.directions[0]!.clusterMembers.map(({ paperKey }) => paperKey)).toEqual([key(1), key(2)]);
    const attached = result.directions.find(({ id }) => id === "d-a")!;
    expect(attached.clusterMembers).toEqual([
      { paperKey: key(1), confidence: 0.9 },
      { paperKey: key(2), confidence: 0.9 },
      { paperKey: key(4), confidence: SUGGESTION_MEMBER_CONFIDENCE },
      { paperKey: key(5), confidence: SUGGESTION_MEMBER_CONFIDENCE },
    ]);
    expect(attached.updatedAt).toBe(secondTime.toISOString());
    expect(attached.timeline).toEqual([
      { kind: "created", at: t0 },
      { kind: "members-updated", at: secondTime.toISOString() },
    ]);
    expect(result.directions.find(({ id }) => id === "d-b")!.clusterMembers).toEqual([
      { paperKey: key(3), confidence: 0.9 },
    ]);
  });

  it("allows attachments to a locked direction", () => {
    const result = applyAttachSuggestion({
      profile: profile([direction("d-lock", { members: [key(1)], locked: true })]),
      suggestion: attach("d-lock", [key(2)], "New papers continue the locked direction."),
      now: secondTime,
    });
    const attached = result.directions[0]!;
    expect(attached.clusterMembers.map(({ paperKey }) => paperKey)).toEqual([key(1), key(2)]);
    expect(attached.lockedAt).toBe(t0);
  });

  it("rejects papers already members of any direction", () => {
    const base = profile([
      direction("d-a", { members: [key(1), key(2)] }),
      direction("d-b", { members: [key(3)] }),
    ]);
    expect(() => applyAttachSuggestion({
      profile: base,
      suggestion: attach("d-a", [key(2)], "Duplicate within the target."),
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict", details: { paperKey: key(2) } }));
    expect(() => applyAttachSuggestion({
      profile: base,
      suggestion: attach("d-a", [key(3)], "Member of another direction."),
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict", details: { paperKey: key(3) } }));
  });

  it("rejects unknown, merged, and invalid-suggestion inputs", () => {
    const merged = profile([direction("d-a"), direction("d-m", { mergedIntoDirectionId: "d-a" })]);
    expect(() => applyAttachSuggestion({
      profile: profile([direction("d-a")]),
      suggestion: attach("d-ghost", [key(2)], "Phantom direction."),
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "not-found" }));
    expect(() => applyAttachSuggestion({
      profile: merged,
      suggestion: attach("d-m", [key(2)], "Merged direction."),
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => applyAttachSuggestion({
      profile: profile([direction("d-a")]),
      suggestion: newSuggestion([key(2)], "Wrong kind."),
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
  });
});

describe("buildNewDirectionDraft", () => {
  it("derives draft fields from a new suggestion", () => {
    const draft = buildNewDirectionDraft(newSuggestion(
      [key(1), key(2)],
      "A fresh research theme with no covering direction.",
    ));
    expect(draft).toEqual<NewDirectionDraft>({
      name: "A fresh research theme with no covering direction.",
      description: "A fresh research theme with no covering direction.",
      discoveryCues: [],
      representativePaperKeys: [key(1), key(2)],
      clusterMembers: [
        { paperKey: key(1), confidence: SUGGESTION_MEMBER_CONFIDENCE },
        { paperKey: key(2), confidence: SUGGESTION_MEMBER_CONFIDENCE },
      ],
    });
  });

  it("caps representatives at five and truncates an overlong name", () => {
    const paperKeys = [1, 2, 3, 4, 5, 6, 7].map(key);
    const reason = "x".repeat(200);
    const draft = buildNewDirectionDraft(newSuggestion(paperKeys, reason));
    expect(draft.representativePaperKeys).toEqual(paperKeys.slice(0, 5));
    expect(draft.clusterMembers).toHaveLength(7);
    expect(draft.name).toHaveLength(120);
    expect(draft.description).toHaveLength(200);
  });

  it("rejects non-new or non-canonical suggestions", () => {
    expect(() => buildNewDirectionDraft(attach("d-a", [key(1)], "Not a new suggestion.")))
      .toThrow(expect.objectContaining({ code: "invalid-input" }));
    expect(() => buildNewDirectionDraft({ kind: "new", paperKeys: [key(2), key(1)], reason: "Unsorted keys." }))
      .toThrow(expect.objectContaining({ code: "invalid-input" }));
  });
});

describe("applySplitSuggestion", () => {
  it("removes split members, records the split event, and creates a derived direction", () => {
    const original = frozen(profile([direction("d-a", {
      members: [key(1), key(2), key(3), key(4)],
      representatives: [representative(key(1), "e")],
    })]));
    const result = applySplitSuggestion({
      profile: original,
      suggestion: split("d-a", [key(1), key(3)], "Cluster overlaps d-a members but forms a distinct theme."),
      createId: (kind) => `${kind}.d-a.derived`,
      now: secondTime,
    });
    expect(result.profile).not.toBe(original);
    const source = result.profile.directions.find(({ id }) => id === "d-a")!;
    expect(source.clusterMembers.map(({ paperKey }) => paperKey)).toEqual([key(2), key(4)]);
    expect(source.timeline).toEqual([
      { kind: "created", at: t0 },
      { kind: "split", at: secondTime.toISOString(), sourceDirectionId: "d-a" },
    ]);
    expect(source.updatedAt).toBe(secondTime.toISOString());
    expect(result.newDirectionId).toBe("split.d-a.derived");
    const derived = result.profile.directions.find(({ id }) => id === "split.d-a.derived")!;
    expect(derived).toMatchObject({
      id: "split.d-a.derived",
      status: "active",
      name: "Cluster overlaps d-a members but forms a distinct theme.",
      description: "Cluster overlaps d-a members but forms a distinct theme.",
      discoveryCues: ["Cluster overlaps d-a members but forms a distinct theme."],
      clusterMembers: [
        { paperKey: key(1), confidence: SUGGESTION_MEMBER_CONFIDENCE },
        { paperKey: key(3), confidence: SUGGESTION_MEMBER_CONFIDENCE },
      ],
      lineage: { proposalIds: [SPLIT_DERIVED_PROPOSAL_MARKER], candidateIds: [], directionIds: [] },
      createdAt: secondTime.toISOString(),
      updatedAt: secondTime.toISOString(),
      timeline: [{ kind: "created", at: secondTime.toISOString() }],
    });
    // Real evidence is preserved for split papers that were representatives;
    // remaining representatives carry a deterministic placeholder fingerprint.
    expect(derived.representatives).toEqual([
      { paperKey: key(1), evidenceFingerprint: `sha256:${"e".repeat(64)}` },
      { paperKey: key(3), evidenceFingerprint: `sha256:${sha256Hex(key(3))}` },
    ]);
    expect(derived.representativeSetFingerprint)
      .toBe(createPersonalLibraryRepresentativeSetFingerprint(derived.representatives));
  });

  it("rejects locked, merged, unknown, and non-member splits", () => {
    expect(() => applySplitSuggestion({
      profile: profile([direction("d-lock", { members: [key(1)], locked: true })]),
      suggestion: split("d-lock", [key(1)], "Locked direction."),
      createId: () => "d-new",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => applySplitSuggestion({
      profile: profile([direction("d-a"), direction("d-m", { mergedIntoDirectionId: "d-a" })]),
      suggestion: split("d-m", [key(2)], "Merged direction."),
      createId: () => "d-new",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => applySplitSuggestion({
      profile: profile([direction("d-a", { members: [key(1)] })]),
      suggestion: split("d-ghost", [key(1)], "Phantom direction."),
      createId: () => "d-new",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "not-found" }));
    expect(() => applySplitSuggestion({
      profile: profile([direction("d-a", { members: [key(1)] })]),
      suggestion: split("d-a", [key(9)], "Not a member."),
      createId: () => "d-new",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
  });

  it("rejects invalid or colliding ids produced by createId", () => {
    const base = profile([direction("d-a", { members: [key(1)] }), direction("d-b", { members: [key(2)] })]);
    expect(() => applySplitSuggestion({
      profile: base,
      suggestion: split("d-a", [key(1)], "Split out."),
      createId: () => "not a valid id!",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
    expect(() => applySplitSuggestion({
      profile: base,
      suggestion: split("d-a", [key(1)], "Split out."),
      createId: () => "d-b",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
  });
});

describe("applyMergeSuggestion", () => {
  it("merges two terminal directions with review merge semantics and a derived target", () => {
    const original = frozen(profile([
      direction("d-a", { members: [key(1), key(2)], representatives: [representative(key(1), "e")] }),
      direction("d-b", { members: [key(3)], representatives: [representative(key(3), "f")] }),
    ]));
    const result = applyMergeSuggestion({
      profile: original,
      suggestion: merge(["d-a", "d-b"], "Both directions cover the same method."),
      createId: (kind) => `merged.${kind}`,
      now: secondTime,
    });
    expect(result).not.toBe(original);
    expect(result.directions.filter(({ status }) => status === "merged").map(({ id }) => id))
      .toEqual(["d-a", "d-b"]);
    expect(result.directions.find(({ id }) => id === "d-a")).toMatchObject({
      status: "merged", mergedIntoDirectionId: "merged.merge",
    });
    const target = result.directions.find(({ id }) => id === "merged.merge")!;
    expect(target).toMatchObject({
      id: "merged.merge",
      status: "active",
      name: "Both directions cover the same method.",
      description: "Both directions cover the same method.",
      discoveryCues: ["Both directions cover the same method."],
      clusterMembers: [],
      lineage: {
        proposalIds: ["proposal.d-a", "proposal.d-b"],
        candidateIds: ["candidate.d-a", "candidate.d-b"],
        directionIds: ["d-a", "d-b"],
      },
      createdAt: secondTime.toISOString(),
      updatedAt: secondTime.toISOString(),
      timeline: [
        { kind: "created", at: secondTime.toISOString() },
        { kind: "merged", at: secondTime.toISOString(), sourceDirectionIds: ["d-a", "d-b"] },
      ],
    });
    expect(target.representatives).toEqual([
      { paperKey: key(1), evidenceFingerprint: `sha256:${"e".repeat(64)}` },
      { paperKey: key(3), evidenceFingerprint: `sha256:${"f".repeat(64)}` },
    ]);
    expect(target.representativeSetFingerprint)
      .toBe(createPersonalLibraryRepresentativeSetFingerprint(target.representatives));
    expect(original.directions.every(({ status }) => status === "active")).toBe(true);
  });

  it("rejects locked, merged, and missing sources", () => {
    const lockedBase = profile([
      direction("d-a", { members: [key(1)], locked: true }),
      direction("d-b", { members: [key(2)] }),
    ]);
    expect(() => applyMergeSuggestion({
      profile: lockedBase,
      suggestion: merge(["d-a", "d-b"], "Merge."),
      createId: () => "d-c",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    const mergedBase = profile([
      direction("d-a"),
      direction("d-b", { mergedIntoDirectionId: "d-a" }),
    ]);
    expect(() => applyMergeSuggestion({
      profile: mergedBase,
      suggestion: merge(["d-b", "d-c"], "Merge."),
      createId: () => "d-x",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "not-found" }));
    expect(() => applyMergeSuggestion({
      profile: mergedBase,
      suggestion: merge(["d-a", "d-b"], "Merge."),
      createId: () => "d-x",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
  });

  it("rejects a colliding target id", () => {
    expect(() => applyMergeSuggestion({
      profile: profile([
        direction("d-a", { members: [key(1)] }),
        direction("d-b", { members: [key(2)] }),
      ]),
      suggestion: merge(["d-a", "d-b"], "Merge."),
      createId: () => "d-a",
      now: secondTime,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
  });
});
