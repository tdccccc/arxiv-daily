import { describe, expect, expectTypeOf, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import type { PaperMeta } from "../src/pipeline/arxiv-parser";
import {
  buildPaperFilterRequest,
  decodePaperFilterRecords,
  type FilterRecord,
} from "../src/pipeline/paper-filter";
import {
  DAILY_FILTER_CHECKPOINT_SCHEMA_VERSION,
  DAILY_FILTER_PROMPT_CONTRACT_VERSION,
  DAILY_FILTER_RESULT_CONTRACT_VERSION,
  DailyFilterCheckpointStore,
  buildDailyFilterCheckpointFingerprintInput,
  createDailyFilterCompatibilityFingerprint,
  deriveDailyFilterCheckpointPaths,
  prepareDailyFilterCheckpoint,
} from "../src/services/daily-filter-checkpoint-store";
import {
  DailyFilterCheckpointStore as ExportedDailyFilterCheckpointStore,
  buildDailyFilterCheckpointFingerprintInput as exportedBuildFingerprintInput,
  createDailyFilterCompatibilityFingerprint as exportedCreateFingerprint,
  type DailyFilterCheckpointCompatibilityInput,
  type DailyFilterCheckpointFingerprintInput,
} from "../src/index";
import { sha256ForCheckpointTests } from "../src/services/daily-summary-checkpoint-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic } from "../src/settings/topics";

const reportDate = "2026-08-01";
const documentPath = "arxiv-daily/.index/filter-checkpoints/2026-08-01.json";
const backupPath = `${documentPath}.bak`;
const papers: PaperMeta[] = [
  { id: "2608.00001", title: "First", authors: "A", abstract: "Abstract one" },
  { id: "2608.00002", title: "Second", authors: "B", abstract: "Abstract two" },
];
const result: FilterRecord[] = [
  { id: "2608.00002", category: "skip", directions: [], relevanceScore: 0 },
  { id: "2608.00001", category: "topic-a", directions: ["topic-a#1"], relevanceScore: 80 },
];

function compatibility(
  overrides: Partial<DailyFilterCheckpointCompatibilityInput> = {},
): DailyFilterCheckpointCompatibilityInput {
  return {
    papers,
    arxivSettings: {
      category: "astro-ph",
      categories: ["astro-ph", "cs.LG"],
      timezone: "UTC",
      topics: [
        normalizeTopic({ id: "unused-id", name: "Unused name", tag: "topic-a", directions: [{ id: "da", text: "Topic A", origin: "manual" }], detail: true }),
        normalizeTopic({ id: "unused-id-2", name: "Unused name 2", tag: "topic-b", directions: [{ id: "db", text: "Topic B", origin: "manual" }], detail: false }),
      ],
    },
    llm: {
      provider: "custom",
      baseUrl: "https://user:secret@example.test/private/v1?token=secret",
      model: "model-a",
      thinkingMode: false,
      reasoningEffort: "medium",
      apiKey: "never-persist",
    },
    ...overrides,
  };
}

function prepared(
  overrides: Partial<DailyFilterCheckpointCompatibilityInput> = {},
) {
  return prepareDailyFilterCheckpoint(compatibility(overrides));
}

function makeStorage(options: { rejectExistingRenameTarget?: boolean } = {}) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const readText = vi.fn(async (path: string) => {
    if (!(path in files)) throw new Error(`missing ${path}`);
    return files[path]!;
  });
  const exists = vi.fn(async (path: string) => path in files || dirs.has(path));
  const rename = vi.fn(async (from: string, to: string) => {
    if (!(from in files)) throw new Error(`missing ${from}`);
    if (options.rejectExistingRenameTarget && (to in files || dirs.has(to))) {
      throw new Error(`destination exists: ${to}`);
    }
    files[to] = files[from]!;
    delete files[from];
  });
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText,
    writeText: async (path, content) => { files[path] = content; },
    exists,
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { delete files[path]; dirs.delete(path); },
    rename,
  };
  return { files, storage, readText, exists, rename };
}

function makeStore(storage: StorageAdapter, warning = vi.fn()) {
  return new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output, {
    now: () => new Date("2026-08-01T12:00:00.000Z"),
    onWarning: warning,
  });
}

function recomputeDocumentFingerprint(document: any): void {
  document.fingerprint = `sha256:${sha256ForCheckpointTests(
    JSON.stringify(document.fingerprintInput),
  )}`;
}

describe("daily filter checkpoint public contract", () => {
  it("exports the store, fingerprint builders, and compatibility types from Core", () => {
    expect(ExportedDailyFilterCheckpointStore).toBe(DailyFilterCheckpointStore);
    expect(exportedBuildFingerprintInput).toBe(buildDailyFilterCheckpointFingerprintInput);
    expect(exportedCreateFingerprint).toBe(createDailyFilterCompatibilityFingerprint);
    expectTypeOf<DailyFilterCheckpointCompatibilityInput>()
      .not.toHaveProperty("temperature");
    expectTypeOf<DailyFilterCheckpointFingerprintInput["generation"]["mode"]>()
      .toMatchTypeOf<
        | { kind: "temperature"; temperature: number }
        | { kind: "anthropic-thinking"; budgetTokens: number }
        | { kind: "reasoning-thinking"; reasoningEffort: string }
      >();
  });
});

describe("paper filter shared contract", () => {
  it("builds the exact messages and call options used by filtering", () => {
    const input = compatibility();
    const request = buildPaperFilterRequest(input.papers, input.arxivSettings);
    expect(request.options).toEqual({ temperature: 0 });
    expect(request.messages).toHaveLength(2);
    expect(request.identity).toEqual({
      knownIds: ["2608.00001", "2608.00002"],
      validTags: ["topic-a", "topic-b"],
      directions: [
        { ref: "topic-a#1", tag: "topic-a", id: "da", text: "Topic A" },
        { ref: "topic-b#1", tag: "topic-b", id: "db", text: "Topic B" },
      ],
    });
    expect(request.messages[0]?.content).toContain("topic-a|topic-b|skip");
    expect(request.messages[1]?.content).toContain("ID: 2608.00001");
    expect(request.messages[1]?.content).not.toContain("authors");
  });

  it("strictly accepts ordered, omitted, and empty record lists", () => {
    const ids = new Set(papers.map((paper) => paper.id));
    const tags = new Set(["topic-a", "topic-b"]);
    const dirs = buildPaperFilterRequest(papers, compatibility().arxivSettings).identity.directions;
    expect(decodePaperFilterRecords({ papers: result }, ids, tags, dirs)).toEqual({ ok: true, value: result });
    expect(decodePaperFilterRecords({ papers: [] }, ids, tags, dirs)).toEqual({ ok: true, value: [] });
    expect(decodePaperFilterRecords({ papers: [result[1]] }, ids, tags, dirs)).toEqual({
      ok: true,
      value: [result[1]],
    });
  });

  it.each([
    {
      label: "extra root key", value: { papers: result, extra: true },
      reason: "root must be exactly {papers:[...]}",
    },
    {
      label: "extra record key", value: { papers: [{ ...result[0], extra: true }] },
      reason: "paper record has an invalid shape",
    },
    {
      label: "unknown id", value: { papers: [{ id: "unknown", category: "topic-a", directions: ["topic-a#1"], relevanceScore: 80 }] },
      reason: "paper record has an unknown id",
    },
    {
      label: "duplicate id", value: { papers: [result[0], result[0]] },
      reason: "paper record has a duplicate id",
    },
    {
      label: "unknown category", value: { papers: [{ id: papers[0]!.id, category: "unknown", directions: [], relevanceScore: 80 }] },
      reason: `paper ${papers[0]!.id} has an invalid category`,
    },
  ])("rejects $label for its original contract violation", ({ value, reason }) => {
    expect(decodePaperFilterRecords(
      value,
      new Set(papers.map((paper) => paper.id)),
      new Set(["topic-a"]),
      buildPaperFilterRequest(papers, compatibility().arxivSettings).identity.directions,
    )).toEqual({ ok: false, reason });
  });
});

describe("daily filter checkpoint fingerprint", () => {
  it("stores exact rendered messages and effective identity without secrets", () => {
    const input = compatibility();
    const canonical = buildDailyFilterCheckpointFingerprintInput(input);
    const request = buildPaperFilterRequest(papers, input.arxivSettings);
    expect(canonical.request).toEqual({
      messages: request.messages,
      identity: request.identity,
    });
    expect(canonical.generation).toMatchObject({
      provider: "custom",
      model: "model-a",
      mode: { kind: "temperature", temperature: 0 },
    });
    const raw = JSON.stringify(canonical);
    expect(raw).not.toContain("never-persist");
    expect(raw).not.toContain("example.test");
    expect(raw).not.toContain("token=secret");
  });

  it("derives temperature identity only from the shared live request options", () => {
    const input = compatibility();
    const request = buildPaperFilterRequest(input.papers, input.arxivSettings);
    const canonical = buildDailyFilterCheckpointFingerprintInput(input);
    expect(canonical.generation.mode).toEqual({
      kind: "temperature",
      temperature: request.options.temperature,
    });

    // Runtime callers can carry unknown properties, but they cannot create a
    // checkpoint-only override or a fingerprint distinct from the live call.
    const withLegacyOverride = { ...input, temperature: 0.9 };
    expect(buildDailyFilterCheckpointFingerprintInput(withLegacyOverride).generation.mode)
      .toEqual({ kind: "temperature", temperature: request.options.temperature });
    expect(createDailyFilterCompatibilityFingerprint(withLegacyOverride))
      .toBe(createDailyFilterCompatibilityFingerprint(input));
  });

  it.each([
    ["paper order", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, papers: [...input.papers].reverse() })],
    ["title", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, papers: input.papers.map((paper, i) => i ? paper : { ...paper, title: "changed" }) })],
    ["abstract", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, papers: input.papers.map((paper, i) => i ? paper : { ...paper, abstract: "changed" }) })],
    ["category rendering", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, arxivSettings: { ...input.arxivSettings, categories: ["astro-ph"] } })],
    ["topic order", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, arxivSettings: { ...input.arxivSettings, topics: [...input.arxivSettings.topics].reverse() } })],
    ["topic direction text", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, arxivSettings: { ...input.arxivSettings, topics: input.arxivSettings.topics.map((topic, i) => i ? topic : normalizeTopic({ ...topic, directions: [{ ...topic.directions[0]!, text: "changed" }] })) } })],
    ["provider", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, provider: "openai" } })],
    ["endpoint", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, baseUrl: "https://other.test/v1" } })],
    ["model", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, model: "model-b" } })],
    ["mode", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, thinkingMode: true } })],
    ["prompt contract", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, promptContractVersion: DAILY_FILTER_PROMPT_CONTRACT_VERSION + 1 })],
    ["result contract", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, resultContractVersion: DAILY_FILTER_RESULT_CONTRACT_VERSION + 1 })],
  ])("invalidates on %s changes", (_name, mutate) => {
    const input = compatibility();
    expect(createDailyFilterCompatibilityFingerprint(mutate(input))).not.toBe(createDailyFilterCompatibilityFingerprint(input));
  });

  it.each([
    ["API key", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, apiKey: "changed" } })],
    ["authors", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, papers: input.papers.map((paper) => ({ ...paper, authors: "changed" })) })],
    ["topic id/name/detail", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, arxivSettings: { ...input.arxivSettings, topics: input.arxivSettings.topics.map((topic) => ({ ...topic, id: "changed", name: "changed", detail: !topic.detail })) } })],
    ["timezone", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, arxivSettings: { ...input.arxivSettings, timezone: "Asia/Shanghai" } })],
  ])("does not invalidate on non-rendered %s changes", (_name, mutate) => {
    const input = compatibility();
    expect(createDailyFilterCompatibilityFingerprint(mutate(input))).toBe(createDailyFilterCompatibilityFingerprint(input));
  });
});

describe("scored daily filter checkpoints", () => {
  it("persists and reconstructs every relevance score without rounding or reordering", async () => {
    const { files, storage } = makeStorage();
    const records = result.map((record) => ({
      ...record,
      relevanceScore: record.category === "skip" ? 0 : 87.5,
    }));
    await makeStore(storage).save(reportDate, prepared(), records);
    expect(JSON.parse(files[documentPath]!).result).toEqual(records);
    const reconstructed = makeStore(storage);
    expect(await reconstructed.load(reportDate)).toMatchObject({ result: records });
    expect(await reconstructed.lookupReusable(reportDate, prepared())).toEqual(records);
  });

  // Keep each old contract at literal 2 independently. Valid current records
  // isolate version rejection from the separate missing-score shape check.
  it.each([
    ["prompt", { promptContractVersion: 2 }],
    ["result", { resultContractVersion: 2 }],
  ])("rejects the old %s contract 2 independently", async (_name, stale) => {
    const { files, storage } = makeStorage();
    await makeStore(storage).save(reportDate, prepared(), result);
    const document = JSON.parse(files[documentPath]!);
    Object.assign(document.fingerprintInput, stale);
    recomputeDocumentFingerprint(document);
    files[documentPath] = JSON.stringify(document);

    expect(await makeStore(storage).load(reportDate)).toBeNull();
    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toBeNull();
  });

  it("refuses to save a current-contract record with a missing relevance score", async () => {
    const { files, storage } = makeStorage();
    await expect(makeStore(storage).save(reportDate, prepared(), [{
      id: "2608.00001",
      category: "topic-a",
      directions: ["topic-a#1"],
    }])).rejects.toThrow(/invalid daily filter checkpoint result: paper record has an invalid shape/);
    expect(files[documentPath]).toBeUndefined();
  });

  it("refuses to load a current-contract record with a missing relevance score", async () => {
    const { files, storage } = makeStorage();
    await makeStore(storage).save(reportDate, prepared(), result);
    const document = JSON.parse(files[documentPath]!);
    delete document.result[1].relevanceScore;
    files[documentPath] = JSON.stringify(document);

    expect(await makeStore(storage).load(reportDate)).toBeNull();
    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toBeNull();
  });
});

describe("DailyFilterCheckpointStore", () => {
  it("round-trips adaptive Anthropic results", async () => {
    const { storage } = makeStorage();
    const store = makeStore(storage);
    const llm = { ...compatibility().llm, provider: "anthropic", baseUrl: "https://api.anthropic.com/v1", model: "claude-opus-4-7", thinkingMode: true };
    const plain = prepared({ llm });
    await store.save(reportDate, plain, result);
    const reconstructed = makeStore(storage);
    expect(await reconstructed.lookupReusable(reportDate, plain)).toEqual(result);
  });

  it("does not reuse filter results from the old provider request contract", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage);
    const file = store.pathsFor(reportDate).documentPath;
    await store.save(reportDate, prepared(), result);
    const document = JSON.parse(files[file]!);
    delete document.fingerprintInput.generation.requestContractVersion;
    document.fingerprint = `sha256:${sha256ForCheckpointTests(JSON.stringify(document.fingerprintInput))}`;
    files[file] = JSON.stringify(document);
    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toBeNull();
  });

  it("derives its independent date-scoped path and validates dates", () => {
    expect(deriveDailyFilterCheckpointPaths({ normalizePath: (path) => path }, DEFAULT_SETTINGS.output, reportDate)).toEqual({
      directory: "arxiv-daily/.index/filter-checkpoints", documentPath, backupPath,
      personalizedDocumentPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.personalized.json",
      personalizedBackupPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.personalized.json.bak",
      legacyNoveltyDocumentPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.novelty.json",
      legacyNoveltyBackupPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.novelty.json.bak",
    });
    expect(() => deriveDailyFilterCheckpointPaths({ normalizePath: (path) => path }, DEFAULT_SETTINGS.output, "2026-02-30")).toThrow(/invalid checkpoint report date/);
  });

  it("rejects caller-constructed objects at the store boundary", async () => {
    const { storage } = makeStorage();
    const arbitrary = {
      request: buildPaperFilterRequest(papers, compatibility().arxivSettings),
      fingerprintInput: buildDailyFilterCheckpointFingerprintInput(compatibility()),
    };
    await expect(
      makeStore(storage).save(reportDate, arbitrary as any, result),
    ).rejects.toThrow(/prepared exact request snapshot/);
  });

  it.each([[result], [[] as FilterRecord[]]])("strictly saves and reuses ordered result %j", async (records) => {
    const { files, storage } = makeStorage();
    await makeStore(storage).save(reportDate, prepared(), records);
    expect(JSON.parse(files[documentPath]!)).toMatchObject({
      schemaVersion: DAILY_FILTER_CHECKPOINT_SCHEMA_VERSION,
      reportDate,
      result: records,
    });
    expect(files[documentPath]).not.toContain("never-persist");
    expect(files[documentPath]).not.toContain("example.test");
    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toEqual(records);
  });

  it("round-trips after reconstruction when a valid tag contains a pipe", async () => {
    const { files, storage } = makeStorage();
    const input = compatibility({
      arxivSettings: {
        ...compatibility().arxivSettings,
        topics: [normalizeTopic({
          id: "pipe-tag",
          name: "NLP and LLM",
          tag: "nlp|llm",
          directions: [{ id: "dp", text: "NLP and language models", origin: "manual" }],
          detail: false,
        })],
      },
    });
    const records = [{ id: papers[0]!.id, category: "nlp|llm", directions: ["nlp|llm#1"], relevanceScore: 80 }];

    await makeStore(storage).save(
      reportDate,
      prepareDailyFilterCheckpoint(input),
      records,
    );

    const persisted = JSON.parse(files[documentPath]!);
    expect(persisted.fingerprintInput.request.identity).toEqual({
      knownIds: ["2608.00001", "2608.00002"],
      validTags: ["nlp|llm"],
      // Splitting the ref on its final `#` recovers a tag that contains one.
      directions: [{ ref: "nlp|llm#1", tag: "nlp|llm", id: "dp", text: "NLP and language models" }],
    });
    expect(persisted.fingerprintInput.request.messages[0].content)
      .toContain("nlp|llm|skip");
    expect(await makeStore(storage).load(reportDate)).toMatchObject({ result: records });
    expect(await makeStore(storage).lookupReusable(reportDate, prepareDailyFilterCheckpoint(input))).toEqual(records);
  });

  /**
   * P3 moved the filter from a topic's one-line description to its directions,
   * which invalidates every checkpoint written by the old contract. Two things
   * this case has to get right, both learned by getting them wrong first:
   *
   * - One case per version. With both stale versions in a single case,
   *   reverting either bump alone still leaves it green, and it then pins only
   *   "at least one of them changed".
   * - The stale value is the literal 1 the old contract shipped with, not
   *   `CURRENT - 1`. Written relative to the constant, the case follows the
   *   constant back down on a revert and can never fail.
   */
  it.each([
    ["prompt", { promptContractVersion: 1 }],
    ["result", { resultContractVersion: 1 }],
  ])("never reuses a checkpoint written under the shipped %s contract 1", async (_name, stale) => {
    const { files, storage } = makeStorage();
    await makeStore(storage).save(reportDate, prepared(), result);
    const document = JSON.parse(files[documentPath]!);
    Object.assign(document.fingerprintInput, stale);
    recomputeDocumentFingerprint(document);
    files[documentPath] = JSON.stringify(document);

    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toBeNull();
  });

  it("rejects invalid result and unsupported contracts", async () => {
    const { storage } = makeStorage();
    await expect(makeStore(storage).save(reportDate, prepared(), [{ id: "unknown", category: "topic-a", directions: ["topic-a#1"], relevanceScore: 80 }])).rejects.toThrow(/paper record has an unknown id/);
    await expect(makeStore(storage).save(reportDate, prepared({ promptContractVersion: DAILY_FILTER_PROMPT_CONTRACT_VERSION + 1 }), result)).rejects.toThrow(/unsupported/);
  });

  it("does not reuse classifications made before cross-topic specificity was defined", async () => {
    const { files, storage } = makeStorage();
    await makeStore(storage).save(reportDate, prepared(), result);
    const document = JSON.parse(files[documentPath]!);
    document.fingerprintInput.promptContractVersion = 3;
    recomputeDocumentFingerprint(document);
    files[documentPath] = JSON.stringify(document);

    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toBeNull();
  });

  it.each([
    ["extra document key", (document: any) => { document.extra = true; }],
    ["tampered fingerprint", (document: any) => { document.fingerprint = `sha256:${"0".repeat(64)}`; }],
    ["extra result key", (document: any) => { document.result[0].raw = "provider response"; }],
    ["unknown result id", (document: any) => { document.result[0].id = "2608.99999"; }],
    ["duplicate result id", (document: any) => { document.result[1].id = document.result[0].id; }],
  ])("strict load rejects %s", async (_name, mutate) => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    const document = JSON.parse(files[documentPath]!);
    mutate(document);
    files[documentPath] = JSON.stringify(document);
    expect(await store.load(reportDate)).toBeNull();
    expect(await store.lookupReusable(reportDate, prepared())).toBeNull();
  });

  it.each([
    ["missing known result id", (document: any) => {
      document.fingerprintInput.request.identity.knownIds = ["2608.00002"];
    }],
    ["missing valid result tag", (document: any) => {
      document.fingerprintInput.request.identity.validTags = ["topic-b"];
    }],
    ["non-string known id", (document: any) => {
      document.fingerprintInput.request.identity.knownIds = [260800001];
    }],
    ["extra identity key", (document: any) => {
      document.fingerprintInput.request.identity.rawPrompt = "not allowed";
    }],
  ])("strict load rejects recomputed %s identity tampering", async (_name, mutate) => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    const document = JSON.parse(files[documentPath]!);
    mutate(document);
    recomputeDocumentFingerprint(document);
    files[documentPath] = JSON.stringify(document);
    expect(await store.load(reportDate)).toBeNull();
    expect(await store.lookupReusable(reportDate, prepared())).toBeNull();
  });

  it("changed structured identity participates in compatibility fingerprint", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    const document = JSON.parse(files[documentPath]!);
    document.fingerprintInput.request.identity.validTags.push("tampered-tag");
    recomputeDocumentFingerprint(document);
    files[documentPath] = JSON.stringify(document);

    expect(await store.load(reportDate)).toMatchObject({ result });
    expect(await store.lookupReusable(reportDate, prepared())).toBeNull();
  });

  it("treats corrupt primary as miss and recovers valid backup", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    files[backupPath] = files[documentPath]!;
    files[documentPath] = "corrupt";
    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toEqual(result);
    delete files[backupPath];
    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toBeNull();
  });

  it.each([
    ["invalid JSON primary with no backup", "{not-json", undefined],
    ["invalid schema primary with no backup", JSON.stringify({ schemaVersion: 999 }), undefined],
    ["corrupt primary and corrupt backup", "{bad-primary", "{bad-backup"],
  ])("replaces readable-corrupt state: %s", async (_name, primary, backup) => {
    const { files, storage } = makeStorage();
    files[documentPath] = primary;
    if (backup !== undefined) files[backupPath] = backup;
    const store = makeStore(storage);

    expect(await store.lookupReusable(reportDate, prepared())).toBeNull();
    await store.save(reportDate, prepared(), result);

    expect(await store.lookupReusable(reportDate, prepared())).toEqual(result);
    expect(files[documentPath]).not.toContain("bad-primary");
    if (files[backupPath]) expect(files[backupPath]).not.toContain("bad-backup");
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
    expect(files[`${backupPath}.tmp`]).toBeUndefined();
  });

  it("fails lookup closed on primary EIO without consulting a valid backup", async () => {
    const { files, storage, readText } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    files[backupPath] = files[documentPath]!;
    readText.mockClear();
    readText.mockRejectedValueOnce(Object.assign(new Error("EIO"), { code: "EIO" }));

    await expect(store.lookupReusable(reportDate, prepared())).rejects.toThrow(
      /cannot read daily filter checkpoint/,
    );
    expect(readText).toHaveBeenCalledTimes(1);
  });

  it("fails mutation closed on primary EIO even with a valid backup", async () => {
    const { files, storage, readText } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    files[backupPath] = files[documentPath]!;
    const primary = files[documentPath]!;
    readText.mockRejectedValueOnce(Object.assign(new Error("EIO"), { code: "EIO" }));
    await expect(store.save(reportDate, prepared(), [])).rejects.toThrow(/cannot mutate unreadable/);
    expect(files[documentPath]).toBe(primary);
  });

  it("permits replacement from a valid backup when primary is corrupt", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    files[backupPath] = files[documentPath]!;
    files[documentPath] = "corrupt";
    await store.save(reportDate, prepared(), []);
    expect(await store.lookupReusable(reportDate, prepared())).toEqual([]);
  });

  it("rotates backups and cleans temp artifacts", async () => {
    const { files, storage } = makeStorage({ rejectExistingRenameTarget: true });
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    const first = files[documentPath]!;
    await store.save(reportDate, prepared(), []);
    expect(files[backupPath]).toBe(first);
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
    expect(files[`${backupPath}.tmp`]).toBeUndefined();
  });

  it("keeps primary valid when backup publication fails", async () => {
    const { files, storage, rename } = makeStorage({ rejectExistingRenameTarget: true });
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    const primary = files[documentPath]!;
    files[backupPath] = primary;
    rename.mockImplementationOnce(async () => { throw new Error("injected backup failure"); });
    await expect(store.save(reportDate, prepared(), [])).rejects.toThrow(/failed to save/);
    expect(files[documentPath]).toBe(primary);
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
    expect(files[`${backupPath}.tmp`]).toBeUndefined();
  });

  it("serializes same-path saves across store instances", async () => {
    const { storage } = makeStorage();
    await Promise.all([
      makeStore(storage).save(reportDate, prepared(), result),
      makeStore(storage).save(reportDate, prepared(), []),
    ]);
    expect(await makeStore(storage).lookupReusable(reportDate, prepared())).toEqual([]);
  });

  it("restores the previous primary after promotion failure", async () => {
    const { files, storage, rename } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    const primary = files[documentPath]!;
    rename.mockImplementationOnce(async (from, to) => {
      files[to] = files[from]!;
      delete files[from];
    });
    rename.mockImplementationOnce(async () => { throw new Error("injected promotion failure"); });
    await expect(store.save(reportDate, prepared(), [])).rejects.toThrow(/failed to save/);
    expect(files[documentPath]).toBe(primary);
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
  });

  it("removes manual and personalized primary, backup, and temp artifacts", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    const paths = store.pathsFor(reportDate);
    // The retired library-profile classifier left these behind on real vaults;
    // nothing writes them any more, so seed them the way disk would have.
    files[paths.personalizedDocumentPath] = "{}";
    files[backupPath] = files[documentPath]!;
    files[paths.personalizedBackupPath] = files[paths.personalizedDocumentPath]!;
    for (const path of [documentPath, backupPath, paths.personalizedDocumentPath,
      paths.personalizedBackupPath]) files[`${path}.tmp`] = "tmp";
    await store.removeAll(reportDate);
    expect(Object.keys(files)).toEqual([]);
  });

  it("removes orphaned legacy novelty checkpoint files (withdrawn stage) and leaves unrelated files", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    // A run that crashed before commit while the novelty stage existed left these behind.
    files[paths.legacyNoveltyDocumentPath] = "{}";
    files[paths.legacyNoveltyBackupPath] = "{}";
    files[`${paths.legacyNoveltyDocumentPath}.tmp`] = "tmp";
    files[`${paths.legacyNoveltyBackupPath}.tmp`] = "tmp";
    const otherDay = "arxiv-daily/.index/filter-checkpoints/2026-08-02.novelty.json";
    files[otherDay] = "{}";
    await store.removeAll(reportDate);
    expect(Object.keys(files)).toEqual([otherDay]);
    // Cleaning when nothing is there is a no-op, not an error.
    await expect(store.removeAll(reportDate)).resolves.toBeUndefined();
  });
});
