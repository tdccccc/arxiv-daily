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
  DailyFilterCheckpointStore,
  buildDailyFilterCheckpointFingerprintInput,
  createDailyFilterCompatibilityFingerprint,
  deriveDailyFilterCheckpointPaths,
  prepareDailyFilterCheckpoint,
  PERSONALIZED_FILTER_CHECKPOINT_SCHEMA_VERSION,
  NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION,
} from "../src/services/daily-filter-checkpoint-store";
import {
  planPersonalizedFilterCalls,
  preparePersonalizedFilterCheckpoint,
  type PersonalizedDiscoveryInput,
  type PersonalizedDirectionRecord,
} from "../src/pipeline/personalized-paper-filter";
import {
  PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION,
  PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION,
  decodeNoveltyCheckpointRecords,
  decodeNoveltyFingerprintInput,
  fingerprintNoveltyCheckpointInput,
  isPreparedNoveltyCheckpoint,
  planPersonalNoveltyCalls,
  prepareNoveltyCheckpoint,
  type NoveltyCheckpointRecord,
  type NoveltyDailyPaper,
  type NoveltyRepresentativePaper,
  type PersonalNoveltyMatchInput,
  type PersonalNoveltyPaperOutcome,
  type PersonalizedNoveltyInput,
} from "../src/pipeline/personalized-novelty";
import {
  DailyFilterCheckpointStore as ExportedDailyFilterCheckpointStore,
  buildDailyFilterCheckpointFingerprintInput as exportedBuildFingerprintInput,
  createDailyFilterCompatibilityFingerprint as exportedCreateFingerprint,
  type DailyFilterCheckpointCompatibilityInput,
  type DailyFilterCheckpointFingerprintInput,
} from "../src/index";
import { sha256ForCheckpointTests } from "../src/services/daily-summary-checkpoint-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const reportDate = "2026-08-01";
const documentPath = "arxiv-daily/.index/filter-checkpoints/2026-08-01.json";
const backupPath = `${documentPath}.bak`;
const papers: PaperMeta[] = [
  { id: "2608.00001", title: "First", authors: "A", abstract: "Abstract one" },
  { id: "2608.00002", title: "Second", authors: "B", abstract: "Abstract two" },
];
const result: FilterRecord[] = [
  { id: "2608.00002", category: "skip" },
  { id: "2608.00001", category: "topic-a" },
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
        { id: "unused-id", name: "Unused name", tag: "topic-a", description: "Topic A", detail: true },
        { id: "unused-id-2", name: "Unused name 2", tag: "topic-b", description: "Topic B", detail: false },
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

const personalizedDiscovery: PersonalizedDiscoveryInput = {
  directions: [{
    id: "direction.001",
    name: "Direction one",
    description: "Strict personalized direction",
    discoveryCues: ["strict discovery"],
    representatives: [{
      paperKey: "arxiv:2501.00001",
      title: "Representative one",
      evidenceDepth: "metadata-and-abstract",
    }],
  }],
};

function personalizedPrepared() {
  const planned = planPersonalizedFilterCalls(papers, personalizedDiscovery);
  if (!planned.ok) throw new Error("unexpected plan-too-large");
  return preparePersonalizedFilterCheckpoint({ plan: planned.value, llm: compatibility().llm as any });
}

const personalizedResult: PersonalizedDirectionRecord[] = [
  { paperKey: "arxiv:2608.00001", directionIds: ["direction.001"] },
  { paperKey: "arxiv:2608.00002", directionIds: [] },
];

function recomputeDocumentFingerprint(document: any): void {
  document.fingerprint = `sha256:${sha256ForCheckpointTests(
    JSON.stringify(document.fingerprintInput),
  )}`;
}

function noveltyDailyPaper(index: number): NoveltyDailyPaper {
  return {
    paperKey: `arxiv:2608.${String(index).padStart(5, "0")}`,
    title: `New paper ${index}`,
    abstract: `Abstract ${index}`,
  };
}

function noveltyRepresentative(
  index: number,
  overrides: Partial<NoveltyRepresentativePaper> = {},
): NoveltyRepresentativePaper {
  return {
    paperKey: `arxiv:2501.${String(index).padStart(5, "0")}`,
    title: `Prior paper ${index}`,
    authors: [`Author ${index}`],
    abstract: `Prior abstract ${index}`,
    published: "2026-08-01T00:00:00.000Z",
    categories: ["cs.AI", "cs.LG"],
    ...overrides,
  };
}

function noveltyInput(
  overrides: Partial<PersonalizedNoveltyInput> = {},
): PersonalizedNoveltyInput {
  return {
    papers: [noveltyDailyPaper(1)],
    representatives: [noveltyRepresentative(1), noveltyRepresentative(2), noveltyRepresentative(3)],
    ...overrides,
  };
}

function noveltyMatches(
  overrides: Partial<PersonalNoveltyMatchInput> = {},
): PersonalNoveltyMatchInput {
  return {
    paperMatches: [{
      paperKey: "arxiv:2608.00001",
      directionIds: ["direction.001", "direction.002"],
    }],
    directionRepresentatives: [
      { directionId: "direction.001", representativePaperKeys: ["arxiv:2501.00001"] },
      { directionId: "direction.002", representativePaperKeys: ["arxiv:2501.00002", "arxiv:2501.00003"] },
    ],
    ...overrides,
  };
}

function noveltyPrepared(overrides: {
  input?: Partial<PersonalizedNoveltyInput>;
  matches?: Partial<PersonalNoveltyMatchInput>;
  llm?: Record<string, unknown>;
  promptContractVersion?: number;
  resultContractVersion?: number;
} = {}) {
  const input = noveltyInput(overrides.input);
  const matches = noveltyMatches(overrides.matches);
  const planned = planPersonalNoveltyCalls(input, matches);
  if (!planned.ok) throw new Error("unexpected novelty plan-too-large");
  return prepareNoveltyCheckpoint({
    plan: planned.value,
    matches,
    llm: (overrides.llm ?? compatibility().llm) as any,
    promptContractVersion: overrides.promptContractVersion,
    resultContractVersion: overrides.resultContractVersion,
  });
}

const noveltyRecord: NoveltyCheckpointRecord = {
  paperKey: "arxiv:2608.00001",
  status: "novelty",
  novelty: {
    differenceType: "new-method",
    comparisonBasis: ["arxiv:2501.00001"],
    evidenceDepth: "metadata-and-abstract",
    explanation: "Introduces a method absent from the representative abstracts.",
  },
};

const noveltyOutcome: PersonalNoveltyPaperOutcome = {
  paperKey: "arxiv:2608.00001",
  status: "novelty",
  novelty: noveltyRecord.novelty,
};

const noNoveltyOutcome: PersonalNoveltyPaperOutcome = {
  paperKey: "arxiv:2608.00001",
  status: "no-novelty",
  reason: "validation-exhausted",
};

const secondPaperNoNoveltyOutcome: PersonalNoveltyPaperOutcome = {
  paperKey: "arxiv:2608.00002",
  status: "no-novelty",
  reason: "validation-exhausted",
};

function noveltyInputTwoPapers(): PersonalizedNoveltyInput {
  return {
    papers: [noveltyDailyPaper(1), noveltyDailyPaper(2)],
    representatives: [noveltyRepresentative(1), noveltyRepresentative(2)],
  };
}

function noveltyMatchesTwoPapers(): PersonalNoveltyMatchInput {
  return {
    paperMatches: [
      { paperKey: "arxiv:2608.00001", directionIds: ["direction.001"] },
      { paperKey: "arxiv:2608.00002", directionIds: ["direction.002"] },
    ],
    directionRepresentatives: [
      { directionId: "direction.001", representativePaperKeys: ["arxiv:2501.00001"] },
      { directionId: "direction.002", representativePaperKeys: ["arxiv:2501.00002"] },
    ],
  };
}

function noveltyPreparedTwoPapers(overrides: {
  llm?: Record<string, unknown>;
  promptContractVersion?: number;
  resultContractVersion?: number;
} = {}) {
  const planned = planPersonalNoveltyCalls(noveltyInputTwoPapers(), noveltyMatchesTwoPapers());
  if (!planned.ok) throw new Error("unexpected novelty plan-too-large");
  return prepareNoveltyCheckpoint({
    plan: planned.value,
    matches: noveltyMatchesTwoPapers(),
    llm: (overrides.llm ?? compatibility().llm) as any,
    promptContractVersion: overrides.promptContractVersion,
    resultContractVersion: overrides.resultContractVersion,
  });
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
    });
    expect(request.messages[0]?.content).toContain("topic-a|topic-b|skip");
    expect(request.messages[1]?.content).toContain("ID: 2608.00001");
    expect(request.messages[1]?.content).not.toContain("authors");
  });

  it("strictly accepts ordered, omitted, and empty record lists", () => {
    const ids = new Set(papers.map((paper) => paper.id));
    const tags = new Set(["topic-a", "topic-b"]);
    expect(decodePaperFilterRecords({ papers: result }, ids, tags)).toEqual({ ok: true, value: result });
    expect(decodePaperFilterRecords({ papers: [] }, ids, tags)).toEqual({ ok: true, value: [] });
    expect(decodePaperFilterRecords({ papers: [result[1]] }, ids, tags)).toEqual({
      ok: true,
      value: [result[1]],
    });
  });

  it.each([
    { papers: result, extra: true },
    { papers: [{ ...result[0], extra: true }] },
    { papers: [{ id: "unknown", category: "topic-a" }] },
    { papers: [result[0], result[0]] },
    { papers: [{ id: papers[0]!.id, category: "unknown" }] },
  ])("rejects malformed records %#", (value) => {
    expect(decodePaperFilterRecords(
      value,
      new Set(papers.map((paper) => paper.id)),
      new Set(["topic-a"]),
    )).toMatchObject({ ok: false });
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
    ["topic description", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, arxivSettings: { ...input.arxivSettings, topics: input.arxivSettings.topics.map((topic, i) => i ? topic : { ...topic, description: "changed" }) } })],
    ["provider", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, provider: "openai" } })],
    ["endpoint", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, baseUrl: "https://other.test/v1" } })],
    ["model", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, model: "model-b" } })],
    ["mode", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, llm: { ...input.llm, thinkingMode: true } })],
    ["prompt contract", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, promptContractVersion: 2 })],
    ["result contract", (input: DailyFilterCheckpointCompatibilityInput) => ({ ...input, resultContractVersion: 2 })],
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

describe("DailyFilterCheckpointStore", () => {
  it("derives its independent date-scoped path and validates dates", () => {
    expect(deriveDailyFilterCheckpointPaths({ normalizePath: (path) => path }, DEFAULT_SETTINGS.output, reportDate)).toEqual({
      directory: "arxiv-daily/.index/filter-checkpoints", documentPath, backupPath,
      personalizedDocumentPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.personalized.json",
      personalizedBackupPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.personalized.json.bak",
      noveltyDocumentPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.novelty.json",
      noveltyBackupPath: "arxiv-daily/.index/filter-checkpoints/2026-08-01.novelty.json.bak",
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
        topics: [{
          id: "pipe-tag",
          name: "NLP and LLM",
          tag: "nlp|llm",
          description: "NLP and language models",
          detail: false,
        }],
      },
    });
    const records = [{ id: papers[0]!.id, category: "nlp|llm" }];

    await makeStore(storage).save(
      reportDate,
      prepareDailyFilterCheckpoint(input),
      records,
    );

    const persisted = JSON.parse(files[documentPath]!);
    expect(persisted.fingerprintInput.request.identity).toEqual({
      knownIds: ["2608.00001", "2608.00002"],
      validTags: ["nlp|llm"],
    });
    expect(persisted.fingerprintInput.request.messages[0].content)
      .toContain("nlp|llm|skip");
    expect(await makeStore(storage).load(reportDate)).toMatchObject({ result: records });
    expect(await makeStore(storage).lookupReusable(reportDate, prepareDailyFilterCheckpoint(input))).toEqual(records);
  });

  it("rejects invalid result and unsupported contracts", async () => {
    const { storage } = makeStorage();
    await expect(makeStore(storage).save(reportDate, prepared(), [{ id: "unknown", category: "topic-a" }])).rejects.toThrow(/invalid daily filter/);
    await expect(makeStore(storage).save(reportDate, prepared({ promptContractVersion: DAILY_FILTER_PROMPT_CONTRACT_VERSION + 1 }), result)).rejects.toThrow(/unsupported/);
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

  it("persists and reconstructs a strict independent personalized checkpoint privately", async () => {
    const { files, storage } = makeStorage();
    const writeTextWithMode = vi.fn(async (path: string, content: string) => {
      files[path] = content;
    });
    storage.writeTextWithMode = writeTextWithMode;
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);

    await store.savePersonalized(reportDate, personalizedPrepared(), personalizedResult);

    expect(JSON.parse(files[paths.personalizedDocumentPath]!)).toMatchObject({
      schemaVersion: PERSONALIZED_FILTER_CHECKPOINT_SCHEMA_VERSION,
      reportDate,
      result: personalizedResult,
    });
    expect(writeTextWithMode).toHaveBeenCalledWith(
      expect.stringContaining(".personalized.json.tmp"), expect.any(String), 0o600,
    );
    expect(await makeStore(storage).lookupPersonalizedReusable(
      reportDate, personalizedPrepared(),
    )).toEqual(personalizedResult);
  });

  it("rejects arbitrary snapshots, unknown/duplicate/partial results, and fingerprint tampering", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const snapshot = personalizedPrepared();
    await expect(store.savePersonalized(
      reportDate, JSON.parse(JSON.stringify(snapshot)) as any, personalizedResult,
    ))
      .rejects.toThrow(/prepared exact call-plan snapshot/);
    for (const invalid of [
      personalizedResult.slice(0, 1),
      [{ paperKey: "arxiv:2608.99999", directionIds: [] }, personalizedResult[1]],
      [{ paperKey: personalizedResult[0]!.paperKey, directionIds: ["unknown"] }, personalizedResult[1]],
      [{ ...personalizedResult[0], extra: true }, personalizedResult[1]],
    ]) {
      await expect(store.savePersonalized(reportDate, snapshot, invalid))
        .rejects.toThrow(/invalid personalized filter checkpoint/);
    }
    await store.savePersonalized(reportDate, snapshot, personalizedResult);
    const paths = store.pathsFor(reportDate);
    const document = JSON.parse(files[paths.personalizedDocumentPath]!);
    document.fingerprintInput.plan.batches[0].request.messages[1].content = "tampered";
    document.fingerprint = `sha256:${sha256ForCheckpointTests(JSON.stringify(document.fingerprintInput))}`;
    files[paths.personalizedDocumentPath] = JSON.stringify(document);
    expect(await makeStore(storage).loadPersonalized(reportDate)).toBeNull();
  });

  it("recovers personalized backup, rotates atomically, and serializes same-path saves", async () => {
    const { files, storage } = makeStorage({ rejectExistingRenameTarget: true });
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    await store.savePersonalized(reportDate, personalizedPrepared(), personalizedResult);
    const first = files[paths.personalizedDocumentPath]!;
    await Promise.all([
      makeStore(storage).savePersonalized(reportDate, personalizedPrepared(), personalizedResult),
      makeStore(storage).savePersonalized(reportDate, personalizedPrepared(), [
        { ...personalizedResult[0]!, directionIds: [] }, personalizedResult[1]!,
      ]),
    ]);
    expect(files[paths.personalizedBackupPath]).toBeTruthy();
    expect(files[`${paths.personalizedDocumentPath}.tmp`]).toBeUndefined();
    expect(files[`${paths.personalizedBackupPath}.tmp`]).toBeUndefined();
    files[paths.personalizedBackupPath] = first;
    files[paths.personalizedDocumentPath] = "corrupt";
    expect(await makeStore(storage).lookupPersonalizedReusable(
      reportDate, personalizedPrepared(),
    )).toEqual(personalizedResult);
  });

  it("fails personalized lookup closed on unreadable primary", async () => {
    const { files, storage, readText } = makeStorage();
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    await store.savePersonalized(reportDate, personalizedPrepared(), personalizedResult);
    files[paths.personalizedBackupPath] = files[paths.personalizedDocumentPath]!;
    readText.mockRejectedValueOnce(Object.assign(new Error("EIO"), { code: "EIO" }));
    await expect(store.lookupPersonalizedReusable(reportDate, personalizedPrepared()))
      .rejects.toThrow(/cannot read personalized filter checkpoint/);
    expect(readText).toHaveBeenCalledTimes(1);
  });

  it("removes manual and personalized primary, backup, and temp artifacts", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    await store.savePersonalized(reportDate, personalizedPrepared(), personalizedResult);
    const paths = store.pathsFor(reportDate);
    files[backupPath] = files[documentPath]!;
    files[paths.personalizedBackupPath] = files[paths.personalizedDocumentPath]!;
    for (const path of [documentPath, backupPath, paths.personalizedDocumentPath,
      paths.personalizedBackupPath]) files[`${path}.tmp`] = "tmp";
    await store.removeAll(reportDate);
    expect(Object.keys(files)).toEqual([]);
  });
});

describe("novelty filter checkpoint", () => {
  it("persists every planned paper's terminal outcome privately with strict schema and reuse", async () => {
    const { files, storage } = makeStorage();
    const writeTextWithMode = vi.fn(async (path: string, content: string) => {
      files[path] = content;
    });
    storage.writeTextWithMode = writeTextWithMode;
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);

    await store.saveNovelty(reportDate, noveltyPreparedTwoPapers(), [
      noveltyOutcome, secondPaperNoNoveltyOutcome,
    ]);

    const persisted = JSON.parse(files[paths.noveltyDocumentPath]!);
    expect(persisted).toMatchObject({
      schemaVersion: NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION,
      reportDate,
      result: [
        noveltyRecord,
        { paperKey: "arxiv:2608.00002", status: "no-novelty", reason: "validation-exhausted" },
      ],
    });
    // Typed no-novelty terminal outcomes are durable state, unlike degraded
    // transport/output-limit outcomes which are never persisted.
    expect(JSON.stringify(persisted)).toContain("validation-exhausted");
    expect(JSON.stringify(persisted)).not.toContain("never-persist");
    expect(JSON.stringify(persisted)).not.toContain("example.test");
    expect(writeTextWithMode).toHaveBeenCalledWith(
      expect.stringContaining(".novelty.json.tmp"), expect.any(String), 0o600,
    );
    expect(await makeStore(storage).lookupNoveltyReusable(
      reportDate, noveltyPreparedTwoPapers(),
    )).toEqual([
      noveltyRecord,
      { paperKey: "arxiv:2608.00002", status: "no-novelty", reason: "validation-exhausted" },
    ]);
    expect(await makeStore(storage).loadNovelty(reportDate)).toMatchObject({
      schemaVersion: NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION,
      result: [
        noveltyRecord,
        { paperKey: "arxiv:2608.00002", status: "no-novelty", reason: "validation-exhausted" },
      ],
    });
  });

  it("misses on any rendered identity change: basis evidence, directions, model, contracts", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    const hit = () => makeStore(storage).lookupNoveltyReusable(reportDate, noveltyPrepared());
    expect(await hit()).toEqual([noveltyRecord]);

    // Representative evidence change alters the rendered basis evidence.
    const evidenceInput = noveltyInput({
      representatives: [noveltyRepresentative(1, { abstract: "changed evidence" }),
        noveltyRepresentative(2), noveltyRepresentative(3)],
    });
    const evidencePlanned = planPersonalNoveltyCalls(evidenceInput, noveltyMatches());
    if (!evidencePlanned.ok) throw new Error("unexpected plan-too-large");
    const evidencePrepared = prepareNoveltyCheckpoint({
      plan: evidencePlanned.value, matches: noveltyMatches(),
      llm: compatibility().llm as any,
    });
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, evidencePrepared)).toBeNull();

    // Direction identity change (direction→representatives) invalidates.
    const directionPrepared = noveltyPrepared({
      matches: {
        directionRepresentatives: [
          { directionId: "direction.001", representativePaperKeys: ["arxiv:2501.00001"] },
          { directionId: "direction.002", representativePaperKeys: ["arxiv:2501.00002"] },
        ],
      },
    });
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, directionPrepared)).toBeNull();

    // Model identity change invalidates.
    const modelPrepared = noveltyPrepared({ llm: { ...compatibility().llm, model: "model-b" } });
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, modelPrepared)).toBeNull();

    // Contract version changes invalidate (and cannot be persisted as new).
    const contractPrepared = noveltyPrepared({
      promptContractVersion: PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION + 1,
    });
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, contractPrepared)).toBeNull();
    await expect(store.saveNovelty(reportDate, contractPrepared, [noveltyOutcome]))
      .rejects.toThrow(/unsupported personal novelty contract/);
  });

  it("does not invalidate on unrelated catalog entries or non-rendered changes", async () => {
    const { storage } = makeStorage();
    await makeStore(storage).saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    // An unreferenced representative and an unmatched daily paper are not part
    // of any rendered call, so the fingerprint is unchanged.
    const extended = noveltyPrepared({
      input: {
        papers: [noveltyDailyPaper(1), noveltyDailyPaper(2)],
        representatives: [noveltyRepresentative(1), noveltyRepresentative(2),
          noveltyRepresentative(3), noveltyRepresentative(4)],
      },
    });
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, extended))
      .toEqual([noveltyRecord]);
    // API key is not part of the generation identity.
    const otherSecret = noveltyPrepared({
      llm: { ...compatibility().llm, apiKey: "different-secret" },
    });
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, otherSecret))
      .toEqual([noveltyRecord]);
  });

  it("rejects caller-constructed snapshots, unknown paperKeys, and malformed novelty", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const snapshot = noveltyPrepared();
    await expect(store.saveNovelty(
      reportDate, JSON.parse(JSON.stringify(snapshot)) as any, [noveltyOutcome],
    )).rejects.toThrow(/prepared exact call-plan snapshot/);
    await expect(store.lookupNoveltyReusable(
      reportDate, JSON.parse(JSON.stringify(snapshot)) as any,
    )).rejects.toThrow(/prepared exact call-plan snapshot/);
    for (const invalid of [
      [{ ...noveltyOutcome, paperKey: "arxiv:2608.99999" }],
      [{ ...noveltyOutcome, novelty: { ...noveltyRecord.novelty, differenceType: "invented" } }],
      [{ ...noveltyOutcome, novelty: { ...noveltyRecord.novelty, comparisonBasis: ["arxiv:2501.99999"] } }],
      [{ ...noveltyOutcome, novelty: { ...noveltyRecord.novelty, evidenceDepth: "full-text" } }],
      [{ ...noveltyOutcome, novelty: { ...noveltyRecord.novelty, explanation: "" } }],
      [{ ...noveltyOutcome, novelty: { ...noveltyRecord.novelty, extra: true } }],
      // Transport/output-limit/checkpoint/input-invalid reasons are never durable.
      [{ ...noveltyOutcome, status: "no-novelty", reason: "transport" }],
      // A call entry can only terminate validation-exhausted, never plan-too-large.
      [{ ...noveltyOutcome, status: "no-novelty", reason: "plan-too-large" }],
    ]) {
      await expect(store.saveNovelty(reportDate, snapshot, invalid))
        .rejects.toThrow(/invalid personal novelty checkpoint/);
    }
    const paths = store.pathsFor(reportDate);
    expect(files[paths.noveltyDocumentPath]).toBeUndefined();
  });

  it("strictly revalidates persisted records on load and treats tampering as corrupt", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    await store.saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    const document = JSON.parse(files[paths.noveltyDocumentPath]!);
    document.result[0].novelty.explanation = "tampered trailing space ";
    files[paths.noveltyDocumentPath] = JSON.stringify(document);
    expect(await makeStore(storage).loadNovelty(reportDate)).toBeNull();
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, noveltyPrepared())).toBeNull();
  });

  it("recovers a valid backup, misses on corrupt both, and fails closed on unreadable", async () => {
    const { files, storage, readText } = makeStorage();
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    await store.saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    files[paths.noveltyBackupPath] = files[paths.noveltyDocumentPath]!;
    files[paths.noveltyDocumentPath] = "corrupt";
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, noveltyPrepared()))
      .toEqual([noveltyRecord]);
    files[paths.noveltyBackupPath] = "{bad-backup";
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, noveltyPrepared())).toBeNull();
    files[paths.noveltyDocumentPath] = "still corrupt";
    files[paths.noveltyBackupPath] = "still bad";
    readText.mockClear();
    readText.mockRejectedValueOnce(Object.assign(new Error("EIO"), { code: "EIO" }));
    await expect(store.lookupNoveltyReusable(reportDate, noveltyPrepared()))
      .rejects.toThrow(/cannot read personal novelty checkpoint/);
    expect(readText).toHaveBeenCalledTimes(1);
  });

  it("rotates novelty backups atomically and cleans temp artifacts", async () => {
    const { files, storage } = makeStorage({ rejectExistingRenameTarget: true });
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    await store.saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    const first = files[paths.noveltyDocumentPath]!;
    await store.saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    expect(files[paths.noveltyBackupPath]).toBe(first);
    expect(files[`${paths.noveltyDocumentPath}.tmp`]).toBeUndefined();
    expect(files[`${paths.noveltyBackupPath}.tmp`]).toBeUndefined();
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, noveltyPrepared()))
      .toEqual([noveltyRecord]);
  });

  it("serializes same-path novelty saves and permits replacement from a valid backup", async () => {
    const { files, storage } = makeStorage();
    await Promise.all([
      makeStore(storage).saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]),
      makeStore(storage).saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]),
    ]);
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, noveltyPrepared()))
      .toEqual([noveltyRecord]);
    const paths = makeStore(storage).pathsFor(reportDate);
    files[paths.noveltyBackupPath] = files[paths.noveltyDocumentPath]!;
    files[paths.noveltyDocumentPath] = "corrupt";
    await makeStore(storage).saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, noveltyPrepared()))
      .toEqual([noveltyRecord]);
  });

  it("rejects a result array longer than the planned call entries", async () => {
    const { storage } = makeStorage();
    const store = makeStore(storage);
    const snapshot = noveltyPreparedTwoPapers();
    await expect(store.saveNovelty(reportDate, snapshot, [
      noveltyOutcome,
      secondPaperNoNoveltyOutcome,
      { ...noveltyOutcome, paperKey: "arxiv:2608.00003", novelty: {
        ...noveltyRecord.novelty, comparisonBasis: ["arxiv:2501.00001"],
      } },
    ])).rejects.toThrow(/exceeds the planned call entries/);
  });

  it("rejects partial persisted results at load and replaces them with complete coverage", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    await store.saveNovelty(reportDate, noveltyPreparedTwoPapers(), [
      noveltyOutcome, secondPaperNoNoveltyOutcome,
    ]);
    const document = JSON.parse(files[paths.noveltyDocumentPath]!);
    // Drop the second paper's terminal outcome: partial coverage must never be
    // reused as a hit.
    document.result = [document.result[0]];
    files[paths.noveltyDocumentPath] = JSON.stringify(document);
    expect(await makeStore(storage).loadNovelty(reportDate)).toBeNull();
    expect(await makeStore(storage).lookupNoveltyReusable(
      reportDate, noveltyPreparedTwoPapers(),
    )).toBeNull();
    await makeStore(storage).saveNovelty(reportDate, noveltyPreparedTwoPapers(), [
      noveltyOutcome, secondPaperNoNoveltyOutcome,
    ]);
    expect(await makeStore(storage).lookupNoveltyReusable(
      reportDate, noveltyPreparedTwoPapers(),
    )).toEqual([
      noveltyRecord,
      { paperKey: "arxiv:2608.00002", status: "no-novelty", reason: "validation-exhausted" },
    ]);
  });

  it("never serves the original identity after recomputed fingerprint identity tampering", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const paths = store.pathsFor(reportDate);
    const snapshot = noveltyPrepared();
    await store.saveNovelty(reportDate, snapshot, [noveltyOutcome]);
    const document = JSON.parse(files[paths.noveltyDocumentPath]!);
    document.fingerprintInput.matches.directionRepresentatives[0].representativePaperKeys =
      ["arxiv:2501.00003"];
    document.fingerprint = fingerprintNoveltyCheckpointInput(document.fingerprintInput);
    files[paths.noveltyDocumentPath] = JSON.stringify(document);
    // The tampered document is internally consistent (its fingerprint
    // recomputes) but its identity no longer matches the original prepared
    // snapshot, so it can never be reused for the original rendered calls.
    expect(document.fingerprint).not.toBe(snapshot.fingerprint);
    expect(await makeStore(storage).lookupNoveltyReusable(reportDate, snapshot)).toBeNull();
  });

  it("removes novelty artifacts together with manual and personalized artifacts", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.save(reportDate, prepared(), result);
    await store.savePersonalized(reportDate, personalizedPrepared(), personalizedResult);
    await store.saveNovelty(reportDate, noveltyPrepared(), [noveltyOutcome]);
    const paths = store.pathsFor(reportDate);
    for (const path of [documentPath, backupPath, paths.personalizedDocumentPath,
      paths.personalizedBackupPath, paths.noveltyDocumentPath, paths.noveltyBackupPath]) {
      files[`${path}.tmp`] = "tmp";
    }
    files[backupPath] = files[documentPath]!;
    files[paths.personalizedBackupPath] = files[paths.personalizedDocumentPath]!;
    files[paths.noveltyBackupPath] = files[paths.noveltyDocumentPath]!;
    await store.removeAll(reportDate);
    expect(Object.keys(files)).toEqual([]);
  });

  it("exposes strict record decoding and public snapshot branding helpers", () => {
    const planned = planPersonalNoveltyCalls(noveltyInput(), noveltyMatches());
    if (!planned.ok) throw new Error("unexpected plan-too-large");
    expect(decodeNoveltyCheckpointRecords([noveltyRecord], planned.value)).toEqual({
      ok: true,
      value: [noveltyRecord],
    });
    expect(decodeNoveltyCheckpointRecords(
      [noveltyRecord, noveltyRecord], planned.value,
    )).toMatchObject({ ok: false });
    expect(decodeNoveltyCheckpointRecords(
      [{ ...noveltyRecord, paperKey: "arxiv:2608.00002" }], planned.value,
    )).toMatchObject({ ok: false, reason: expect.stringContaining("unknown or unplanned") });
    expect(decodeNoveltyCheckpointRecords("not-an-array", planned.value))
      .toMatchObject({ ok: false });
    expect(decodeNoveltyCheckpointRecords([], planned.value))
      .toMatchObject({ ok: false, reason: expect.stringContaining("cover every planned paper") });
    // A call entry can only terminate validation-exhausted, never plan-too-large.
    expect(decodeNoveltyCheckpointRecords([
      { paperKey: "arxiv:2608.00001", status: "no-novelty", reason: "plan-too-large" },
    ], planned.value)).toMatchObject({ ok: false, reason: expect.stringContaining("planned entry") });
    const twoPapers = planPersonalNoveltyCalls(noveltyInputTwoPapers(), noveltyMatchesTwoPapers());
    if (!twoPapers.ok) throw new Error("unexpected plan-too-large");
    expect(decodeNoveltyCheckpointRecords([
      noveltyRecord, secondPaperNoNoveltyOutcome,
    ], twoPapers.value)).toEqual({
      ok: true,
      value: [noveltyRecord, secondPaperNoNoveltyOutcome],
    });
    // Partial coverage of a two-paper plan is invalid (treated as a miss).
    expect(decodeNoveltyCheckpointRecords([noveltyRecord], twoPapers.value))
      .toMatchObject({ ok: false, reason: expect.stringContaining("cover every planned paper") });
    // More records than planned call entries are rejected by the bound.
    expect(decodeNoveltyCheckpointRecords([
      noveltyRecord, secondPaperNoNoveltyOutcome,
      { ...noveltyRecord, paperKey: "arxiv:2608.00003" },
    ], twoPapers.value)).toMatchObject({ ok: false, reason: expect.stringContaining("exceeds the planned call entries") });
    const snapshot = noveltyPrepared();
    expect(isPreparedNoveltyCheckpoint(snapshot)).toBe(true);
    expect(isPreparedNoveltyCheckpoint(JSON.parse(JSON.stringify(snapshot)))).toBe(false);
    expect(snapshot.fingerprint).toBe(fingerprintNoveltyCheckpointInput(snapshot.fingerprintInput));
    expect(Object.isFrozen(snapshot)).toBe(true);
    expect(Object.isFrozen(snapshot.fingerprintInput)).toBe(true);
    expect(() => prepareNoveltyCheckpoint({
      plan: JSON.parse(JSON.stringify(planned.value)),
      matches: noveltyMatches(),
      llm: compatibility().llm as any,
    })).toThrow(/exact prepared call plan/);
  });

  it("rejects a plan whose request options drift from the bounded call contract", async () => {
    const planned = planPersonalNoveltyCalls(noveltyInput(), noveltyMatches());
    if (!planned.ok) throw new Error("unexpected plan-too-large");
    const tampered = JSON.parse(JSON.stringify(planned.value));
    tampered.entries[0].request.options.maxCompletionTokens = 1;
    expect(decodeNoveltyFingerprintInput({
      fingerprintVersion: 1,
      promptContractVersion: PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION,
      resultContractVersion: PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION,
      matches: noveltyMatches(),
      plan: tampered,
      generation: {
        provider: "custom",
        endpointDigest: `sha256:${"a".repeat(64)}`,
        model: "model-a",
        mode: { kind: "temperature", temperature: 0 },
      },
    })).toBeNull();
  });
});
