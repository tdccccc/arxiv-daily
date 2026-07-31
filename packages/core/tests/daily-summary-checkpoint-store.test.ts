import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import type { DailyPaperResult } from "../src/pipeline/daily-summary-assembler";
import {
  DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION,
  DailySummaryCheckpointStore,
  buildCheckpointEndpointDigest,
  buildDailySummaryCheckpointFingerprintInput,
  createDailySummaryCompatibilityFingerprint,
  decodeDailyPaperResult,
  deriveDailySummaryCheckpointPaths,
  sha256ForCheckpointTests,
  type DailySummaryCheckpointCompatibilityInput,
} from "../src/services/daily-summary-checkpoint-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const reportDate = "2026-08-01";
const documentPath = "arxiv-daily/.index/daily-summary-checkpoints/2026-08-01.json";
const backupPath = `${documentPath}.bak`;

const structuredResult: DailyPaperResult = {
  kind: "structured",
  summary: {
    id: "2608.00001",
    coreProblem: "Problem",
    keyMethod: "Method",
    mainResult: "Result",
    whyRelevant: "Relevant",
    limitations: "Limits",
  },
};

const validationFallback: DailyPaperResult = {
  kind: "fallback",
  reasonCode: "validation-exhausted",
  attempts: 3,
  originalAbstract: "Abstract.",
};

const transportFallback: DailyPaperResult = {
  kind: "fallback",
  reasonCode: "transport-exhausted",
  attempts: 1,
  originalAbstract: "Abstract.",
};

function compatibility(
  overrides: Partial<DailySummaryCheckpointCompatibilityInput> = {},
): DailySummaryCheckpointCompatibilityInput {
  return {
    paper: {
      id: "2608.00001",
      title: "Paper title",
      authors: "A. Author, B. Author",
      abstract: "Abstract.",
      abstractConclusion: "## Abstract\nAbstract.\n\n## Conclusion\nConclusion.",
      fullSections: "## Methods\nMethods.",
    },
    summaryLanguage: "zh",
    llm: {
      provider: "custom",
      baseUrl: "https://user:secret@example.test/v1?token=secret#fragment",
      model: "model-a",
      thinkingMode: false,
      reasoningEffort: "medium",
      apiKey: "must-never-persist",
    },
    temperature: 0,
    ...overrides,
  };
}

function makeStorage(options: { atomic?: boolean; rejectExistingRenameTarget?: boolean } = {}) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const rename = vi.fn(async (from: string, to: string) => {
    if (!(from in files)) throw new Error(`missing ${from}`);
    if (options.rejectExistingRenameTarget && (to in files || dirs.has(to))) {
      throw new Error(`destination exists: ${to}`);
    }
    files[to] = files[from]!;
    delete files[from];
  });
  const writeTextAtomic = vi.fn(async (path: string, content: string) => {
    files[path] = content;
  });
  const readText = vi.fn(async (path: string) => {
    if (!(path in files)) throw new Error(`missing ${path}`);
    return files[path]!;
  });
  const exists = vi.fn(async (path: string) => path in files || dirs.has(path));
  const storage: StorageAdapter = {
    normalizePath: (path) => path
      .replace(/\\/g, "/")
      .replace(/\/+/g, "/")
      .replace(/^\/+|\/+$/g, ""),
    readText,
    writeText: async (path, content) => { files[path] = content; },
    ...(options.atomic ? { writeTextAtomic } : {}),
    exists,
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { delete files[path]; dirs.delete(path); },
    rename,
  };
  return { files, dirs, storage, exists, readText, rename, writeTextAtomic };
}

function makeStore(
  storage: StorageAdapter,
  onWarning = vi.fn(),
  now = () => new Date("2026-08-01T12:00:00.000Z"),
) {
  return new DailySummaryCheckpointStore(
    storage,
    DEFAULT_SETTINGS.output,
    { now, onWarning },
  );
}

function secondCompatibility(): DailySummaryCheckpointCompatibilityInput {
  const first = compatibility();
  return {
    ...first,
    paper: {
      ...first.paper,
      id: "2608.00002",
      title: "Second paper",
    },
  };
}

describe("daily summary checkpoint fingerprint", () => {
  it("uses a deterministic versioned canonical input and excludes credentials", () => {
    const first = compatibility();
    const second = compatibility({
      llm: {
        ...first.llm,
        apiKey: "different-secret",
      },
    });

    expect(createDailySummaryCompatibilityFingerprint(first)).toBe(
      "sha256:c0d5e6caeb38a9f70384e1274997d4d05890367377b9c650948423bcc724d6f5",
    );
    expect(createDailySummaryCompatibilityFingerprint(second)).toBe(
      createDailySummaryCompatibilityFingerprint(first),
    );
    const canonical = buildDailySummaryCheckpointFingerprintInput(first);
    expect(canonical).toMatchObject({
      fingerprintVersion: 2,
      paper: {
        paperKey: "arxiv:2608.00001",
        sourceContent: {
          id: "2608.00001",
          trustedOriginalAbstract: "Abstract.",
        },
      },
      generation: {
        endpointDigest: buildCheckpointEndpointDigest(first.llm.baseUrl),
        summaryLanguage: "zh",
        provider: "custom",
        model: "model-a",
        mode: { kind: "temperature", temperature: 0 },
      },
      promptContractVersion: 1,
      resultContractVersion: 1,
    });
    expect(JSON.stringify(canonical)).not.toContain("secret");
    expect(JSON.stringify(canonical)).not.toContain("apiKey");
  });

  it.each([
    ["", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    ["abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"],
  ])("matches the SHA-256 standard vector for %j", (input, expected) => {
    expect(sha256ForCheckpointTests(input)).toBe(expected);
  });

  it("normalizes only effective prompt and fallback inputs", () => {
    const base = compatibility();
    const normalized = compatibility({
      summaryLanguage: undefined,
      paper: {
        ...base.paper,
        abstract: "  Abstract.  ",
        abstractConclusion: `  ${base.paper.abstractConclusion}  `,
        fullSections: "   ",
      },
    });
    const equivalent = compatibility({
      summaryLanguage: "zh",
      paper: { ...base.paper, fullSections: null },
    });
    expect(createDailySummaryCompatibilityFingerprint(normalized))
      .toBe(createDailySummaryCompatibilityFingerprint(equivalent));
    expect(buildDailySummaryCheckpointFingerprintInput(normalized).paper.sourceContent)
      .toMatchObject({ trustedOriginalAbstract: "Abstract.", fullSections: null });
  });

  it("ignores temperature and normalizes provider-specific effective thinking semantics", () => {
    const base = compatibility();
    const reasoning = compatibility({
      llm: { ...base.llm, thinkingMode: true, provider: "custom" },
      temperature: 0,
    });
    expect(createDailySummaryCompatibilityFingerprint({ ...reasoning, temperature: 0.9 }))
      .toBe(createDailySummaryCompatibilityFingerprint(reasoning));

    const anthropic = compatibility({
      llm: { ...base.llm, thinkingMode: true, provider: "anthropic", reasoningEffort: "unknown" },
    });
    const medium = compatibility({
      llm: { ...anthropic.llm, reasoningEffort: "medium" },
    });
    expect(createDailySummaryCompatibilityFingerprint(anthropic))
      .toBe(createDailySummaryCompatibilityFingerprint(medium));
    expect(buildDailySummaryCheckpointFingerprintInput(anthropic).generation.mode)
      .toEqual({ kind: "anthropic-thinking", budgetTokens: 8192 });
  });

  it("digests the exact effective chat URL without exposing endpoint text", () => {
    const sensitive = "https://user:pass@example.test/private/token?v=secret#fragment";
    const digest = buildCheckpointEndpointDigest(sensitive);
    expect(digest).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(digest).not.toContain("example.test");
    expect(buildCheckpointEndpointDigest("https://example.test/private/token?v=one"))
      .not.toBe(buildCheckpointEndpointDigest("https://example.test/private/token?v=two"));
    expect(buildCheckpointEndpointDigest("https://user:one@example.test/v1"))
      .not.toBe(buildCheckpointEndpointDigest("https://user:two@example.test/v1"));
    expect(() => buildCheckpointEndpointDigest("relative/path")).toThrow(/absolute URL/);
    expect(() => buildCheckpointEndpointDigest("file:///tmp/api")).toThrow(/http or https/);
  });

  it.each([
    ["paper id", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, paper: { ...value.paper, id: "2608.00099" } })],
    ["title", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, paper: { ...value.paper, title: "Changed" } })],
    ["authors", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, paper: { ...value.paper, authors: "Changed" } })],
    ["abstract", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, paper: { ...value.paper, abstract: "Changed" } })],
    ["abstract/conclusion source", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, paper: { ...value.paper, abstractConclusion: "Changed" } })],
    ["full sections", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, paper: { ...value.paper, fullSections: "Changed" } })],
    ["summary language", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, summaryLanguage: "en" as const })],
    ["provider", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, llm: { ...value.llm, provider: "openai" } })],
    ["endpoint", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, llm: { ...value.llm, baseUrl: "https://other.test/v1" } })],
    ["model", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, llm: { ...value.llm, model: "model-b" } })],
    ["thinking mode", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, llm: { ...value.llm, thinkingMode: true } })],
    ["effective reasoning effort", (value: DailySummaryCheckpointCompatibilityInput) => ({
      ...value,
      llm: { ...value.llm, thinkingMode: true, reasoningEffort: "high" },
    })],
    ["temperature", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, temperature: 0.2 })],
    ["prompt contract", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, promptContractVersion: 2 })],
    ["result contract", (value: DailySummaryCheckpointCompatibilityInput) => ({ ...value, resultContractVersion: 2 })],
  ])("changes when %s changes", (_name, mutate) => {
    const input = compatibility();
    expect(createDailySummaryCompatibilityFingerprint(mutate(input)))
      .not.toBe(createDailySummaryCompatibilityFingerprint(input));
  });
});

describe("strict DailyPaperResult decoding", () => {
  it.each([structuredResult, validationFallback, transportFallback])(
    "round-trips $kind",
    (result) => {
      expect(decodeDailyPaperResult(result, "2608.00001")).toEqual(result);
    },
  );

  it("trims and canonicalizes structured values exactly like generation", () => {
    const decoded = decodeDailyPaperResult({
      ...structuredResult,
      summary: {
        ...structuredResult.summary,
        coreProblem: "  Problem  ",
        mainResult: String.raw`Uses \(\alpha=0.1\).`,
      },
    }, "2608.00001");
    expect(decoded).toMatchObject({
      kind: "structured",
      summary: { coreProblem: "Problem", mainResult: String.raw`Uses $\alpha=0.1$.` },
    });
  });

  it.each([
    ["unknown kind", { kind: "other" }],
    ["extra root key", { ...structuredResult, extra: true }],
    ["mismatched summary id", { kind: "structured", summary: { ...structuredResult.summary, id: "2608.99999" } }],
    ["missing structured field", { kind: "structured", summary: { ...structuredResult.summary, limitations: undefined } }],
    ["empty structured field", { kind: "structured", summary: { ...structuredResult.summary, keyMethod: " " } }],
    ["invalid structured math", { kind: "structured", summary: { ...structuredResult.summary, mainResult: String.raw`Bare \alpha.` } }],
    ["unknown fallback reason", { ...validationFallback, reasonCode: "unknown" }],
    ["zero attempts", { ...validationFallback, attempts: 0 }],
    ["too many attempts", { ...validationFallback, attempts: 4 }],
    ["NaN attempts", { ...validationFallback, attempts: Number.NaN }],
    ["string attempts", { ...validationFallback, attempts: "3" }],
    ["fractional attempts", { ...validationFallback, attempts: 1.5 }],
    ["extra fallback key", { ...validationFallback, diagnostic: "raw model response" }],
  ])("rejects %s", (_name, value) => {
    expect(decodeDailyPaperResult(value, "2608.00001")).toBeNull();
  });
});

describe("DailySummaryCheckpointStore", () => {
  it("derives a date-scoped path under the configured hidden index", () => {
    expect(deriveDailySummaryCheckpointPaths(
      { normalizePath: (path) => path },
      DEFAULT_SETTINGS.output,
      reportDate,
    )).toEqual({
      directory: "arxiv-daily/.index/daily-summary-checkpoints",
      documentPath,
      backupPath,
    });
  });

  it("rejects non-calendar report dates before deriving storage paths", () => {
    expect(() => deriveDailySummaryCheckpointPaths(
      { normalizePath: (path) => path },
      DEFAULT_SETTINGS.output,
      "2026-02-31",
    )).toThrow(/invalid checkpoint report date/);
  });

  it.each([structuredResult, validationFallback])(
    "persists and reuses an exact compatible $kind result after reconstruction",
    async (result) => {
      const { files, storage } = makeStorage();
      const first = makeStore(storage);
      await first.upsert(reportDate, compatibility(), result);

      const raw = files[documentPath]!;
      expect(JSON.parse(raw)).toMatchObject({
        schemaVersion: DAILY_SUMMARY_CHECKPOINT_SCHEMA_VERSION,
        reportDate,
        entries: {
          "arxiv:2608.00001": { paperKey: "arxiv:2608.00001", result },
        },
      });
      expect(raw).not.toContain("must-never-persist");
      expect(raw).not.toContain("user:secret");
      expect(raw).not.toContain("token=secret");
      expect(raw).not.toContain("example.test");
      expect(raw).not.toContain("/v1");
      expect(await makeStore(storage).lookupReusable(reportDate, compatibility())).toEqual(result);
    },
  );

  it.each([
    ["prompt", { promptContractVersion: 2 }],
    ["result", { resultContractVersion: 2 }],
  ])("refuses to persist an unsupported %s contract", async (_name, overrides) => {
    const { storage } = makeStorage();
    await expect(makeStore(storage).upsert(
      reportDate,
      compatibility(overrides),
      structuredResult,
    )).rejects.toThrow(/unsupported daily summary contract versions/);
  });

  it("rejects a fallback whose original abstract disagrees with trusted fingerprint input", async () => {
    const { storage } = makeStorage();
    await expect(makeStore(storage).upsert(reportDate, compatibility(), {
      ...validationFallback,
      originalAbstract: "Tampered abstract.",
    })).rejects.toThrow(/invalid checkpoint result/);
  });

  it("isolates a persisted fallback whose original abstract was tampered with", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), validationFallback);
    const document = JSON.parse(files[documentPath]!);
    document.entries["arxiv:2608.00001"].result.originalAbstract = "Tampered abstract.";
    files[documentPath] = JSON.stringify(document);

    expect((await store.load(reportDate)).entries).toEqual({});
    expect(await store.lookupReusable(reportDate, compatibility())).toBeNull();
  });

  it("records transport exhaustion but intentionally does not reuse it", async () => {
    const { storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), transportFallback);
    expect((await store.load(reportDate)).entries["arxiv:2608.00001"]?.result)
      .toEqual(transportFallback);
    expect(await store.lookupReusable(reportDate, compatibility())).toBeNull();
  });

  it("allows a retried transport fallback to be overwritten by a reusable result", async () => {
    const { storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), transportFallback);
    expect(await store.lookupReusable(reportDate, compatibility())).toBeNull();

    await store.upsert(reportDate, compatibility(), structuredResult);

    expect((await store.load(reportDate)).entries["arxiv:2608.00001"]?.result)
      .toEqual(structuredResult);
    expect(await store.lookupReusable(reportDate, compatibility())).toEqual(structuredResult);
  });

  it("persists no path token and treats a query-only endpoint change as a miss", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const input = compatibility({
      llm: { ...compatibility().llm, baseUrl: "https://example.test/private/path-token?tenant=one" },
    });
    await store.upsert(reportDate, input, structuredResult);

    expect(files[documentPath]).not.toContain("path-token");
    expect(files[documentPath]).not.toContain("tenant=one");
    await expect(store.lookupReusable(reportDate, {
      ...input,
      llm: { ...input.llm, baseUrl: "https://example.test/private/path-token?tenant=two" },
    })).resolves.toBeNull();
  });

  it("returns a miss for changed compatibility while preserving reusable siblings", async () => {
    const { storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    await store.upsert(reportDate, secondCompatibility(), {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    });

    expect(await store.lookupReusable(reportDate, compatibility({
      llm: { ...compatibility().llm, model: "changed" },
    }))).toBeNull();
    expect(await store.lookupReusable(reportDate, secondCompatibility())).toMatchObject({
      kind: "structured",
      summary: { id: "2608.00002" },
    });
  });

  it.each([
    ["extra structured field", (entry: any) => { entry.result.summary.extra = "not allowed"; }],
    ["invalid structured math", (entry: any) => { entry.result.summary.mainResult = String.raw`Bare \alpha.`; }],
    ["unknown prompt contract", (entry: any) => {
      entry.fingerprintInput.promptContractVersion = 2;
      entry.fingerprint = `sha256:${sha256ForCheckpointTests(JSON.stringify(entry.fingerprintInput))}`;
    }],
    ["unknown result contract", (entry: any) => {
      entry.fingerprintInput.resultContractVersion = 2;
      entry.fingerprint = `sha256:${sha256ForCheckpointTests(JSON.stringify(entry.fingerprintInput))}`;
    }],
  ])("isolates %s while retaining valid siblings", async (_name, mutate) => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    await store.upsert(reportDate, secondCompatibility(), {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    });
    const document = JSON.parse(files[documentPath]!);
    mutate(document.entries["arxiv:2608.00001"]);
    files[documentPath] = JSON.stringify(document);

    const loaded = await store.load(reportDate);
    expect(loaded.entries["arxiv:2608.00001"]).toBeUndefined();
    expect(loaded.entries["arxiv:2608.00002"]).toBeDefined();
    expect(await store.lookupReusable(reportDate, compatibility())).toBeNull();
  });

  it.each([
    ["map key", (document: any) => {
      document.entries["arxiv:2608.00002"] = document.entries["arxiv:2608.00001"];
      delete document.entries["arxiv:2608.00001"];
    }],
    ["entry paperKey", (document: any) => {
      document.entries["arxiv:2608.00001"].paperKey = "arxiv:2608.00002";
    }],
    ["fingerprint paper identity", (document: any) => {
      const entry = document.entries["arxiv:2608.00001"];
      entry.fingerprintInput.paper.paperKey = "arxiv:2608.00002";
      entry.fingerprintInput.paper.sourceContent.id = "2608.00002";
      entry.fingerprint = `sha256:${sha256ForCheckpointTests(JSON.stringify(entry.fingerprintInput))}`;
    }],
  ])("rejects shifted paper binding through %s", async (_name, mutate) => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    const document = JSON.parse(files[documentPath]!);
    mutate(document);
    files[documentPath] = JSON.stringify(document);
    expect((await store.load(reportDate)).entries).toEqual({});
  });

  it.each([
    ["malformed JSON", "not json"],
    ["unknown schema", JSON.stringify({ schemaVersion: 99, reportDate, updatedAt: "2026-08-01T12:00:00Z", entries: {} })],
    ["wrong report date", JSON.stringify({ schemaVersion: 1, reportDate: "2026-07-31", updatedAt: "2026-08-01T12:00:00Z", entries: {} })],
    ["extra document key", JSON.stringify({ schemaVersion: 1, reportDate, updatedAt: "2026-08-01T12:00:00Z", entries: {}, extra: true })],
  ])("treats %s without backup as empty and warns", async (_name, raw) => {
    const { files, storage } = makeStorage();
    files[documentPath] = raw;
    const warning = vi.fn();
    const loaded = await makeStore(storage, warning).load(reportDate);
    expect(loaded.entries).toEqual({});
    expect(warning).toHaveBeenCalled();
  });

  it("keeps lookup tolerant but rejects mutation after a transient primary EIO without backup", async () => {
    const { files, storage, readText } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    const previous = files[documentPath]!;
    readText.mockRejectedValueOnce(Object.assign(new Error("read EIO"), { code: "EIO" }));

    await expect(store.lookupReusable(reportDate, compatibility())).resolves.toBeNull();
    expect(files[documentPath]).toBe(previous);

    readText.mockRejectedValueOnce(Object.assign(new Error("read EIO"), { code: "EIO" }));
    await expect(store.upsert(reportDate, compatibility(), validationFallback))
      .rejects.toThrow(/cannot mutate unreadable/);
    expect(files[documentPath]).toBe(previous);
  });

  it("rejects mutation on primary EIO even when a valid stale backup exists", async () => {
    const { files, storage, readText } = makeStorage();
    const store = makeStore(storage);
    const secondResult: DailyPaperResult = {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    };
    const thirdInput = {
      ...secondCompatibility(),
      paper: {
        ...secondCompatibility().paper,
        id: "2608.00003",
        title: "Third paper",
      },
    };
    const thirdResult: DailyPaperResult = {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00003" },
    };
    await store.upsert(reportDate, compatibility(), structuredResult);
    await store.upsert(reportDate, secondCompatibility(), secondResult);
    const primaryWithAAndB = files[documentPath]!;
    expect(Object.keys(JSON.parse(files[backupPath]!).entries)).toEqual([
      "arxiv:2608.00001",
    ]);
    readText.mockRejectedValueOnce(
      Object.assign(new Error("read EIO"), { code: "EIO" }),
    );

    await expect(store.upsert(reportDate, thirdInput, thirdResult))
      .rejects.toThrow(/cannot mutate unreadable/);

    expect(files[documentPath]).toBe(primaryWithAAndB);
    expect(Object.keys(JSON.parse(files[documentPath]!).entries).sort()).toEqual([
      "arxiv:2608.00001",
      "arxiv:2608.00002",
    ]);
    await expect(store.lookupReusable(reportDate, secondCompatibility()))
      .resolves.toEqual(secondResult);
    await expect(store.lookupReusable(reportDate, thirdInput)).resolves.toBeNull();
  });

  it("rejects mutation when checking primary existence fails", async () => {
    const { files, storage, exists } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    const primary = files[documentPath]!;
    exists.mockRejectedValueOnce(
      Object.assign(new Error("stat EIO"), { code: "EIO" }),
    );

    await expect(store.upsert(reportDate, secondCompatibility(), {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    })).rejects.toThrow(/cannot mutate unreadable/);
    expect(files[documentPath]).toBe(primary);
  });

  it("continues mutation from a valid backup when the primary is corrupt", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    await store.upsert(reportDate, secondCompatibility(), {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    });
    files[documentPath] = "corrupt";

    await store.upsert(reportDate, compatibility(), validationFallback);

    expect(await store.lookupReusable(reportDate, compatibility())).toEqual(validationFallback);
    // The last valid backup predates the second upsert; strict mutation uses it
    // rather than trusting entries found only in the corrupt primary.
    expect(await store.lookupReusable(reportDate, secondCompatibility())).toBeNull();
  });

  it("continues mutation from a valid backup when the primary is missing", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    files[backupPath] = files[documentPath]!;
    delete files[documentPath];

    await store.upsert(reportDate, secondCompatibility(), {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    });

    expect(Object.keys((await store.load(reportDate)).entries).sort()).toEqual([
      "arxiv:2608.00001",
      "arxiv:2608.00002",
    ]);
  });

  it("rejects mutation when both primary and backup are corrupt", async () => {
    const { files, storage } = makeStorage();
    files[documentPath] = "corrupt primary";
    files[backupPath] = "corrupt backup";

    await expect(makeStore(storage).upsert(reportDate, compatibility(), structuredResult))
      .rejects.toThrow(/cannot mutate unreadable/);
    expect(files[documentPath]).toBe("corrupt primary");
    expect(files[backupPath]).toBe("corrupt backup");
  });

  it("recovers a valid backup when the primary is corrupt", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    files[backupPath] = files[documentPath]!;
    files[documentPath] = "corrupt";
    const warning = vi.fn();

    expect(await makeStore(storage, warning).lookupReusable(reportDate, compatibility()))
      .toEqual(structuredResult);
    expect(warning.mock.calls.some(([message]) => String(message).includes("recovered from backup")))
      .toBe(true);
  });

  it("rotates backups across three upserts when rename rejects existing targets", async () => {
    const { files, storage } = makeStorage({ rejectExistingRenameTarget: true });
    const store = makeStore(storage);
    const secondResult: DailyPaperResult = {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    };
    const thirdInput = {
      ...secondCompatibility(),
      paper: { ...secondCompatibility().paper, id: "2608.00003", title: "Third paper" },
    };
    const thirdResult: DailyPaperResult = {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00003" },
    };

    await store.upsert(reportDate, compatibility(), structuredResult);
    await store.upsert(reportDate, secondCompatibility(), secondResult);
    await store.upsert(reportDate, thirdInput, thirdResult);
    files[documentPath] = "corrupt";

    const recovered = makeStore(storage);
    await expect(recovered.lookupReusable(reportDate, compatibility())).resolves.toEqual(structuredResult);
    await expect(recovered.lookupReusable(reportDate, secondCompatibility())).resolves.toEqual(secondResult);
    await expect(recovered.lookupReusable(reportDate, thirdInput)).resolves.toBeNull();
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
    expect(files[`${backupPath}.tmp`]).toBeUndefined();
  });

  it("keeps primary valid and cleans both temp files when backup publication fails", async () => {
    const { files, storage, rename } = makeStorage({ rejectExistingRenameTarget: true });
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    const primary = files[documentPath]!;
    files[backupPath] = primary;
    rename.mockImplementationOnce(async () => {
      throw new Error("injected backup publish failure");
    });

    await expect(store.upsert(reportDate, secondCompatibility(), {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    })).rejects.toThrow(/failed to save daily summary checkpoint/);
    expect(files[documentPath]).toBe(primary);
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
    expect(files[`${backupPath}.tmp`]).toBeUndefined();
    expect(await store.lookupReusable(reportDate, compatibility())).toEqual(structuredResult);
  });

  it("serializes same-path mutations across store instances", async () => {
    const { storage } = makeStorage();
    const first = makeStore(storage);
    const second = makeStore(storage);
    await Promise.all([
      first.upsert(reportDate, compatibility(), structuredResult),
      second.upsert(reportDate, secondCompatibility(), {
        ...structuredResult,
        summary: { ...structuredResult.summary, id: "2608.00002" },
      }),
    ]);

    expect(Object.keys((await first.load(reportDate)).entries).sort())
      .toEqual(["arxiv:2608.00001", "arxiv:2608.00002"]);
  });

  it("restores the previous primary when fallback replacement rename fails", async () => {
    const { files, storage, rename } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    const previous = files[documentPath]!;
    rename.mockImplementationOnce(async (from, to) => {
      if (!(from in files)) throw new Error(`missing ${from}`);
      files[to] = files[from]!;
      delete files[from];
    });
    rename.mockImplementationOnce(async () => {
      throw new Error("injected rename failure");
    });

    await expect(store.upsert(reportDate, compatibility(), validationFallback))
      .rejects.toThrow(/failed to save daily summary checkpoint/);
    expect(files[documentPath]).toBe(previous);
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
    expect(files[`${backupPath}.tmp`]).toBeUndefined();
    expect(await store.lookupReusable(reportDate, compatibility())).toEqual(structuredResult);
  });

  it("uses core-owned replacement even when the adapter exposes atomic writes", async () => {
    const { files, storage, writeTextAtomic } = makeStorage({ atomic: true });
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    const previous = files[documentPath]!;
    await store.upsert(reportDate, compatibility(), validationFallback);

    expect(writeTextAtomic).not.toHaveBeenCalled();
    expect(files[backupPath]).toBe(previous);
    expect(await store.lookupReusable(reportDate, compatibility())).toEqual(validationFallback);
  });

  it("removes one entry and then all checkpoint artifacts", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    await store.upsert(reportDate, compatibility(), structuredResult);
    await store.upsert(reportDate, secondCompatibility(), {
      ...structuredResult,
      summary: { ...structuredResult.summary, id: "2608.00002" },
    });
    expect(await store.remove(reportDate, "2608.00001")).toBe(true);
    expect(await store.remove(reportDate, "missing")).toBe(false);
    expect(Object.keys((await store.load(reportDate)).entries)).toEqual(["arxiv:2608.00002"]);

    files[backupPath] = "stale backup";
    files[`${documentPath}.tmp`] = "interrupted replacement";

    await store.removeAll(reportDate);
    expect(files[documentPath]).toBeUndefined();
    expect(files[backupPath]).toBeUndefined();
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
  });
});
