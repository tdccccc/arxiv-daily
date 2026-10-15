import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter, StorageEntry } from "../src/core/adapters";
import {
  FILTER_DECISION_RECORD_SCHEMA_VERSION,
  FILTER_DECISION_RETENTION_DAYS,
  FilterDecisionRecordStore,
  deriveFilterDecisionRecordPaths,
  type FilterDecisionRecordInput,
} from "../src/services/filter-decision-record-store";
import { buildCheckpointGenerationIdentity } from "../src/services/daily-summary-checkpoint-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const reportDate = "2026-08-01";
const directory = "arxiv-daily/.index/filter-decisions";
const documentPath = `${directory}/${reportDate}.json`;

function makeStorage(options: { withList?: boolean } = {}) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText: async (path) => {
      if (!(path in files)) throw new Error(`missing ${path}`);
      return files[path]!;
    },
    writeText: async (path, content) => { files[path] = content; },
    exists: async (path) => path in files || dirs.has(path),
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { delete files[path]; dirs.delete(path); },
    rename: async (from, to) => {
      if (!(from in files)) throw new Error(`missing ${from}`);
      if (to in files) throw new Error(`destination exists: ${to}`);
      files[to] = files[from]!;
      delete files[from];
    },
  };
  if (options.withList !== false) {
    storage.list = async (dir): Promise<StorageEntry[]> => {
      if (!dirs.has(dir)) throw new Error(`missing directory ${dir}`);
      return Object.keys(files)
        .filter((path) => path.startsWith(`${dir}/`) && !path.slice(dir.length + 1).includes("/"))
        .map((path) => ({ path, type: "file" as const }));
    };
  }
  return { files, dirs, storage };
}

function makeStore(storage: StorageAdapter, onWarning = vi.fn(), retentionDays?: number) {
  return new FilterDecisionRecordStore(storage, DEFAULT_SETTINGS.output, {
    now: () => new Date("2026-08-01T12:00:00.000Z"),
    onWarning,
    retentionDays,
  });
}

function input(overrides: Partial<FilterDecisionRecordInput> = {}): FilterDecisionRecordInput {
  return {
    reportDate,
    directions: [
      { ref: "topic-a#1", tag: "topic-a", id: "da", text: "Direction A" },
      { ref: "topic-b#1", tag: "topic-b", id: "db", text: "Direction B" },
    ],
    promptContractVersion: 4,
    resultContractVersion: 3,
    generation: buildCheckpointGenerationIdentity({
      provider: "custom",
      baseUrl: "https://user:secret@example.test/private/v1?token=secret",
      model: "model-a",
      thinkingMode: false,
      reasoningEffort: "medium",
    }, 0),
    candidates: [
      {
        id: "2608.00001",
        title: "First",
        category: "topic-a",
        directions: [{ id: "da", text: "Direction A", tag: "topic-a" }],
        relevanceScore: 80,
        outcome: "in_report",
      },
      {
        id: "2608.00002",
        title: "Second",
        category: "skip",
        directions: [],
        relevanceScore: 0,
        outcome: "skipped_by_model",
      },
    ],
    ...overrides,
  };
}

function dayOf(date: string): string {
  return `${directory}/${date}.json`;
}

describe("FilterDecisionRecordStore", () => {
  it("derives one document per report date under the index directory", () => {
    const { storage } = makeStorage();
    expect(deriveFilterDecisionRecordPaths(storage, DEFAULT_SETTINGS.output, reportDate)).toEqual({
      directory,
      documentPath,
    });
    expect(() => deriveFilterDecisionRecordPaths(storage, DEFAULT_SETTINGS.output, "2026-13-40"))
      .toThrow(/invalid filter decision report date/);
  });

  it("writes a versioned document and reads it back", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage);

    const saved = await store.save(input());

    expect(saved).toMatchObject({
      schemaVersion: FILTER_DECISION_RECORD_SCHEMA_VERSION,
      reportDate,
      writtenAt: "2026-08-01T12:00:00.000Z",
      promptContractVersion: 4,
      resultContractVersion: 3,
    });
    expect(JSON.parse(files[documentPath]!)).toEqual(saved);
    await expect(store.load(reportDate)).resolves.toEqual(saved);
    expect(Object.keys(files)).toEqual([documentPath]);
  });

  it("keeps the endpoint as a digest and carries no secret or abstract fields", async () => {
    const { storage, files } = makeStorage();
    await makeStore(storage).save(input());

    const raw = files[documentPath]!;
    expect(raw).not.toContain("secret");
    expect(raw).not.toContain("example.test");
    expect(JSON.parse(raw).generation.endpointDigest).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(JSON.parse(raw)).not.toHaveProperty("messages");
    expect(JSON.parse(raw).candidates[0]).not.toHaveProperty("abstract");
  });

  it("replaces the record of the same report date", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage);
    await store.save(input());

    await store.save(input({ candidates: [] }));

    expect((await store.load(reportDate))?.candidates).toEqual([]);
    expect(Object.keys(files)).toEqual([documentPath]);
  });

  it("reads a missing record as absent without a warning", async () => {
    const { storage } = makeStorage();
    const warn = vi.fn();
    await expect(makeStore(storage, warn).load(reportDate)).resolves.toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });

  it("reads corrupt, invalid or mismatched documents as absent with a warning", async () => {
    const { storage, files } = makeStorage();
    const warn = vi.fn();
    const store = makeStore(storage, warn);
    const good = await store.save(input());

    files[documentPath] = "{not json";
    await expect(store.load(reportDate)).resolves.toBeNull();

    files[documentPath] = JSON.stringify({ ...good, schemaVersion: 99 });
    await expect(store.load(reportDate)).resolves.toBeNull();

    files[documentPath] = JSON.stringify({ ...good, reportDate: "2026-08-02" });
    await expect(store.load(reportDate)).resolves.toBeNull();

    files[documentPath] = JSON.stringify({
      ...good,
      candidates: [{ ...good.candidates[0], outcome: "kept" }],
    });
    await expect(store.load(reportDate)).resolves.toBeNull();

    files[documentPath] = JSON.stringify({ ...good, extra: true });
    await expect(store.load(reportDate)).resolves.toBeNull();

    expect(warn).toHaveBeenCalledTimes(5);
    expect(warn.mock.calls.every(([message]) => String(message).includes(documentPath))).toBe(true);
  });

  it("reads an unreadable document as absent with a warning", async () => {
    const { storage, files } = makeStorage();
    const warn = vi.fn();
    const store = makeStore(storage, warn);
    await store.save(input());
    storage.readText = async () => { throw new Error("io denied"); };
    expect(files[documentPath]).toBeDefined();

    await expect(store.load(reportDate)).resolves.toBeNull();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining(documentPath), expect.any(Error));
  });

  it("lists report dates in order and ignores other files", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage);
    await store.save(input({ reportDate: "2026-07-30" }));
    await store.save(input({ reportDate: "2026-07-28" }));
    files[`${directory}/notes.txt`] = "x";
    files[`${dayOf("2026-07-29")}.tmp`] = "x";

    await expect(store.listReportDates()).resolves.toEqual(["2026-07-28", "2026-07-30"]);
  });

  it("lists nothing before any record exists or when the host cannot list", async () => {
    await expect(makeStore(makeStorage().storage).listReportDates()).resolves.toEqual([]);
    await expect(makeStore(makeStorage({ withList: false }).storage).listReportDates())
      .resolves.toEqual([]);
  });

  it("retains records for 180 days by default", () => {
    expect(FILTER_DECISION_RETENTION_DAYS).toBe(180);
  });

  it("prunes records older than the retention window when a record is written", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage);
    // 2026-02-02 is exactly 180 days before the report date.
    for (const date of ["2026-01-01", "2026-02-01", "2026-02-02", "2026-05-01"]) {
      await store.save(input({ reportDate: date }));
    }
    expect(Object.keys(files)).toHaveLength(4);

    await store.save(input());

    expect(Object.keys(files).sort()).toEqual([
      dayOf("2026-02-02"),
      dayOf("2026-05-01"),
      documentPath,
    ]);
  });

  it("honours a custom retention window", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage, vi.fn(), 7);
    await store.save(input({ reportDate: "2026-07-24" }));
    await store.save(input({ reportDate: "2026-07-25" }));

    await store.save(input());

    expect(Object.keys(files).sort()).toEqual([dayOf("2026-07-25"), documentPath]);
  });

  it("does not let a backfilled old date prune newer records", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage);
    await store.save(input());

    await store.save(input({ reportDate: "2026-03-01" }));

    expect(Object.keys(files).sort()).toEqual([dayOf("2026-03-01"), documentPath]);
  });

  it("still saves when pruning fails and warns", async () => {
    const { storage, files } = makeStorage();
    const warn = vi.fn();
    const store = makeStore(storage, warn);
    await store.save(input({ reportDate: "2026-01-01" }));
    const originalRemove = storage.remove;
    storage.remove = async (path) => {
      if (path === dayOf("2026-01-01")) throw new Error("remove denied");
      await originalRemove(path);
    };

    await expect(store.save(input())).resolves.toMatchObject({ reportDate });

    expect(files[documentPath]).toBeDefined();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("prune"), expect.any(Error));
  });

  it("restores the previous record when the replacement cannot be promoted", async () => {
    const { storage, files } = makeStorage();
    const store = makeStore(storage);
    const first = await store.save(input());
    const originalRename = storage.rename;
    let fail = true;
    storage.rename = async (from, to) => {
      if (fail && to === documentPath) { fail = false; throw new Error("rename denied"); }
      await originalRename(from, to);
    };

    await expect(store.save(input({ candidates: [] }))).rejects.toThrow(/failed to save/);

    expect(JSON.parse(files[documentPath]!)).toEqual(first);
    expect(Object.keys(files)).toEqual([documentPath]);
  });
});
