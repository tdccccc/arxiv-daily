import { describe, it, expect, vi } from "vitest";
import {
  RunHistoryStore,
  deriveRunHistoryStorePaths,
  formatRunHistoryRecords,
  type RunHistoryRecord,
} from "../src/services/run-history";
import { deriveStorageStateStorePaths } from "../src/services/state-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { StorageAdapter } from "../src/core/adapters";

function makeStorage(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  const dirs = new Set<string>();
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
    },
    async readText(path: string) {
      return files[this.normalizePath(path)];
    },
    async writeText(path: string, content: string) {
      files[this.normalizePath(path)] = content;
    },
    async exists(path: string) {
      const normalized = this.normalizePath(path);
      return normalized in files || dirs.has(normalized);
    },
    async mkdir(path: string) {
      dirs.add(this.normalizePath(path));
    },
    async rename(from: string, to: string) {
      const source = this.normalizePath(from);
      const target = this.normalizePath(to);
      files[target] = files[source];
      delete files[source];
    },
    async remove(path: string) {
      const normalized = this.normalizePath(path);
      delete files[normalized];
      dirs.delete(normalized);
    },
  } satisfies StorageAdapter;
  return { files, dirs, storage };
}

function record(overrides: Partial<RunHistoryRecord> = {}): RunHistoryRecord {
  return {
    schemaVersion: 1,
    at: "2026-06-25T10:00:00.000Z",
    date: "2026-06-24",
    event: "completed",
    trigger: "manual",
    status: "completed",
    resultKind: "completed",
    papersWritten: 10,
    dailyPath: "arxiv-daily/daily/2026-06-24.md",
    ...overrides,
  };
}

describe("RunHistoryStore", () => {
  it("derives run history path beside run state", () => {
    expect(
      deriveStorageStateStorePaths(DEFAULT_SETTINGS.output, (path) => path),
    ).toEqual({
      indexDir: "arxiv-daily/.index",
      runStatePath: "arxiv-daily/.index/run-state.json",
      runHistoryPath: "arxiv-daily/.index/run-history.jsonl",
    });

    expect(
      deriveRunHistoryStorePaths(DEFAULT_SETTINGS.output, (path) => path),
    ).toEqual({
      indexDir: "arxiv-daily/.index",
      runHistoryPath: "arxiv-daily/.index/run-history.jsonl",
    });
  });

  it("appends records as JSONL and creates the index directory", async () => {
    const { files, dirs, storage } = makeStorage();
    const store = RunHistoryStore.fromStorage(storage, DEFAULT_SETTINGS.output);

    await store.append(record({ event: "started", status: "running" }));
    await store.append(record({ at: "2026-06-25T10:01:00.000Z" }));

    expect(dirs.has("arxiv-daily")).toBe(true);
    expect(dirs.has("arxiv-daily/.index")).toBe(true);
    const lines = files["arxiv-daily/.index/run-history.jsonl"].trim().split("\n");
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0])).toMatchObject({
      schemaVersion: 1,
      event: "started",
      trigger: "manual",
      date: "2026-06-24",
    });
  });

  it("serializes concurrent appends in one runtime", async () => {
    const { files, storage } = makeStorage();
    const store = RunHistoryStore.fromStorage(storage, DEFAULT_SETTINGS.output);

    await Promise.all([
      store.append(record({ event: "started", at: "2026-06-25T10:00:00.000Z" })),
      store.append(record({ event: "completed", at: "2026-06-25T10:01:00.000Z" })),
    ]);

    expect(files["arxiv-daily/.index/run-history.jsonl"].trim().split("\n")).toHaveLength(2);
  });

  it("reads latest valid records in reverse chronological order", async () => {
    const existing = [
      JSON.stringify(record({ at: "2026-06-25T09:00:00.000Z", date: "2026-06-23" })),
      "not json",
      JSON.stringify(record({ at: "2026-06-25T10:00:00.000Z", date: "2026-06-24" })),
      JSON.stringify({ schemaVersion: 2, at: "x" }),
      "",
    ].join("\n");
    const { storage } = makeStorage({
      "arxiv-daily/.index/run-history.jsonl": existing,
    });
    const store = RunHistoryStore.fromStorage(storage, DEFAULT_SETTINGS.output);

    const latest = await store.readLatest(10);

    expect(latest.map((entry) => entry.date)).toEqual(["2026-06-24", "2026-06-23"]);
  });

  it("formats error messages for sharing from Dashboard", () => {
    expect(
      formatRunHistoryRecords([
        record({
          event: "failed",
          status: "failed_transient",
          resultKind: "failed_transient",
          reason: "network timeout",
          errorMessage: "network timeout",
        }),
      ]),
    ).toContain("error=network timeout");
  });

  it("safeAppend logs warnings instead of throwing", async () => {
    const logger = { warn: vi.fn() };
    const store = new RunHistoryStore(
      {
        async append() {
          throw new Error("disk full");
        },
        async readLatest() {
          return [];
        },
      },
      logger,
    );

    await store.safeAppend(record());

    expect(logger.warn).toHaveBeenCalledWith(
      "run history append failed",
      expect.any(Error),
    );
  });
});
