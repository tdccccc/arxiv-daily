import { describe, expect, it, vi } from "vitest";
import {
  createStorageStateStore,
  DEFAULT_SETTINGS,
  type StorageAdapter,
} from "@arxiv-daily/core";
import { migrateLegacyRunState } from "../src/services/legacy-run-state";

function memoryStorage(): StorageAdapter {
  const data = new Map<string, string>();
  return {
    normalizePath: (path) => path.replace(/\\/g, "/"),
    readText: async (path) => {
      const value = data.get(path);
      if (value === undefined) throw new Error(`missing: ${path}`);
      return value;
    },
    writeText: async (path, content) => { data.set(path, content); },
    exists: async (path) => data.has(path),
    mkdir: async () => {},
    remove: async (path) => { data.delete(path); },
    rename: async (from, to) => {
      const value = data.get(from);
      if (value === undefined) throw new Error(`missing: ${from}`);
      data.set(to, value);
      data.delete(from);
    },
  };
}

describe("migrateLegacyRunState", () => {
  it("keeps valid legacy entries and drops unreadable ones instead of failing", async () => {
    const store = createStorageStateStore(memoryStorage(), DEFAULT_SETTINGS.output);
    await store.load();
    const warn = vi.fn();

    await migrateLegacyRunState(store, {
      "2026-09-22": { status: "completed", lastAttempt: 1, attempts: 1 },
      "2026-09-23": { status: "completed", attempts: 1 },
    } as never, warn);

    expect(Object.keys(store.snapshot())).toEqual(["2026-09-22"]);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("2026-09-23"));
    await expect(store.setSkipped("2026-09-24", "test")).resolves.toBeUndefined();
  });

  it("leaves an existing run state alone", async () => {
    const store = createStorageStateStore(memoryStorage(), DEFAULT_SETTINGS.output);
    await store.load();
    await store.setSkipped("2026-09-24", "current");

    await migrateLegacyRunState(store, {
      "2026-09-22": { status: "completed", lastAttempt: 1, attempts: 1 },
    }, vi.fn());

    expect(Object.keys(store.snapshot())).toEqual(["2026-09-24"]);
  });
});
