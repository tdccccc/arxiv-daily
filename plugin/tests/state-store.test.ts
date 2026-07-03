import { describe, it, expect, vi } from "vitest";
import {
  createStorageStateStore,
  deriveStorageStateStorePaths,
  StateStore,
} from "../src/services/state-store";
import type { RunState } from "../src/settings/types";
import type { StorageAdapter } from "../src/core/adapters";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

function makeStore(initial: RunState = {}) {
  const data: { runState: RunState } = { runState: { ...initial } };
  const load = vi.fn(async () => ({ runState: { ...data.runState } }));
  const save = vi.fn(async (d: { runState: RunState }) => {
    data.runState = { ...d.runState };
  });
  return { store: new StateStore(load, save), data, save };
}

function makeStorage(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  const dirs = new Set<string>();
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    async readText(path: string) {
      return files[path];
    },
    async writeText(path: string, content: string) {
      files[path] = content;
    },
    async exists(path: string) {
      return path in files || dirs.has(path);
    },
    async mkdir(path: string) {
      dirs.add(path);
    },
    async rename(from: string, to: string) {
      if (to in files || dirs.has(to)) {
        throw new Error("Destination file already exists!");
      }
      files[to] = files[from];
      delete files[from];
    },
    async remove(path: string) {
      delete files[path];
      dirs.delete(path);
    },
  } satisfies StorageAdapter;
  return { files, dirs, storage };
}

describe("StateStore", () => {
  it("get returns pending when no entry", async () => {
    const { store } = makeStore();
    await store.load();
    expect(store.get("2026-05-11").status).toBe("pending");
    expect(store.get("2026-05-11").attempts).toBe(0);
  });

  it("setRunning marks running with bumped attempts", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    const e = store.get("2026-05-11");
    expect(e.status).toBe("running");
    expect(e.attempts).toBe(1);
  });

  it("setCompleted marks completed and records papers", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 7);
    const e = store.get("2026-05-11");
    expect(e.status).toBe("completed");
    expect(e.papersWritten).toBe(7);
  });

  it("setFailed transient keeps attempts low; permanent after threshold", async () => {
    const { store } = makeStore();
    await store.load();
    for (let i = 0; i < 9; i++) {
      await store.setRunning("d");
      await store.setFailed("d", "transient", "boom");
      expect(store.get("d").status).toBe("failed_transient");
    }
    await store.setRunning("d");
    await store.setFailed("d", "transient", "boom");
    expect(store.get("d").status).toBe("failed_permanent");
  });

  it("setFailed permanent applies immediately", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("d");
    await store.setFailed("d", "permanent", "bad config");
    expect(store.get("d").status).toBe("failed_permanent");
  });

  it("isDone returns true for completed and failed_permanent", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("a");
    await store.setCompleted("a", 1);
    expect(store.isDone("a")).toBe(true);
    await store.setFailed("b", "permanent", "x");
    expect(store.isDone("b")).toBe(true);
    expect(store.isDone("c")).toBe(false);
  });

  it("persists via save callback on each mutation", async () => {
    const { store, save } = makeStore();
    await store.load();
    await store.setRunning("d");
    await store.setCompleted("d", 3);
    expect(save).toHaveBeenCalledTimes(2);
  });

  it("snapshot returns a copy of current state", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("a");
    const snap = store.snapshot();
    expect(snap.a.status).toBe("running");
    snap.a.status = "completed";
    expect(store.get("a").status).toBe("running");
  });

  it("clearDate removes one entry", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("a");
    await store.setRunning("b");
    await store.clearDate("a");
    expect(store.get("a").status).toBe("pending");
    expect(store.get("b").status).toBe("running");
  });

  it("clearAll removes every entry", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("a");
    await store.setRunning("b");
    await store.clearAll();
    expect(store.snapshot()).toEqual({});
  });

  it("replaceAll swaps state with a copy", async () => {
    const { store } = makeStore();
    await store.load();
    const next: RunState = {
      "2026-06-13": {
        status: "completed",
        lastAttempt: 1,
        attempts: 1,
        papersWritten: 2,
      },
    };
    await store.replaceAll(next);
    next["2026-06-13"].status = "running";
    expect(store.get("2026-06-13").status).toBe("completed");
  });

  it("failedDates returns sorted failed entries only", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("2026-05-12");
    await store.setFailed("2026-05-12", "transient", "x");
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 1);
    await store.setRunning("2026-05-10");
    await store.setFailed("2026-05-10", "permanent", "x");
    expect(store.failedDates()).toEqual(["2026-05-10", "2026-05-12"]);
  });

  it("derives storage run state path next to the paper index", () => {
    expect(
      deriveStorageStateStorePaths(
        DEFAULT_SETTINGS.output,
        (path) => path.replace(/\\/g, "/"),
      ),
    ).toEqual({
      indexDir: "arxiv-daily/.index",
      runStatePath: "arxiv-daily/.index/run-state.json",
      runHistoryPath: "arxiv-daily/.index/run-history.jsonl",
    });
  });

  it("persists run state through storage", async () => {
    const { files, dirs, storage } = makeStorage();
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output);

    await store.load();
    await store.setRunning("2026-06-13");
    await store.setCompleted("2026-06-13", 2);

    expect(dirs.has("arxiv-daily")).toBe(true);
    expect(dirs.has("arxiv-daily/.index")).toBe(true);
    const saved = JSON.parse(files["arxiv-daily/.index/run-state.json"]);
    expect(saved.schemaVersion).toBe(1);
    expect(saved.runState["2026-06-13"].status).toBe("completed");

    const reloaded = createStorageStateStore(storage, DEFAULT_SETTINGS.output);
    await reloaded.load();
    expect(reloaded.get("2026-06-13").papersWritten).toBe(2);
  });

  it("serializes concurrent storage mutations from separate stores targeting the same path", async () => {
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({
        schemaVersion: 1,
        runState: {},
      }),
    });
    const delayedStorage = {
      ...storage,
      async readText(path: string) {
        const content = await storage.readText(path);
        await new Promise((resolve) => setTimeout(resolve, 0));
        return content;
      },
    } satisfies StorageAdapter;
    const storeA = createStorageStateStore(delayedStorage, DEFAULT_SETTINGS.output);
    const storeB = createStorageStateStore(delayedStorage, DEFAULT_SETTINGS.output);

    await Promise.all([
      storeA.setRunning("2026-06-13"),
      storeB.setSkipped("2026-06-14", "user skipped"),
    ]);

    const saved = JSON.parse(files["arxiv-daily/.index/run-state.json"]);
    expect(saved.runState["2026-06-13"].status).toBe("running");
    expect(saved.runState["2026-06-14"].status).toBe("skipped");
  });

  it("falls back to the backup file when run-state.json is corrupt", async () => {
    const { storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": "{not-json",
      "arxiv-daily/.index/run-state.json.bak": JSON.stringify({
        runState: {
          "2026-06-13": {
            status: "completed",
            lastAttempt: 1,
            attempts: 1,
            papersWritten: 2,
          },
        },
      }),
    });
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.load()).resolves.toBeUndefined();

    expect(store.get("2026-06-13").status).toBe("completed");
    expect(store.get("2026-06-13").papersWritten).toBe(2);
  });

  it("falls back to an empty run state when run-state.json and backup are corrupt", async () => {
    const { storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": "{not-json",
      "arxiv-daily/.index/run-state.json.bak": "{also-not-json",
    });
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.load()).resolves.toBeUndefined();

    expect(store.snapshot()).toEqual({});
  });
});
