import { describe, it, expect, vi } from "vitest";
import {
  createStorageStateStore,
  deriveStorageStateStorePaths,
  StateStore,
} from "../src/services/state-store";
import type { RunState } from "../src/settings/types";
import type { StorageAdapter } from "../src/core/adapters";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

function cloneRunState(runState: RunState): RunState {
  return Object.fromEntries(
    Object.entries(runState).map(([date, entry]) => [date, { ...entry }]),
  );
}

function makeStore(initial: RunState = {}) {
  const data: { runState: RunState } = { runState: cloneRunState(initial) };
  const load = vi.fn(async () => ({ runState: cloneRunState(data.runState) }));
  const save = vi.fn(async (d: { runState: RunState }) => {
    data.runState = cloneRunState(d.runState);
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

  it("setPending preserves attempts after a running attempt", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("d");
    await store.setPending("d", "no papers yet");
    expect(store.get("d")).toMatchObject({
      status: "pending",
      attempts: 1,
      error: "no papers yet",
    });
  });

  it("recovers stale running dates to permanent failures", async () => {
    const { store } = makeStore({
      "2026-06-13": {
        status: "running",
        lastAttempt: 1,
        attempts: 3,
      },
      "2026-06-14": {
        status: "running",
        lastAttempt: 9_999_999,
        attempts: 1,
      },
    });
    await store.load();

    await expect(store.recoverStaleRunning(3_700_001, 3_600_000))
      .resolves.toEqual(["2026-06-13"]);

    expect(store.get("2026-06-13").status).toBe("failed_permanent");
    expect(store.get("2026-06-13").attempts).toBe(3);
    expect(store.get("2026-06-14").status).toBe("running");
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

  it("does not publish a mutation candidate before save succeeds", async () => {
    const data: { runState: RunState } = { runState: {} };
    let visibleDuringSave: RunState | undefined;
    let store!: StateStore;
    const save = vi.fn(async (candidate: { runState: RunState }) => {
      visibleDuringSave = store.snapshot();
      data.runState = cloneRunState(candidate.runState);
    });
    store = new StateStore(
      async () => ({ runState: cloneRunState(data.runState) }),
      save,
    );
    await store.load();

    await store.setRunning("2026-06-24");

    expect(visibleDuringSave).toEqual({});
    expect(store.get("2026-06-24").status).toBe("running");
  });

  it("publishes durable readback and rethrows when save fails before commit", async () => {
    const commitFailure = new Error("disk full");
    const durable: { runState: RunState } = {
      runState: {
        old: { status: "completed", lastAttempt: 1, attempts: 1, papersWritten: 2 },
      },
    };
    const store = new StateStore(
      async () => ({ runState: cloneRunState(durable.runState) }),
      async () => {
        throw commitFailure;
      },
    );
    await store.load();

    await expect(store.setRunning("new")).rejects.toBe(commitFailure);

    expect(store.snapshot()).toEqual(durable.runState);
  });

  it("keeps pre-mutation state when a failed save readback only partially matches the candidate", async () => {
    const commitFailure = new Error("fsync failed");
    const targetDate = "2026-06-24";
    const preMutationDurable: RunState = {
      unrelated: {
        status: "skipped",
        lastAttempt: 2,
        attempts: 0,
        error: "operator",
      },
    };
    const durable: { runState: RunState } = {
      runState: cloneRunState(preMutationDurable),
    };
    const store = new StateStore(
      async () => ({ runState: cloneRunState(durable.runState) }),
      async ({ runState }) => {
        durable.runState = cloneRunState(runState);
        durable.runState.unrelated = {
          ...durable.runState.unrelated,
          lastAttempt: 3,
        };
        throw commitFailure;
      },
    );
    await store.load();

    await expect(store.setCompleted(targetDate, 4)).rejects.toBe(commitFailure);

    expect(durable.runState[targetDate]?.status).toBe("completed");
    expect(store.snapshot()).toEqual(preMutationDurable);
    expect(store.get(targetDate).status).toBe("pending");
  });

  it("keeps pre-mutation state when confirmation readback only partially matches the candidate", async () => {
    const targetDate = "2026-06-24";
    const preMutationDurable: RunState = {
      unrelated: {
        status: "skipped",
        lastAttempt: 2,
        attempts: 0,
        error: "operator",
      },
    };
    const durable: { runState: RunState } = {
      runState: cloneRunState(preMutationDurable),
    };
    const store = new StateStore(
      async () => ({ runState: cloneRunState(durable.runState) }),
      async ({ runState }) => {
        durable.runState = cloneRunState(runState);
        durable.runState.unrelated = {
          ...durable.runState.unrelated,
          lastAttempt: 3,
        };
      },
    );
    await store.load();

    await expect(store.setCompleted(targetDate, 4)).rejects.toThrow(
      "run-state persistence confirmation mismatch",
    );

    expect(durable.runState[targetDate]?.status).toBe("completed");
    expect(store.snapshot()).toEqual(preMutationDurable);
    expect(store.get(targetDate).status).toBe("pending");
  });

  it("restores the freshly loaded pre-mutation snapshot when save and readback both fail", async () => {
    const commitFailure = new Error("disk full");
    const readbackFailure = new Error("disk unavailable");
    const staleMemory: RunState = {
      old: { status: "completed", lastAttempt: 1, attempts: 1, papersWritten: 2 },
    };
    const preMutationDurable: RunState = {
      latest: { status: "skipped", lastAttempt: 2, attempts: 0, error: "operator" },
    };
    const load = vi
      .fn<() => Promise<{ runState: RunState }>>()
      .mockResolvedValueOnce({ runState: cloneRunState(staleMemory) })
      .mockResolvedValueOnce({ runState: cloneRunState(preMutationDurable) })
      .mockRejectedValueOnce(readbackFailure);
    const store = new StateStore(load, async () => {
      throw commitFailure;
    });
    await store.load();

    await expect(store.setRunning("new")).rejects.toBe(commitFailure);

    expect(store.snapshot()).toEqual(preMutationDurable);
  });

  it("accepts a save error when readback confirms the intended candidate", async () => {
    const durable: { runState: RunState } = { runState: {} };
    const store = new StateStore(
      async () => ({ runState: cloneRunState(durable.runState) }),
      async (candidate) => {
        durable.runState = cloneRunState(candidate.runState);
        throw new Error("rename reported failure after durable write");
      },
    );
    await store.load();

    await expect(store.setCompleted("2026-06-24", 4)).resolves.toBeUndefined();

    expect(store.get("2026-06-24")).toMatchObject({
      status: "completed",
      papersWritten: 4,
    });
  });

  it("keeps the mutation queue usable after a rejected save", async () => {
    const durable: { runState: RunState } = { runState: {} };
    let failNextSave = true;
    const store = new StateStore(
      async () => ({ runState: cloneRunState(durable.runState) }),
      async (candidate) => {
        if (failNextSave) {
          failNextSave = false;
          throw new Error("temporary disk failure");
        }
        durable.runState = cloneRunState(candidate.runState);
      },
    );
    await store.load();

    await expect(store.setRunning("2026-06-24")).rejects.toThrow(
      "temporary disk failure",
    );
    expect(store.get("2026-06-24").status).toBe("pending");

    await expect(store.setRunning("2026-06-24")).resolves.toBeUndefined();
    expect(store.get("2026-06-24")).toMatchObject({
      status: "running",
      attempts: 1,
    });
  });

  const failedMutationCases: Array<{
    name: string;
    initial: RunState;
    mutate: (store: StateStore) => Promise<unknown>;
  }> = [
    {
      name: "setSkipped",
      initial: {},
      mutate: (store) => store.setSkipped("d", "skip"),
    },
    { name: "setRunning", initial: {}, mutate: (store) => store.setRunning("d") },
    {
      name: "setCompleted",
      initial: {},
      mutate: (store) => store.setCompleted("d", 2),
    },
    {
      name: "setPending",
      initial: {},
      mutate: (store) => store.setPending("d", "later"),
    },
    {
      name: "setFailed",
      initial: {},
      mutate: (store) => store.setFailed("d", "transient", "network"),
    },
    {
      name: "clearDate",
      initial: { d: { status: "running", lastAttempt: 1, attempts: 1 } },
      mutate: (store) => store.clearDate("d"),
    },
    {
      name: "clearAll",
      initial: { d: { status: "running", lastAttempt: 1, attempts: 1 } },
      mutate: (store) => store.clearAll(),
    },
    {
      name: "replaceAll",
      initial: { d: { status: "running", lastAttempt: 1, attempts: 1 } },
      mutate: (store) =>
        store.replaceAll({
          replacement: { status: "completed", lastAttempt: 2, attempts: 1 },
        }),
    },
    {
      name: "recoverStaleRunning",
      initial: { d: { status: "running", lastAttempt: 1, attempts: 1 } },
      mutate: (store) => store.recoverStaleRunning(100, 10),
    },
  ];

  it.each(failedMutationCases)(
    "$name publishes durable state rather than a failed candidate",
    async ({ initial, mutate }) => {
      const durable = { runState: cloneRunState(initial) };
      const store = new StateStore(
        async () => ({ runState: cloneRunState(durable.runState) }),
        async () => {
          throw new Error("save failed");
        },
      );
      await store.load();

      await expect(mutate(store)).rejects.toThrow("save failed");

      expect(store.snapshot()).toEqual(initial);
    },
  );

  it("serializes load publication behind an earlier load and before a later mutation", async () => {
    const durable: { runState: RunState } = {
      runState: {
        old: { status: "completed", lastAttempt: 1, attempts: 1, papersWritten: 1 },
      },
    };
    let releaseFirstLoad!: () => void;
    const firstLoadCanFinish = new Promise<void>((resolve) => {
      releaseFirstLoad = resolve;
    });
    let loadCalls = 0;
    const store = new StateStore(
      async () => {
        loadCalls += 1;
        const snapshot = cloneRunState(durable.runState);
        if (loadCalls === 1) await firstLoadCanFinish;
        return { runState: snapshot };
      },
      async ({ runState }) => {
        durable.runState = cloneRunState(runState);
      },
    );

    const staleLoad = store.load();
    const laterMutation = store.setRunning("new");
    await new Promise((resolve) => setTimeout(resolve, 0));
    releaseFirstLoad();
    await Promise.all([staleLoad, laterMutation]);

    expect(store.get("old").status).toBe("completed");
    expect(store.get("new")).toMatchObject({ status: "running", attempts: 1 });
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

  it("rejects mutations when the authoritative primary is corrupt instead of using backup state", async () => {
    const logger = { warn: vi.fn() };
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": "{not-json",
      "arxiv-daily/.index/run-state.json.bak": JSON.stringify({
        runState: {
          backup: { status: "completed", lastAttempt: 1, attempts: 1 },
        },
      }),
    });
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output, logger);

    await expect(store.setRunning("new")).rejects.toThrow();

    expect(files["arxiv-daily/.index/run-state.json"]).toBe("{not-json");
    expect(store.get("new").status).toBe("pending");
  });

  it("rejects mutations when the authoritative primary omits runState", async () => {
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({ schemaVersion: 1 }),
    });
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output, {
      warn: vi.fn(),
    });

    await expect(store.clearAll()).rejects.toThrow("invalid run-state root");

    expect(JSON.parse(files["arxiv-daily/.index/run-state.json"])).toEqual({
      schemaVersion: 1,
    });
  });

  it("rejects mutations against an explicitly unsupported run-state schema", async () => {
    const original = {
      schemaVersion: 2,
      runState: {
        future: { status: "completed", lastAttempt: 1, attempts: 1 },
      },
    };
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify(original),
    });
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output, {
      warn: vi.fn(),
    });

    await expect(store.setRunning("new")).rejects.toThrow(
      "unsupported run-state schema version: 2",
    );

    expect(JSON.parse(files["arxiv-daily/.index/run-state.json"])).toEqual(
      original,
    );
    expect(store.get("new").status).toBe("pending");
  });

  it("rejects mutations when the authoritative primary has schema-invalid entries", async () => {
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({
        runState: {
          invalid: { status: "running", lastAttempt: 1, attempts: "one" },
        },
      }),
    });
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output, {
      warn: vi.fn(),
    });

    await expect(store.clearAll()).rejects.toThrow("invalid run-state entry");

    const saved = JSON.parse(files["arxiv-daily/.index/run-state.json"]);
    expect(saved.runState.invalid.attempts).toBe("one");
  });

  it.each([
    ["error", "42"],
    ["papersWritten", "1e400"],
  ] as const)(
    "rejects authoritative entries when optional %s has an invalid value",
    async (field, serializedValue) => {
      const original = `{"schemaVersion":1,"runState":{"invalid":{"status":"completed","lastAttempt":1,"attempts":1,"${field}":${serializedValue}}}}`;
      const { files, storage } = makeStorage({
        "arxiv-daily/.index/run-state.json": original,
      });
      const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output, {
        warn: vi.fn(),
      });

      await expect(store.clearAll()).rejects.toThrow("invalid run-state entry");

      expect(files["arxiv-daily/.index/run-state.json"]).toBe(original);
      expect(store.snapshot()).toEqual({});
    },
  );

  it("rejects mutations when the authoritative primary is unreadable", async () => {
    const logger = { warn: vi.fn() };
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({ runState: {} }),
      "arxiv-daily/.index/run-state.json.bak": JSON.stringify({ runState: {} }),
    });
    const unreadableStorage = {
      ...storage,
      async readText(path: string) {
        if (path === "arxiv-daily/.index/run-state.json") {
          throw new Error("permission denied");
        }
        return storage.readText(path);
      },
    } satisfies StorageAdapter;
    const store = createStorageStateStore(
      unreadableStorage,
      DEFAULT_SETTINGS.output,
      logger,
    );

    await expect(store.clearAll()).rejects.toThrow("permission denied");

    expect(files["arxiv-daily/.index/run-state.json"]).toBeDefined();
    expect(store.snapshot()).toEqual({});
  });

  it("does not falsely confirm an empty clearAll candidate through corrupt fallback state", async () => {
    const initial = JSON.stringify({
      runState: {
        old: { status: "completed", lastAttempt: 1, attempts: 1 },
      },
    });
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": initial,
    });
    const corruptingStorage = {
      ...storage,
      async rename(from: string, to: string) {
        if (
          from === "arxiv-daily/.index/run-state.json.tmp" &&
          to === "arxiv-daily/.index/run-state.json"
        ) {
          files[to] = "{corrupt-after-write";
          delete files[from];
          delete files["arxiv-daily/.index/run-state.json.bak"];
          throw new Error("rename confirmation failed");
        }
        await storage.rename(from, to);
      },
    } satisfies StorageAdapter;
    const store = createStorageStateStore(
      corruptingStorage,
      DEFAULT_SETTINGS.output,
      { warn: vi.fn() },
    );
    await store.load();

    await expect(store.clearAll()).rejects.toThrow("rename confirmation failed");

    expect(store.get("old").status).toBe("completed");
    expect(files["arxiv-daily/.index/run-state.json"]).toBe("{corrupt-after-write");
  });

  it("fails closed on an unsupported primary schema instead of loading a valid backup", async () => {
    const logger = { warn: vi.fn() };
    const { storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({
        schemaVersion: 2,
        runState: {
          future: {
            status: "completed",
            lastAttempt: 2,
            attempts: 1,
            papersWritten: 3,
          },
        },
      }),
      "arxiv-daily/.index/run-state.json.bak": JSON.stringify({
        schemaVersion: 1,
        runState: {
          backup: {
            status: "completed",
            lastAttempt: 1,
            attempts: 1,
            papersWritten: 2,
          },
        },
      }),
    });
    const store = createStorageStateStore(
      storage,
      DEFAULT_SETTINGS.output,
      logger,
    );

    await expect(store.load()).rejects.toThrow(
      "unsupported run-state schema version: 2",
    );

    expect(store.snapshot()).toEqual({});
    expect(store.get("future").status).toBe("pending");
    expect(store.get("backup").status).toBe("pending");
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it("loads a schema-less legacy run-state document", async () => {
    const { storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({
        runState: {
          legacy: {
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

    expect(store.get("legacy")).toMatchObject({
      status: "completed",
      papersWritten: 2,
    });
  });

  it.each([
    ["schema 1", '"schemaVersion":1,', "error", "42"],
    ["schema 1", '"schemaVersion":1,', "papersWritten", "1e400"],
    ["schema-less legacy", "", "error", "42"],
    ["schema-less legacy", "", "papersWritten", "1e400"],
  ] as const)(
    "fails closed when ordinary load sees invalid optional %s %s",
    async (_documentKind, schemaPrefix, field, serializedValue) => {
      const original = `{${schemaPrefix}"runState":{"invalid":{"status":"completed","lastAttempt":1,"attempts":1,"${field}":${serializedValue}}}}`;
      const { storage } = makeStorage({
        "arxiv-daily/.index/run-state.json": original,
      });
      const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output, {
        warn: vi.fn(),
      });

      await expect(store.load()).rejects.toThrow("invalid run-state entry");

      expect(store.snapshot()).toEqual({});
      expect(store.get("invalid").status).toBe("pending");
    },
  );

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

  it("falls back to backup when the primary run-state file is missing", async () => {
    const { storage } = makeStorage({
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

    await store.load();

    expect(store.get("2026-06-13").status).toBe("completed");
    expect(store.get("2026-06-13").papersWritten).toBe(2);
  });

  it("ignores schema-invalid run-state entries with wrong field types", async () => {
    const { storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({
        runState: {
          badStatus: { status: "unknown", lastAttempt: 1, attempts: 1 },
          badAttempts: { status: "running", lastAttempt: 1, attempts: "1" },
          good: { status: "pending", lastAttempt: 2, attempts: 3 },
        },
      }),
    });
    const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output);

    await store.load();

    expect(store.snapshot()).toEqual({
      good: { status: "pending", lastAttempt: 2, attempts: 3 },
    });
  });

  it("restores the previous primary file if atomic write fails after backup rename", async () => {
    const { files, storage } = makeStorage({
      "arxiv-daily/.index/run-state.json": JSON.stringify({
        runState: {
          old: { status: "completed", lastAttempt: 1, attempts: 1 },
        },
      }),
    });
    const flakyStorage = {
      ...storage,
      async rename(from: string, to: string) {
        if (from.endsWith(".tmp") && to.endsWith("run-state.json")) {
          throw new Error("crash before tmp promote");
        }
        await storage.rename(from, to);
      },
    } satisfies StorageAdapter;
    const store = createStorageStateStore(flakyStorage, DEFAULT_SETTINGS.output);

    await expect(store.setRunning("new")).rejects.toThrow("crash before tmp promote");

    expect(files["arxiv-daily/.index/run-state.json"]).toBeDefined();
    expect(files["arxiv-daily/.index/run-state.json.bak"]).toBeUndefined();
    const recovered = createStorageStateStore(storage, DEFAULT_SETTINGS.output);
    await recovered.load();
    expect(recovered.get("old").status).toBe("completed");
    expect(recovered.get("new").status).toBe("pending");
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
