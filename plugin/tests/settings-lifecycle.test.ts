import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import {
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
  DEFAULT_SETTINGS,
  Logger,
  OperationRegistry,
  RunHistoryStore,
  RunLock,
  SchedulerService,
  StateStore,
  type StorageAdapter,
} from "@arxiv-daily/core";
import ArxivDailyPlugin, { resolvePluginDir } from "../main.ts";
import { settingsAndStateFromPersistedData } from "../src/settings/load";
import { SettingsChangeService } from "../src/settings/change-service";

const pluginMainSource = readFileSync(
  resolve(dirname(fileURLToPath(import.meta.url)), "../main.ts"),
  "utf8",
);

const savedCategories = ["cs.CL", "stat.ML", "custom.quantum-ai"];

function persistedData(categories: string[]) {
  return {
    settings: {
      arxiv: {
        category: categories[0] ?? "cs.CL",
        categories,
        topics: [],
        timezone: "UTC",
      },
    },
    runState: {},
  };
}

function memoryStorage(files: Record<string, string> = {}): StorageAdapter {
  const data = new Map(Object.entries(files));
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

function makeStateStore() {
  const data: { runState: Record<string, any> } = { runState: {} };
  return new StateStore(
    async () => ({
      runState: Object.fromEntries(
        Object.entries(data.runState).map(([date, entry]) => [date, { ...entry }]),
      ),
    }),
    async ({ runState }) => {
      data.runState = Object.fromEntries(
        Object.entries(runState).map(([date, entry]) => [date, { ...entry }]),
      );
    },
  );
}

describe("plugin directory resolution", () => {
  it("prioritizes the manifest directory", () => {
    expect(resolvePluginDir("custom/plugin", ".config", "arxiv-daily")).toBe(
      "custom/plugin",
    );
  });

  it("falls back to the vault configuration directory", () => {
    expect(resolvePluginDir(undefined, ".vault-config", "arxiv-daily")).toBe(
      ".vault-config/plugins/arxiv-daily",
    );
  });
});

describe("plugin settings reload lifecycle", () => {
  it("routes base URL persistence through effective-endpoint cancellation", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    const cancel = vi.fn();
    Object.assign(plugin, {
      settings,
      operations: {
        snapshot: () => [{ id: "generation:1", kind: "personal-library-direction-generation" }],
        cancel,
      },
      logger: { setSensitiveValues: vi.fn() },
      saveData: vi.fn().mockResolvedValue(undefined),
    });

    await plugin.setLlmBaseUrl(`${settings.llm.baseUrl}/`);
    expect(cancel).not.toHaveBeenCalled();
    await plugin.setLlmBaseUrl("https://other.example/v1");
    expect(cancel).toHaveBeenCalledWith("generation:1", "model endpoint changed");
  });

  it("rolls back base URL when persistence fails", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    const previous = settings.llm.baseUrl;
    Object.assign(plugin, {
      settings,
      operations: { snapshot: () => [], cancel: vi.fn() },
      logger: { setSensitiveValues: vi.fn() },
      saveData: vi.fn().mockRejectedValue(new Error("disk full")),
    });

    await expect(plugin.setLlmBaseUrl("https://other.example/v1")).rejects.toThrow("disk full");
    expect(settings.llm.baseUrl).toBe(previous);
  });

  it("serializes overlapping base URL setters without stale rollback", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    const persisted: string[] = [];
    let releaseFirst!: () => void;
    const saveData = vi.fn(async (data: { settings: typeof settings }) => {
      const value = data.settings.llm.baseUrl;
      if (saveData.mock.calls.length === 1) {
        await new Promise<void>((resolve) => { releaseFirst = resolve; });
        throw new Error("first failed");
      }
      persisted.push(value);
    });
    Object.assign(plugin, {
      settings,
      operations: { snapshot: () => [], cancel: vi.fn() },
      logger: { setSensitiveValues: vi.fn() },
      saveData,
    });

    const first = plugin.setLlmBaseUrl("https://first.example/v1");
    const firstRejected = expect(first).rejects.toThrow("first failed");
    await vi.waitFor(() => expect(saveData).toHaveBeenCalledTimes(1));
    const second = plugin.setLlmBaseUrl("https://second.example/v1");
    expect(settings.llm.baseUrl).toBe("https://first.example/v1");
    releaseFirst();
    await firstRejected;
    await second;

    expect(settings.llm.baseUrl).toBe("https://second.example/v1");
    expect(persisted).toEqual(["https://second.example/v1"]);
  });

  it("passes the exact candidate to saveData instead of capturing live settings", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const live = structuredClone(DEFAULT_SETTINGS);
    const candidate = structuredClone(DEFAULT_SETTINGS);
    candidate.llm.model = "candidate-model";
    const saveData = vi.fn().mockResolvedValue(undefined);
    Object.assign(plugin, { settings: live, saveData });

    await (plugin as any).persistSettings(candidate);

    expect(saveData).toHaveBeenCalledWith({ settings: candidate });
    expect((saveData.mock.calls[0]?.[0] as { settings: typeof candidate }).settings)
      .toBe(candidate);
    expect(live.llm.model).toBe(DEFAULT_SETTINGS.llm.model);
  });

  it("validates schedule enablement without requiring structuredClone", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    Object.assign(plugin, { settings: structuredClone(DEFAULT_SETTINGS) });
    vi.stubGlobal("structuredClone", undefined);
    try {
      await expect(plugin.setScheduleEnabled(true)).resolves.toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("detects both operation-registry work and scheduler active runs", () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const activeRuns = vi.fn().mockReturnValue([]);
    Object.assign(plugin, {
      operations: new OperationRegistry(),
      scheduler: { activeRuns },
    });

    expect(plugin.hasActiveOutputWork()).toBe(false);
    const operation = plugin.operations.begin("pdf-download", "Download", "paper-1");
    expect(plugin.hasActiveOutputWork()).toBe(true);
    operation.finish();
    expect(plugin.hasActiveOutputWork()).toBe(false);

    activeRuns.mockReturnValue(["2026-08-10"]);
    expect(plugin.hasActiveOutputWork()).toBe(true);
  });

  it("holds paper-index and paper-note leases against transitions in both directions", async () => {
    const plugin = new ArxivDailyPlugin();
    Object.assign(plugin, {
      scheduler: { activeRuns: vi.fn().mockReturnValue([]) },
    });
    let finishWork!: () => void;
    const work = new Promise<void>((resolve) => { finishWork = resolve; });

    const active = plugin.withOutputOperation(
      "paper-note",
      "Create note",
      "2608.00001",
      () => work,
    );
    expect(plugin.hasActiveOutputWork()).toBe(true);
    expect(() => (plugin as any).beginOutputTransition()).toThrow(/active/i);

    finishWork();
    await active;
    const release = (plugin as any).beginOutputTransition();
    await expect(plugin.withOutputOperation(
      "paper-index",
      "Mark paper",
      "2608.00001",
      async () => undefined,
    )).rejects.toThrow(/output directories/i);
    release();
    expect(plugin.hasActiveOutputWork()).toBe(false);
  });

  it("rejects new operations while an output-store transition is active", () => {
    const plugin = new ArxivDailyPlugin();
    Object.assign(plugin, {
      scheduler: { activeRuns: vi.fn().mockReturnValue([]) },
    });

    const release = (plugin as any).beginOutputTransition();
    expect(() =>
      plugin.operations.begin("pdf-download", "Download", "paper-1"),
    ).toThrow(/output directories/i);
    release();

    const operation = plugin.operations.begin(
      "pdf-download",
      "Download",
      "paper-1",
    );
    expect(plugin.operations.snapshot()).toHaveLength(1);
    operation.finish();
  });

  it("ignores a stale enable-modal result after a newer disabled intent", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.llm.apiKey = "configured";
    settings.arxiv.topics.push({
      id: "topic-1",
      name: "Language models",
      tag: "language-models",
      description: "Language model research",
      detail: false,
    });
    let resolveModal!: (value: string) => void;
    const modal = new Promise<string>((resolve) => { resolveModal = resolve; });
    const persistSettings = vi.fn().mockResolvedValue(undefined);
    const scheduler = { start: vi.fn(), stop: vi.fn(), tickToday: vi.fn() };
    Object.assign(plugin, {
      settings,
      scheduler,
      progress: { setDisabled: vi.fn() },
      stateStore: { setSkipped: vi.fn() },
      logger: { notice: vi.fn() },
    });
    vi.spyOn(plugin as any, "chooseScheduleEnableAction").mockReturnValue(modal);
    plugin.settingsChanges = new SettingsChangeService({
      settings,
      persistSettings,
      setScheduleEnabled: (enabled) => (plugin as any).applyScheduleEnabledRuntime(enabled),
    });

    const enabling = plugin.setScheduleEnabled(true);
    const disabling = plugin.setScheduleEnabled(false);
    await expect(disabling).resolves.toBe(true);
    resolveModal("run");
    await expect(enabling).resolves.toBe(false);

    expect(settings.schedule.enabled).toBe(false);
    expect(persistSettings).not.toHaveBeenCalled();
    expect(scheduler.start).not.toHaveBeenCalled();
    expect(scheduler.tickToday).not.toHaveBeenCalled();
  });

  it("queues the latest disabled intent behind an in-flight enable save and skips the stale run result", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.llm.apiKey = "configured";
    settings.arxiv.topics.push({
      id: "topic-1",
      name: "Language models",
      tag: "language-models",
      description: "Language model research",
      detail: false,
    });
    let finishEnableSave!: () => void;
    const enableSave = new Promise<void>((resolve) => { finishEnableSave = resolve; });
    const persisted: boolean[] = [];
    const scheduler = { start: vi.fn(), stop: vi.fn(), tickToday: vi.fn() };
    Object.assign(plugin, {
      settings,
      scheduler,
      progress: { setDisabled: vi.fn() },
      stateStore: { setSkipped: vi.fn() },
      logger: { notice: vi.fn() },
    });
    vi.spyOn(plugin as any, "chooseScheduleEnableAction").mockResolvedValue("run");
    plugin.settingsChanges = new SettingsChangeService({
      settings,
      persistSettings: vi.fn(async (candidate) => {
        persisted.push(candidate.schedule.enabled);
        if (candidate.schedule.enabled) await enableSave;
      }),
      setScheduleEnabled: (enabled) => (plugin as any).applyScheduleEnabledRuntime(enabled),
    });

    const enabling = plugin.setScheduleEnabled(true);
    await vi.waitFor(() => expect(persisted).toEqual([true]));
    const disabling = plugin.setScheduleEnabled(false);
    finishEnableSave();
    await Promise.all([enabling, disabling]);

    expect(persisted).toEqual([true, false]);
    expect(settings.schedule.enabled).toBe(false);
    expect(scheduler.start).toHaveBeenCalledTimes(1);
    expect(scheduler.stop).toHaveBeenCalledTimes(1);
    expect(scheduler.tickToday).not.toHaveBeenCalled();
  });

  it("does not stop a running scheduler when disabling persistence fails", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.schedule.enabled = true;
    const scheduler = { start: vi.fn(), stop: vi.fn() };
    Object.assign(plugin, { settings, scheduler, progress: { setDisabled: vi.fn() } });
    plugin.settingsChanges = new SettingsChangeService({
      settings,
      persistSettings: vi.fn().mockRejectedValue(new Error("disk full")),
      setScheduleEnabled: (enabled) => (plugin as any).applyScheduleEnabledRuntime(enabled),
    });

    await expect(plugin.setScheduleEnabled(false)).rejects.toThrow("disk full");

    expect(settings.schedule.enabled).toBe(true);
    expect(scheduler.stop).not.toHaveBeenCalled();
    expect(scheduler.start).not.toHaveBeenCalled();
  });

  it("stops a running scheduler only after disabling persistence succeeds", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.schedule.enabled = true;
    const order: string[] = [];
    const scheduler = {
      start: vi.fn(),
      stop: vi.fn(() => order.push("stop")),
    };
    Object.assign(plugin, { settings, scheduler, progress: { setDisabled: vi.fn() } });
    plugin.settingsChanges = new SettingsChangeService({
      settings,
      persistSettings: vi.fn(async () => { order.push("persist"); }),
      setScheduleEnabled: (enabled) => (plugin as any).applyScheduleEnabledRuntime(enabled),
    });

    await expect(plugin.setScheduleEnabled(false)).resolves.toBe(true);

    expect(settings.schedule.enabled).toBe(false);
    expect(order).toEqual(["persist", "stop"]);
    expect(scheduler.start).not.toHaveBeenCalled();
  });

  it("prepares candidate state/history stores and installs both scheduler references", async () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const candidate = structuredClone(DEFAULT_SETTINGS);
    candidate.output.dailyDir = "reports/daily";
    candidate.output.papersDir = "reports/papers";
    const statePath = "reports/.index/run-state.json";
    const historyPath = "reports/.index/run-history.jsonl";
    const storage = memoryStorage({
      [statePath]: JSON.stringify({
        schemaVersion: 1,
        runState: {
          "2026-08-09": {
            status: "completed",
            lastAttempt: 1,
            attempts: 1,
            papersWritten: 2,
          },
        },
      }),
      [historyPath]: `${JSON.stringify({
        schemaVersion: 1,
        at: "2026-08-09T00:00:00.000Z",
        date: "2026-08-09",
        event: "completed",
        trigger: "manual",
      })}\n`,
    });
    const scheduler = {
      replacePersistenceStores: vi.fn(),
    };
    const progress = { setDisabled: vi.fn(), setIdle: vi.fn() };
    Object.assign(plugin, {
      settings: structuredClone(DEFAULT_SETTINGS),
      host: { storage },
      logger: { warn: vi.fn() },
      scheduler,
      progress,
    });

    const prepared = await (plugin as any).prepareOutputStores(candidate);
    expect(prepared.stateStore).toBeInstanceOf(StateStore);
    expect(prepared.runHistoryStore).toBeInstanceOf(RunHistoryStore);
    expect(prepared.stateStore.snapshot()).toHaveProperty("2026-08-09.status", "completed");
    expect(await prepared.runHistoryStore.readLatest(1)).toHaveLength(1);

    (plugin as any).installOutputStores(prepared);
    expect(plugin.stateStore).toBe(prepared.stateStore);
    expect(plugin.runHistoryStore).toBe(prepared.runHistoryStore);
    expect(scheduler.replacePersistenceStores).toHaveBeenCalledWith(
      prepared.stateStore,
      prepared.runHistoryStore,
    );
    expect(progress.setDisabled).toHaveBeenCalledTimes(1);
  });

  it("keeps plugin and scheduler store pairs intact when atomic installation rejects", () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const oldStateStore = { name: "old-state" };
    const oldHistoryStore = { name: "old-history" };
    const nextStateStore = { name: "next-state" };
    const nextHistoryStore = { name: "next-history" };
    let schedulerPair = {
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
    };
    const scheduler = {
      replacePersistenceStores: vi.fn(() => {
        throw new Error("scheduler pair replacement failed");
      }),
    };
    Object.assign(plugin, {
      settings: structuredClone(DEFAULT_SETTINGS),
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
      scheduler,
      progress: { setDisabled: vi.fn(), setIdle: vi.fn() },
    });

    expect(() => (plugin as any).installOutputStores({
      stateStore: nextStateStore,
      runHistoryStore: nextHistoryStore,
    })).toThrow("scheduler pair replacement failed");

    expect(plugin.stateStore).toBe(oldStateStore);
    expect(plugin.runHistoryStore).toBe(oldHistoryStore);
    expect(schedulerPair).toEqual({
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
    });
  });

  it("keeps all output-store references installed when progress refresh throws", () => {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const oldStateStore = { name: "old-state" };
    const oldHistoryStore = { name: "old-history" };
    const nextStateStore = { name: "next-state" };
    const nextHistoryStore = { name: "next-history" };
    let schedulerPair = {
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
    };
    const scheduler = {
      replacePersistenceStores: vi.fn((stateStore, runHistoryStore) => {
        schedulerPair = { stateStore, runHistoryStore };
      }),
    };
    Object.assign(plugin, {
      settings: structuredClone(DEFAULT_SETTINGS),
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
      scheduler,
      progress: {
        setDisabled: vi.fn(() => { throw new Error("progress unavailable"); }),
        setIdle: vi.fn(),
      },
    });

    expect(() => (plugin as any).installOutputStores({
      stateStore: nextStateStore,
      runHistoryStore: nextHistoryStore,
    })).toThrow("progress unavailable");

    expect(plugin.stateStore).toBe(nextStateStore);
    expect(plugin.runHistoryStore).toBe(nextHistoryStore);
    expect(schedulerPair).toEqual({
      stateStore: nextStateStore,
      runHistoryStore: nextHistoryStore,
    });
  });

  it("releases the settings gate and keeps durable paths with recovered stores after scheduler install failure", async () => {
    const plugin = new ArxivDailyPlugin();
    const settings = structuredClone(DEFAULT_SETTINGS);
    const oldStateStore = { name: "old-state" };
    const oldHistoryStore = { name: "old-history" };
    const nextStateStore = { name: "next-state" };
    const nextHistoryStore = { name: "next-history" };
    const schedulerPair = {
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
    };
    const scheduler = {
      activeRuns: vi.fn().mockReturnValue([]),
      replacePersistenceStores: vi.fn(() => {
        throw new Error("scheduler pair replacement failed");
      }),
    };
    const reportPostCommitError = vi.fn();
    Object.assign(plugin, {
      settings,
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
      scheduler,
      progress: { setDisabled: vi.fn(), setIdle: vi.fn() },
    });
    plugin.settingsChanges = new SettingsChangeService({
      settings,
      beginOutputTransition: () => (plugin as any).beginOutputTransition(),
      hasActiveOutputWork: () => plugin.hasActiveOutputWork(),
      prepareOutputStores: vi.fn().mockResolvedValue({
        stateStore: nextStateStore,
        runHistoryStore: nextHistoryStore,
      }),
      installOutputStores: (prepared) => (plugin as any).installOutputStores(prepared),
      persistSettings: vi.fn().mockResolvedValue(undefined),
      reportPostCommitError,
    } as never);

    await expect(
      plugin.settingsChanges.changeValue("output.dailyDir", "reports/daily"),
    ).resolves.toBeUndefined();

    expect(settings.output.dailyDir).toBe("reports/daily");
    expect(plugin.stateStore).toBe(oldStateStore);
    expect(plugin.runHistoryStore).toBe(oldHistoryStore);
    expect(schedulerPair).toEqual({
      stateStore: oldStateStore,
      runHistoryStore: oldHistoryStore,
    });
    expect(reportPostCommitError).toHaveBeenCalledWith(
      "install output stores",
      expect.objectContaining({ message: "scheduler pair replacement failed" }),
    );
    const operation = plugin.operations.begin("pdf-download", "Download", "paper-1");
    operation.finish();
  });

  it("sanitizes detail selection immediately before saving", () => {
    const saveBody = pluginMainSource.match(
      /async saveSettings\(\): Promise<void> \{[\s\S]*?\n  \}/,
    )?.[0];
    expect(saveBody).toBeDefined();
    expect(saveBody!.indexOf("sanitizeDetailSelection")).toBeLessThan(
      saveBody!.indexOf("persistSettings"),
    );
  });

  it("passes configured detail selection policy to the production pipeline", () => {
    const configuredPolicy = {
      profile: "custom" as const,
      normalThreshold: 82,
      exceptionalThreshold: 97,
      softLimit: 6,
    };
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const warn = vi.fn();
    Object.assign(plugin, {
      settings: {
        ...DEFAULT_SETTINGS,
        detailSelection: configuredPolicy,
      },
      logger: { warn },
      progress: {},
      host: { markupParser: {}, storage: {} },
      buildSharedDeps: () => ({
        llm: {},
        fetcher: {},
        paperFetcher: {},
        writer: {},
      }),
      buildPaperIndex: () => ({}),
    });

    const pipeline = (plugin as any).buildPipeline();

    expect((pipeline as any).deps.detailSelection).toEqual(configuredPolicy);
    expect((pipeline as any).deps.detailSelection).not.toBe(
      plugin.settings.detailSelection,
    );
    const checkpointStores = (pipeline as any).deps.checkpointStores;
    expect(checkpointStores.filter).toBeInstanceOf(DailyFilterCheckpointStore);
    expect(checkpointStores.summary).toBeInstanceOf(DailySummaryCheckpointStore);
    const checkpointError = new Error("unreadable checkpoint");
    for (const store of [checkpointStores.filter, checkpointStores.summary]) {
      expect(store.storage).toBe((plugin as any).host.storage);
      expect(store.output).toEqual(plugin.settings.output);
      expect(store.output).not.toBe(plugin.settings.output);
      store.options.onWarning("checkpoint warning", checkpointError);
    }
    expect(warn).toHaveBeenCalledTimes(2);
    expect(warn).toHaveBeenNthCalledWith(
      1,
      "checkpoint warning",
      checkpointError,
    );
    expect(warn).toHaveBeenNthCalledWith(
      2,
      "checkpoint warning",
      checkpointError,
    );
  });

  it("logs persisted sanitation warnings after logger initialization", () => {
    const loggerInit = pluginMainSource.indexOf("this.logger = new Logger(");
    const warningLoop = pluginMainSource.indexOf("for (const warning of settingsWarnings)");
    expect(loggerInit).toBeGreaterThan(-1);
    expect(warningLoop).toBeGreaterThan(loggerInit);
    expect(pluginMainSource).toContain("this.logger.warn(`settings: ${warning}`)");
  });

  it("preserves saved non-default categories exactly across repeated reload migration", () => {
    const saved = persistedData(savedCategories);

    const firstLoad = settingsAndStateFromPersistedData(saved);
    const reloaded = settingsAndStateFromPersistedData({
      settings: firstLoad.settings,
      runState: firstLoad.runState,
    });

    expect(firstLoad.settings.arxiv.categories).toEqual(savedCategories);
    expect(reloaded.settings.arxiv.categories).toEqual(savedCategories);
    expect(saved.settings.arxiv.categories).toEqual(savedCategories);
  });

  it("preserves an explicitly empty category list across reload", () => {
    const firstLoad = settingsAndStateFromPersistedData(persistedData([]));
    const reloaded = settingsAndStateFromPersistedData({ settings: firstLoad.settings });

    expect(firstLoad.settings.arxiv.categories).toEqual([]);
    expect(reloaded.settings.arxiv.categories).toEqual([]);
  });

  it("migrates the legacy singular category only when categories is absent", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        arxiv: {
          category: "hep-th",
          topics: [],
          timezone: "UTC",
        },
      },
    });

    expect(loaded.settings.arxiv.categories).toEqual(["hep-th"]);
  });

  it("adds balanced detail selection settings to old persisted configs", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: { arxiv: persistedData(["cs.CL"]).settings.arxiv },
    });

    expect(loaded.settings.detailSelection).toEqual({
      profile: "balanced",
      normalThreshold: 75,
      exceptionalThreshold: 92,
      softLimit: 3,
    });
  });

  it("adds a disabled local parser sidecar to old persisted configs", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: { arxiv: persistedData(["cs.CL"]).settings.arxiv },
    });

    expect(loaded.settings.pdfParserSidecar).toEqual(DEFAULT_SETTINGS.pdfParserSidecar);
  });

  it("requires a literal persisted true before enabling the local parser sidecar", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        pdfParserSidecar: {
          enabled: "true",
          capabilitiesUrl: "http://127.0.0.1:9000/v1/capabilities",
          parseUrl: "http://127.0.0.1:9000/v1/parse",
        },
      },
    });

    expect(loaded.settings.pdfParserSidecar).toEqual({
      enabled: false,
      capabilitiesUrl: "http://127.0.0.1:9000/v1/capabilities",
      parseUrl: "http://127.0.0.1:9000/v1/parse",
    });
  });

  it("canonicalizes conflicting persisted values under a named profile", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        detailSelection: {
          profile: "conservative",
          normalThreshold: 12,
          exceptionalThreshold: 13,
          softLimit: 20,
        },
      },
    });

    expect(loaded.settings.detailSelection).toEqual({
      profile: "conservative",
      normalThreshold: 85,
      exceptionalThreshold: 95,
      softLimit: 1,
    });
  });

  it("sanitizes detail selection settings across save and reload", () => {
    const firstLoad = settingsAndStateFromPersistedData({
      settings: {
        detailSelection: {
          profile: "custom",
          normalThreshold: 101,
          exceptionalThreshold: 20,
          softLimit: 4.6,
        },
      },
    });
    const reloaded = settingsAndStateFromPersistedData({ settings: firstLoad.settings });

    expect(firstLoad.settings.detailSelection).toEqual({
      profile: "custom",
      normalThreshold: 100,
      exceptionalThreshold: 100,
      softLimit: 5,
    });
    expect(reloaded.settings.detailSelection).toEqual(firstLoad.settings.detailSelection);
  });

  it("preserves canonical and arbitrary-minute run-window values on reload", () => {
    const firstLoad = settingsAndStateFromPersistedData({
      settings: {
        schedule: {
          enabled: false,
          runAtLocal: "09:07",
          runUntilLocal: "18:43",
          tickIntervalMin: 20,
        },
      },
    });
    const reloaded = settingsAndStateFromPersistedData({ settings: firstLoad.settings });

    expect(reloaded.settings.schedule.runAtLocal).toBe("09:07");
    expect(reloaded.settings.schedule.runUntilLocal).toBe("18:43");
  });

  it("sanitizes persisted output paths independently with warnings", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        output: {
          dailyDir: " 研究\\日报 ",
          papersDir: "../outside",
        },
      },
    });

    expect(loaded.settings.output.dailyDir).toBe("研究/日报");
    expect(loaded.settings.output.papersDir).toBe("arxiv-daily/papers");
    expect(loaded.warnings).toEqual([
      expect.stringContaining("output.papersDir"),
    ]);
  });

  it("resets both defaults when persisted output directories collide portably", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        output: { dailyDir: "Café/Notes", papersDir: "CAFE\u0301/notes" },
      },
    });
    expect(loaded.settings.output.dailyDir).toBe("arxiv-daily/daily");
    expect(loaded.settings.output.papersDir).toBe("arxiv-daily/papers");
    expect(loaded.warnings.join(" ")).toMatch(/collided.*both defaults/i);
  });

  it("resets both defaults when an invalid field fallback would collide", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        output: {
          dailyDir: "arxiv-daily/papers",
          papersDir: "../outside",
        },
      },
    });
    expect(loaded.settings.output.dailyDir).toBe("arxiv-daily/daily");
    expect(loaded.settings.output.papersDir).toBe("arxiv-daily/papers");
    expect(loaded.warnings).toHaveLength(2);
  });

  it("falls back from an invalid persisted timezone with a warning", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: { arxiv: { timezone: "Mars/Olympus_Mons" } },
    });

    expect(loaded.settings.arxiv.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);
    expect(loaded.warnings).toEqual([
      expect.stringMatching(/timezone.*restored default/i),
    ]);
    expect(() => new Intl.DateTimeFormat("en-US", {
      timeZone: loaded.settings.arxiv.timezone,
    })).not.toThrow();
  });

  it.each(["", "   ", 42])(
    "warns when an explicitly persisted timezone is invalid (%j)",
    (timezone) => {
      const loaded = settingsAndStateFromPersistedData({
        settings: { arxiv: { timezone } },
      });

      expect(loaded.settings.arxiv.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);
      expect(loaded.warnings).toEqual([
        expect.stringMatching(/timezone.*restored default/i),
      ]);
    },
  );

  it("does not silently rewrite invalid legacy run-window values during load", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        schedule: {
          enabled: false,
          runAtLocal: "24:00",
          runUntilLocal: "legacy-value",
          tickIntervalMin: 20,
        },
      },
    });

    expect(loaded.settings.schedule.runAtLocal).toBe("24:00");
    expect(loaded.settings.schedule.runUntilLocal).toBe("legacy-value");
  });

  it("keeps plugin, scheduler, and history on the old store when reload is rejected during an active pipeline", async () => {
    const oldStore = makeStateStore();
    await oldStore.load();
    const oldHistory = {
      records: [] as any[],
      safeAppend: vi.fn(async (record: any) => {
        oldHistory.records.push(record);
      }),
    };
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    let finishPipeline!: () => void;
    const pipelineCanFinish = new Promise<void>((resolve) => {
      finishPipeline = resolve;
    });
    const scheduler = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store: oldStore,
      lock: new RunLock(),
      logger: new Logger("error"),
      runHistory: oldHistory,
      runForDate: async () => {
        markStarted();
        await pipelineCanFinish;
        return { kind: "completed", papersWritten: 2 };
      },
    });
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    Object.assign(plugin, {
      settings: {
        ...DEFAULT_SETTINGS,
        output: {
          ...DEFAULT_SETTINGS.output,
          dailyDir: "replacement/daily",
          papersDir: "replacement/papers",
        },
      },
      logger: new Logger("error"),
      host: {
        storage: {
          normalizePath: (path: string) => path,
          exists: async () => false,
          readText: async () => "",
          writeText: async () => undefined,
          mkdir: async () => undefined,
          rename: async () => undefined,
          remove: async () => undefined,
        },
      },
      stateStore: oldStore,
      runHistoryStore: oldHistory,
      scheduler,
      progress: { setIdle: vi.fn(), setDisabled: vi.fn() },
    });

    const run = scheduler.runForDateNow("2026-08-10");
    await started;
    await expect(plugin.reloadStateStoreForOutputPaths()).rejects.toThrow(
      "cannot replace scheduler store while work is active",
    );

    expect(plugin.stateStore).toBe(oldStore);
    expect(plugin.runHistoryStore).toBe(oldHistory);

    finishPipeline();
    await expect(run).resolves.toEqual({ kind: "completed", papersWritten: 2 });
    expect(oldStore.get("2026-08-10")).toMatchObject({
      status: "completed",
      papersWritten: 2,
    });
    expect(oldHistory.records.map((record) => record.event)).toEqual([
      "started",
      "completed",
    ]);
  });

  it("keeps plugin, scheduler, and history on the old store when reload is rejected for a pending completion", async () => {
    const data: { runState: Record<string, any> } = { runState: {} };
    let rejectCompletion = true;
    const oldStore = new StateStore(
      async () => ({
        runState: Object.fromEntries(
          Object.entries(data.runState).map(([date, entry]) => [date, { ...entry }]),
        ),
      }),
      async ({ runState }) => {
        if (
          rejectCompletion &&
          Object.values(runState).some((entry) => entry.status === "completed")
        ) {
          throw new Error("completion storage unavailable");
        }
        data.runState = Object.fromEntries(
          Object.entries(runState).map(([date, entry]) => [date, { ...entry }]),
        );
      },
    );
    await oldStore.load();
    const oldHistory = {
      records: [] as any[],
      safeAppend: vi.fn(async (record: any) => {
        oldHistory.records.push(record);
      }),
    };
    const scheduler = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store: oldStore,
      lock: new RunLock(),
      logger: new Logger("error"),
      runHistory: oldHistory,
      runForDate: async () => ({ kind: "completed", papersWritten: 3 }),
    });
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    Object.assign(plugin, {
      settings: {
        ...DEFAULT_SETTINGS,
        output: {
          ...DEFAULT_SETTINGS.output,
          dailyDir: "replacement/daily",
          papersDir: "replacement/papers",
        },
      },
      logger: new Logger("error"),
      host: {
        storage: {
          normalizePath: (path: string) => path,
          exists: async () => false,
          readText: async () => "",
          writeText: async () => undefined,
          mkdir: async () => undefined,
          rename: async () => undefined,
          remove: async () => undefined,
        },
      },
      stateStore: oldStore,
      runHistoryStore: oldHistory,
      scheduler,
      progress: { setIdle: vi.fn(), setDisabled: vi.fn() },
    });

    await expect(scheduler.runForDateNow("2026-08-10")).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });
    await expect(plugin.reloadStateStoreForOutputPaths()).rejects.toThrow(
      "cannot replace scheduler store while work is active",
    );

    expect(plugin.stateStore).toBe(oldStore);
    expect(plugin.runHistoryStore).toBe(oldHistory);

    rejectCompletion = false;
    await expect(scheduler.runForDateNow("2026-08-10")).resolves.toEqual({
      kind: "completed",
      papersWritten: 3,
    });
    expect(oldStore.get("2026-08-10")).toMatchObject({
      status: "completed",
      papersWritten: 3,
    });
    expect(oldHistory.records.at(-1)?.event).toBe("completed");
  });
});
