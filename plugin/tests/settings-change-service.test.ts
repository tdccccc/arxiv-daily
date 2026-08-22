import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import {
  SettingsChangeError,
  SettingsChangeService,
} from "../src/settings/change-service";

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function makeSettings() {
  return structuredClone(DEFAULT_SETTINGS);
}

describe("SettingsChangeService", () => {
  it("keeps live settings and runtime untouched when persistence fails", async () => {
    const settings = makeSettings();
    const rootIdentity = settings;
    const nestedIdentities = Object.values(settings);
    const setLoggerLevel = vi.fn();
    const refreshSensitiveValues = vi.fn();
    const service = new SettingsChangeService({
      settings,
      persistSettings: vi.fn().mockRejectedValue(new Error("disk full")),
      setLoggerLevel,
      refreshSensitiveValues,
    });

    const failure = await service
      .changeValue("advanced.logLevel", "debug")
      .catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(SettingsChangeError);
    expect((failure as SettingsChangeError).restoreValue("advanced.logLevel")).toBe("info");
    expect(settings).toBe(rootIdentity);
    expect(Object.values(settings)).toEqual(nestedIdentities);
    expect(settings.advanced.logLevel).toBe("info");
    expect(setLoggerLevel).not.toHaveBeenCalled();
    expect(refreshSensitiveValues).not.toHaveBeenCalled();
  });

  it("validates daily and papers directories together before preparing or persisting", async () => {
    const settings = makeSettings();
    const prepareOutputStores = vi.fn();
    const persistSettings = vi.fn();
    const service = new SettingsChangeService({
      settings,
      persistSettings,
      prepareOutputStores,
    });

    await expect(
      service.changeValue("output.dailyDir", "ARXIV-DAILY\\PAPERS"),
    ).rejects.toThrow(/daily and papers directories/i);

    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(prepareOutputStores).not.toHaveBeenCalled();
    expect(persistSettings).not.toHaveBeenCalled();
  });

  it("rejects a non-loopback local parser sidecar before persistence", async () => {
    const settings = makeSettings();
    const persistSettings = vi.fn();
    const service = new SettingsChangeService({ settings, persistSettings });

    await expect(service.change({
      changes: [
        { key: "pdfParserSidecar.enabled", value: true },
        { key: "pdfParserSidecar.capabilitiesUrl", value: "https://parser.example/v1/capabilities" },
        { key: "pdfParserSidecar.parseUrl", value: "https://parser.example/v1/parse" },
      ],
    })).rejects.toThrow(/loopback/i);

    expect(settings.pdfParserSidecar).toEqual(DEFAULT_SETTINGS.pdfParserSidecar);
    expect(persistSettings).not.toHaveBeenCalled();
  });

  it("leaves stores, settings, and persistence untouched when prepared store loading fails", async () => {
    const settings = makeSettings();
    const oldStateStore = { name: "old-state" };
    const oldHistoryStore = { name: "old-history" };
    const installed = {
      pluginState: oldStateStore,
      pluginHistory: oldHistoryStore,
      schedulerState: oldStateStore,
      schedulerHistory: oldHistoryStore,
    };
    const installOutputStores = vi.fn();
    const persistSettings = vi.fn();
    const prepareOutputStores = vi.fn().mockRejectedValue(
      new Error("candidate state store load failed"),
    );
    const service = new SettingsChangeService({
      settings,
      persistSettings,
      prepareOutputStores,
      installOutputStores,
    });

    await expect(
      service.changeValue("output.dailyDir", "reports/daily"),
    ).rejects.toThrow("candidate state store load failed");

    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(installed).toEqual({
      pluginState: oldStateStore,
      pluginHistory: oldHistoryStore,
      schedulerState: oldStateStore,
      schedulerHistory: oldHistoryStore,
    });
    expect(persistSettings).not.toHaveBeenCalled();
    expect(installOutputStores).not.toHaveBeenCalled();
  });

  it("does not install prepared stores when candidate persistence fails", async () => {
    const settings = makeSettings();
    const nextStateStore = { name: "next-state" };
    const nextHistoryStore = { name: "next-history" };
    const installOutputStores = vi.fn();
    const service = new SettingsChangeService({
      settings,
      persistSettings: vi.fn().mockRejectedValue(new Error("disk full")),
      prepareOutputStores: vi.fn().mockResolvedValue({
        stateStore: nextStateStore,
        runHistoryStore: nextHistoryStore,
      }),
      installOutputStores,
    } as never);

    await expect(
      service.changeValue("output.dailyDir", "reports/daily"),
    ).rejects.toThrow("disk full");

    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(installOutputStores).not.toHaveBeenCalled();
  });

  it("persists the candidate before committing in place and installing both plugin and scheduler stores", async () => {
    const settings = makeSettings();
    const rootIdentity = settings;
    const outputIdentity = settings.output;
    const oldStateStore = { name: "old-state" };
    const oldHistoryStore = { name: "old-history" };
    const nextStateStore = { name: "next-state" };
    const nextHistoryStore = { name: "next-history" };
    const installed = {
      pluginState: oldStateStore,
      pluginHistory: oldHistoryStore,
      schedulerState: oldStateStore,
      schedulerHistory: oldHistoryStore,
    };
    const order: string[] = [];
    const service = new SettingsChangeService({
      settings,
      persistSettings: vi.fn(async (candidate) => {
        order.push("persist");
        expect(candidate.output.dailyDir).toBe("reports/daily");
        expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
      }),
      prepareOutputStores: vi.fn(async () => {
        order.push("prepare");
        return {
          stateStore: nextStateStore,
          runHistoryStore: nextHistoryStore,
        } as never;
      }),
      installOutputStores: vi.fn((prepared) => {
        order.push("install");
        installed.pluginState = prepared.stateStore as typeof nextStateStore;
        installed.pluginHistory = prepared.runHistoryStore as typeof nextHistoryStore;
        installed.schedulerState = prepared.stateStore as typeof nextStateStore;
        installed.schedulerHistory = prepared.runHistoryStore as typeof nextHistoryStore;
      }),
    });

    await service.changeValue("output.dailyDir", " reports\\daily ");

    expect(order).toEqual(["prepare", "persist", "install"]);
    expect(settings).toBe(rootIdentity);
    expect(settings.output).toBe(outputIdentity);
    expect(settings.output.dailyDir).toBe("reports/daily");
    expect(installed).toEqual({
      pluginState: nextStateStore,
      pluginHistory: nextHistoryStore,
      schedulerState: nextStateStore,
      schedulerHistory: nextHistoryStore,
    });
  });

  it("rejects output-root changes while relevant operations or runs are active", async () => {
    const settings = makeSettings();
    const persistSettings = vi.fn();
    const prepareOutputStores = vi.fn();
    const service = new SettingsChangeService({
      settings,
      persistSettings,
      prepareOutputStores,
      hasActiveOutputWork: () => true,
    });

    await expect(
      service.changeValue("output.papersDir", "reports/papers"),
    ).rejects.toThrow(/active/i);

    expect(settings.output.papersDir).toBe(DEFAULT_SETTINGS.output.papersDir);
    expect(prepareOutputStores).not.toHaveBeenCalled();
    expect(persistSettings).not.toHaveBeenCalled();
  });

  it("serializes concurrent changes and builds each candidate from the last commit", async () => {
    const settings = makeSettings();
    const firstSave = deferred();
    const persisted: Array<{ model: string; baseUrl: string }> = [];
    const persistSettings = vi.fn(async (candidate) => {
      persisted.push({
        model: candidate.llm.model,
        baseUrl: candidate.llm.baseUrl,
      });
      if (persisted.length === 1) await firstSave.promise;
    });
    const service = new SettingsChangeService({ settings, persistSettings });

    const first = service.changeValue("llm.model", "first-model");
    const second = service.changeValue("llm.baseUrl", "https://second.example/v1");
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    expect(settings.llm.model).toBe(DEFAULT_SETTINGS.llm.model);

    firstSave.resolve();
    await Promise.all([first, second]);

    expect(persisted).toEqual([
      {
        model: "first-model",
        baseUrl: DEFAULT_SETTINGS.llm.baseUrl,
      },
      {
        model: "first-model",
        baseUrl: "https://second.example/v1",
      },
    ]);
    expect(settings.llm.model).toBe("first-model");
    expect(settings.llm.baseUrl).toBe("https://second.example/v1");
  });

  it("applies logger, timezone, timer, and schedule effects only after persistence commits", async () => {
    const settings = makeSettings();
    settings.llm.apiKey = "configured";
    settings.arxiv.topics.push({
      id: "topic-1",
      name: "Language models",
      tag: "language-models",
      description: "Research about language models",
      detail: false,
    });
    const save = deferred();
    const effects: string[] = [];
    const service = new SettingsChangeService({
      settings,
      persistSettings: vi.fn(async () => {
        effects.push("persist-start");
        await save.promise;
        effects.push("persist-done");
      }),
      setLoggerLevel: (level) => effects.push(`level:${level}`),
      setLoggerTimezone: (timezone) => effects.push(`timezone:${timezone}`),
      restartScheduler: () => effects.push("restart"),
      setScheduleEnabled: (enabled) => effects.push(`enabled:${enabled}`),
    });

    const changing = service.change({
      changes: [
        { key: "advanced.logLevel", value: "debug" },
        { key: "arxiv.timezone", value: "UTC" },
        { key: "schedule.tickIntervalMin", value: 5 },
        { key: "schedule.enabled", value: true },
      ],
    });
    await vi.waitFor(() => expect(effects).toEqual(["persist-start"]));
    expect(settings.advanced.logLevel).toBe("info");
    expect(settings.arxiv.timezone).toBe("Asia/Shanghai");
    expect(settings.schedule.tickIntervalMin).toBe(20);
    expect(settings.schedule.enabled).toBe(false);

    save.resolve();
    await changing;

    expect(effects).toEqual([
      "persist-start",
      "persist-done",
      "level:debug",
      "timezone:UTC",
      "restart",
      "enabled:true",
    ]);
  });

  it("rejects an invalid timezone without persistence", async () => {
    const settings = makeSettings();
    const persistSettings = vi.fn();
    const service = new SettingsChangeService({ settings, persistSettings });

    await expect(
      service.changeValue("arxiv.timezone", "Mars/Olympus_Mons"),
    ).rejects.toThrow(/timezone/i);

    expect(settings.arxiv.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);
    expect(persistSettings).not.toHaveBeenCalled();
  });

  it("rechecks active work after preparing output stores and rejects before persistence", async () => {
    const settings = makeSettings();
    let active = false;
    const persistSettings = vi.fn();
    const installOutputStores = vi.fn();
    const service = new SettingsChangeService({
      settings,
      persistSettings,
      hasActiveOutputWork: () => active,
      prepareOutputStores: vi.fn(async () => {
        active = true;
        return {
          stateStore: { name: "candidate-state" },
          runHistoryStore: { name: "candidate-history" },
        } as never;
      }),
      installOutputStores,
    });

    await expect(
      service.changeValue("output.dailyDir", "reports/daily"),
    ).rejects.toThrow(/active/i);

    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(persistSettings).not.toHaveBeenCalled();
    expect(installOutputStores).not.toHaveBeenCalled();
  });

  it("preserves every existing object and array identity on scalar commit", async () => {
    const settings = makeSettings();
    settings.arxiv.topics.push({
      id: "topic-1",
      name: "Language models",
      tag: "language-models",
      description: "Research about language models",
      detail: false,
    });
    const identities = {
      root: settings,
      llm: settings.llm,
      arxiv: settings.arxiv,
      categories: settings.arxiv.categories,
      topics: settings.arxiv.topics,
      topic: settings.arxiv.topics[0],
      detailSelection: settings.detailSelection,
      output: settings.output,
      schedule: settings.schedule,
      advanced: settings.advanced,
      email: settings.email,
    };
    const service = new SettingsChangeService({
      settings,
      persistSettings: vi.fn().mockResolvedValue(undefined),
    });

    await service.changeValue("llm.model", "candidate-model");

    expect(settings).toBe(identities.root);
    expect(settings.llm).toBe(identities.llm);
    expect(settings.arxiv).toBe(identities.arxiv);
    expect(settings.arxiv.categories).toBe(identities.categories);
    expect(settings.arxiv.topics).toBe(identities.topics);
    expect(settings.arxiv.topics[0]).toBe(identities.topic);
    expect(settings.detailSelection).toBe(identities.detailSelection);
    expect(settings.output).toBe(identities.output);
    expect(settings.schedule).toBe(identities.schedule);
    expect(settings.advanced).toBe(identities.advanced);
    expect(settings.email).toBe(identities.email);
  });

  it("does not overwrite an unrelated complex-editor mutation made while persisting", async () => {
    const settings = makeSettings();
    settings.arxiv.topics.push({
      id: "topic-1",
      name: "Original topic",
      tag: "original-topic",
      description: "Original description",
      detail: false,
    });
    const save = deferred();
    const persisted: Array<{ model: string; topicName: string }> = [];
    const persistSettings = vi.fn(async (candidate) => {
      persisted.push({
        model: candidate.llm.model,
        topicName: candidate.arxiv.topics[0]!.name,
      });
      if (persisted.length === 1) await save.promise;
    });
    const service = new SettingsChangeService({ settings, persistSettings });

    const changing = service.changeValue("llm.model", "candidate-model");
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    settings.arxiv.topics[0]!.name = "Concurrent topic draft";
    const savingDraft = service.persistCurrent();
    save.resolve();
    await Promise.all([changing, savingDraft]);

    expect(settings.llm.model).toBe("candidate-model");
    expect(settings.arxiv.topics[0]!.name).toBe("Concurrent topic draft");
    expect(persisted).toEqual([
      { model: "candidate-model", topicName: "Original topic" },
      { model: "candidate-model", topicName: "Concurrent topic draft" },
    ]);
  });

  it("holds an output transition gate through prepare, persistence, and install", async () => {
    const settings = makeSettings();
    const save = deferred();
    let transitionActive = false;
    const order: string[] = [];
    const service = new SettingsChangeService({
      settings,
      beginOutputTransition: () => {
        expect(transitionActive).toBe(false);
        transitionActive = true;
        order.push("begin");
        return () => {
          transitionActive = false;
          order.push("release");
        };
      },
      prepareOutputStores: vi.fn(async () => {
        expect(transitionActive).toBe(true);
        order.push("prepare");
        return {
          stateStore: { name: "candidate-state" },
          runHistoryStore: { name: "candidate-history" },
        } as never;
      }),
      persistSettings: vi.fn(async () => {
        expect(transitionActive).toBe(true);
        order.push("persist");
        await save.promise;
      }),
      installOutputStores: vi.fn(() => {
        expect(transitionActive).toBe(true);
        order.push("install");
      }),
    } as never);

    const changing = service.changeValue("output.dailyDir", "reports/daily");
    await vi.waitFor(() => expect(order).toContain("persist"));
    expect(transitionActive).toBe(true);
    save.resolve();
    await changing;

    expect(transitionActive).toBe(false);
    expect(order).toEqual(["begin", "prepare", "persist", "install", "release"]);
  });

  it("releases the output transition gate when candidate persistence fails", async () => {
    const settings = makeSettings();
    const release = vi.fn();
    const installOutputStores = vi.fn();
    const service = new SettingsChangeService({
      settings,
      beginOutputTransition: () => release,
      prepareOutputStores: vi.fn().mockResolvedValue({
        stateStore: { name: "candidate-state" },
        runHistoryStore: { name: "candidate-history" },
      }),
      persistSettings: vi.fn().mockRejectedValue(new Error("disk full")),
      installOutputStores,
    } as never);

    await expect(
      service.changeValue("output.dailyDir", "reports/daily"),
    ).rejects.toThrow("disk full");

    expect(release).toHaveBeenCalledTimes(1);
    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(installOutputStores).not.toHaveBeenCalled();
  });

  it.each(["before", "after"] as const)(
    "rejects a throwing live setter (%s assignment) before persistence or candidate installation",
    async (throwTiming) => {
      const settings = makeSettings();
      let dailyDir = settings.output.dailyDir;
      const setter = vi.fn((value: string) => {
        if (throwTiming === "after") dailyDir = value;
        throw new Error(`live setter threw ${throwTiming} assignment`);
      });
      Object.defineProperty(settings.output, "dailyDir", {
        configurable: true,
        enumerable: true,
        get: () => dailyDir,
        set: setter,
      });
      const release = vi.fn();
      const persistSettings = vi.fn();
      const installOutputStores = vi.fn();
      const reportPostCommitError = vi.fn();
      const service = new SettingsChangeService({
        settings,
        beginOutputTransition: () => release,
        prepareOutputStores: vi.fn().mockResolvedValue({
          stateStore: { name: "candidate-state" },
          runHistoryStore: { name: "candidate-history" },
        }),
        installOutputStores,
        persistSettings,
        reportPostCommitError,
      } as never);

      await expect(
        service.changeValue("output.dailyDir", "reports/daily"),
      ).rejects.toThrow(/live settings.*accessor/i);

      expect(setter).not.toHaveBeenCalled();
      expect(dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
      expect(persistSettings).not.toHaveBeenCalled();
      expect(installOutputStores).not.toHaveBeenCalled();
      expect(reportPostCommitError).not.toHaveBeenCalled();
      expect(release).not.toHaveBeenCalled();
    },
  );

  it("does not install candidate stores or runtime effects when the durable live commit throws", async () => {
    const settings = makeSettings();
    const originalOutput = settings.output;
    settings.output = new Proxy(originalOutput, {
      set(target, key, value) {
        if (key === "dailyDir") throw new Error("live commit failed");
        return Reflect.set(target, key, value);
      },
    });
    const release = vi.fn();
    const installOutputStores = vi.fn();
    const refreshSensitiveValues = vi.fn();
    const setLoggerLevel = vi.fn();
    const reportPostCommitError = vi.fn();
    const service = new SettingsChangeService({
      settings,
      beginOutputTransition: () => release,
      prepareOutputStores: vi.fn().mockResolvedValue({
        stateStore: { name: "candidate-state" },
        runHistoryStore: { name: "candidate-history" },
      }),
      persistSettings: vi.fn().mockResolvedValue(undefined),
      installOutputStores,
      refreshSensitiveValues,
      setLoggerLevel,
      reportPostCommitError,
    } as never);

    await expect(service.change({
      changes: [
        { key: "output.dailyDir", value: "reports/daily" },
        { key: "advanced.logLevel", value: "debug" },
      ],
    })).resolves.toBeUndefined();

    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(settings.advanced.logLevel).toBe("info");
    expect(installOutputStores).not.toHaveBeenCalled();
    expect(refreshSensitiveValues).not.toHaveBeenCalled();
    expect(setLoggerLevel).not.toHaveBeenCalled();
    expect(reportPostCommitError).toHaveBeenCalledWith(
      "commit live settings",
      expect.objectContaining({ message: "live commit failed" }),
    );
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("releases the output transition exactly once when store installation throws", async () => {
    const settings = makeSettings();
    const release = vi.fn();
    const reportPostCommitError = vi.fn();
    const service = new SettingsChangeService({
      settings,
      beginOutputTransition: () => release,
      prepareOutputStores: vi.fn().mockResolvedValue({
        stateStore: { name: "candidate-state" },
        runHistoryStore: { name: "candidate-history" },
      }),
      installOutputStores: vi.fn(() => { throw new Error("install failed"); }),
      persistSettings: vi.fn().mockResolvedValue(undefined),
      reportPostCommitError,
    } as never);

    await expect(
      service.changeValue("output.dailyDir", "reports/daily"),
    ).resolves.toBeUndefined();

    expect(settings.output.dailyDir).toBe("reports/daily");
    expect(reportPostCommitError).toHaveBeenCalledWith(
      "install output stores",
      expect.objectContaining({ message: "install failed" }),
    );
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("releases the output transition when post-commit install and reporting both throw", async () => {
    const settings = makeSettings();
    const release = vi.fn();
    const reportPostCommitError = vi.fn(() => { throw new Error("report failed"); });
    const service = new SettingsChangeService({
      settings,
      beginOutputTransition: () => release,
      prepareOutputStores: vi.fn().mockResolvedValue({
        stateStore: { name: "candidate-state" },
        runHistoryStore: { name: "candidate-history" },
      }),
      installOutputStores: vi.fn(),
      persistSettings: vi.fn().mockResolvedValue(undefined),
      reportPostCommitError,
    } as never);

    await expect(service.change({
      changes: [{ key: "output.dailyDir", value: "reports/daily" }],
      install: () => { throw new Error("custom install failed"); },
    })).resolves.toBeUndefined();

    expect(settings.output.dailyDir).toBe("reports/daily");
    expect(reportPostCommitError).toHaveBeenCalledWith(
      "install prepared settings",
      expect.objectContaining({ message: "custom install failed" }),
    );
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("reports post-commit runtime failures without exposing a rollback error", async () => {
    const settings = makeSettings();
    const reportPostCommitError = vi.fn();
    const service = new SettingsChangeService({
      settings,
      persistSettings: vi.fn().mockResolvedValue(undefined),
      setLoggerLevel: () => { throw new Error("logger unavailable"); },
      reportPostCommitError,
    } as never);

    await expect(
      service.changeValue("advanced.logLevel", "debug"),
    ).resolves.toBeUndefined();

    expect(settings.advanced.logLevel).toBe("debug");
    expect(reportPostCommitError).toHaveBeenCalledWith(
      "apply logger level",
      expect.objectContaining({ message: "logger unavailable" }),
    );
  });

  it("continues processing queued changes after a rejected request", async () => {
    const settings = makeSettings();
    const persistSettings = vi.fn()
      .mockRejectedValueOnce(new Error("disk full"))
      .mockResolvedValueOnce(undefined);
    const service = new SettingsChangeService({ settings, persistSettings });

    const first = service.changeValue("llm.model", "rejected-model");
    const second = service.changeValue("llm.baseUrl", "https://accepted.example/v1");

    await expect(first).rejects.toThrow("disk full");
    await expect(second).resolves.toBeUndefined();
    expect(settings.llm.model).toBe(DEFAULT_SETTINGS.llm.model);
    expect(settings.llm.baseUrl).toBe("https://accepted.example/v1");
    expect(persistSettings).toHaveBeenCalledTimes(2);
  });
});
