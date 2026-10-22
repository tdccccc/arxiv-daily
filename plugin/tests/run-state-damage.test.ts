import { describe, expect, it, vi } from "vitest";
import { Notice } from "obsidian";
import { DEFAULT_SETTINGS, type StorageAdapter } from "@arxiv-daily/core";
import * as obsidianHost from "../src/hosts/obsidian";
import ArxivDailyPlugin from "../main.ts";

const RUN_STATE = "arxiv-daily/.index/run-state.json";

function memoryStorage(files: Record<string, string>): StorageAdapter {
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
    rename: async (from, to) => { data.set(to, data.get(from)!); data.delete(from); },
    writeTextAtomic: async () => {},
    acquireLock: vi.fn(async () => null),
    list: vi.fn(async () => []),
  };
}

/** Runs onload far enough to load run state; it stops when the settings tab is registered. */
async function loadPlugin(files: Record<string, string>) {
  const plugin = new ArxivDailyPlugin();
  const buildHost = vi.spyOn(obsidianHost, "buildObsidianHostAdapters")
    .mockReturnValue({ storage: memoryStorage(files), http: {}, markupParser: {} } as never);
  Object.assign(plugin, {
    settings: structuredClone(DEFAULT_SETTINGS),
    app: {},
    manifest: { id: "arxiv-daily" },
    loadSettingsAndState: vi.fn(async () => []),
    cleanupCachesIfDue: vi.fn(),
    saveData: vi.fn(),
    addStatusBarItem: () => document.createElement("div"),
    addSettingTab: () => { throw new Error("stop after run state is loaded"); },
  });
  Notice.calls = [];
  await expect(plugin.onload()).rejects.toThrow("stop after run state is loaded");
  return { plugin, dispose: () => { plugin.onunload(); buildHost.mockRestore(); } };
}

const damageNotices = () => Notice.calls.filter(call => call.message.includes("run-state.json is damaged"));

describe("a damaged run-state.json at startup", () => {
  it("tells the user the empty history is not real and how to reset it", async () => {
    const { plugin, dispose } = await loadPlugin({ [RUN_STATE]: "<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> other\n" });
    try {
      expect(plugin.stateStore.damage()?.path).toBe(RUN_STATE);
      expect(damageNotices()).toHaveLength(1);
      expect(damageNotices()[0]!.message).toContain("Clear arXiv Daily run state");
    } finally {
      dispose();
    }
  });

  it("stays quiet when there is no run state yet or it is healthy", async () => {
    for (const files of [{}, { [RUN_STATE]: JSON.stringify({ schemaVersion: 1, runState: {} }) }]) {
      const { plugin, dispose } = await loadPlugin(files);
      try {
        expect(plugin.stateStore.damage()).toBeUndefined();
        expect(damageNotices()).toEqual([]);
      } finally {
        dispose();
      }
    }
  });
});
