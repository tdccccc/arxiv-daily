import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS, type StorageAdapter } from "@arxiv-daily/core";
import * as obsidianHost from "../src/hosts/obsidian";
import ArxivDailyPlugin from "../main.ts";
import { SettingsChangeError } from "../src/settings/change-service";

const PLUGIN_DIR = ".obsidian/plugins/arxiv-daily";
const DATA_JSON = `${PLUGIN_DIR}/data.json`;
const OLD_FILE = JSON.stringify({ settings: { llm: { model: "old-model" } } }, undefined, 2);

/**
 * Vault.writeJson as Obsidian ships it: the adapter write is wrapped in a
 * try/catch that eats the error, so saveData resolves whatever happened.
 */
function obsidianSwallowingSaveData(adapter: { write(path: string, text: string): Promise<void> }) {
  return async (data: unknown): Promise<void> => {
    try {
      await adapter.write(DATA_JSON, JSON.stringify(data, undefined, 2));
    } catch {
      // swallowed by the host
    }
  };
}

/** A vault adapter over one file; `failWith` makes every write fail before touching it. */
function vaultAdapter(failWith?: Error) {
  const files = new Map([[DATA_JSON, OLD_FILE]]);
  const adapter = {
    write: vi.fn(async (path: string, text: string) => {
      if (failWith) throw failWith;
      files.set(path, text);
    }),
  };
  return { files, adapter };
}

async function pluginOn(adapter: unknown) {
  const plugin = new ArxivDailyPlugin();
  const storage: StorageAdapter = {
    normalizePath: (path) => path,
    readText: async () => "",
    writeText: async () => {},
    exists: async () => false,
    mkdir: async () => {},
    remove: async () => {},
    rename: async () => {},
    writeTextAtomic: async () => {},
    acquireLock: vi.fn(async () => null),
    list: vi.fn(async () => []),
  } as StorageAdapter;
  const buildHost = vi.spyOn(obsidianHost, "buildObsidianHostAdapters")
    .mockReturnValue({ storage, http: {}, markupParser: {} } as any);
  Object.assign(plugin, {
    settings: structuredClone(DEFAULT_SETTINGS),
    app: { vault: { configDir: ".obsidian", adapter } },
    manifest: { id: "arxiv-daily", dir: PLUGIN_DIR },
    loadSettingsAndState: vi.fn(async () => []),
    cleanupCachesIfDue: vi.fn(),
    addStatusBarItem: () => document.createElement("div"),
    addSettingTab: () => { throw new Error("stop after settingsChanges is wired"); },
  });
  await expect(plugin.onload()).rejects.toThrow("stop after settingsChanges is wired");
  return { plugin, dispose: () => { plugin.onunload(); buildHost.mockRestore(); } };
}

describe("saving settings when the disk write fails", () => {
  it("documents the host behaviour this guards against", async () => {
    const { files, adapter } = vaultAdapter(new Error("ENOSPC: no space left on device"));

    await expect(obsidianSwallowingSaveData(adapter)({ settings: {} })).resolves.toBeUndefined();

    expect(files.get(DATA_JSON)).toBe(OLD_FILE);
  });

  it("rejects, rolls the setting back in memory and leaves the old file alone", async () => {
    const full = new Error("ENOSPC: no space left on device");
    const { files, adapter } = vaultAdapter(full);
    const { plugin, dispose } = await pluginOn(adapter);
    try {
      const before = plugin.settings.llm.baseUrl;

      const change = plugin.settingsChanges.changeValue("llm.baseUrl", "https://new.example/v1");

      await expect(change).rejects.toBeInstanceOf(SettingsChangeError);
      await expect(change).rejects.toThrow("ENOSPC");
      expect(plugin.settings.llm.baseUrl).toBe(before);
      expect(files.get(DATA_JSON)).toBe(OLD_FILE);
      expect(adapter.write).toHaveBeenCalledTimes(1);
    } finally {
      dispose();
    }
  });

  it("rejects an API key change the same way, keeping the old key", async () => {
    const { adapter } = vaultAdapter(new Error("EACCES: permission denied"));
    const { plugin, dispose } = await pluginOn(adapter);
    try {
      plugin.settings.llm.apiKey = "old-key";

      await expect(plugin.settingsChanges.changeValue("llm.apiKey", "new-key"))
        .rejects.toThrow("EACCES");

      expect(plugin.settings.llm.apiKey).toBe("old-key");
    } finally {
      dispose();
    }
  });

  it("surfaces the failure to a direct persistSettings caller", async () => {
    const full = new Error("EIO");
    const { adapter } = vaultAdapter(full);
    const { plugin, dispose } = await pluginOn(adapter);
    try {
      await expect((plugin as any).persistSettings()).rejects.toBe(full);
    } finally {
      dispose();
    }
  });

  it("still writes the same JSON text when the disk is fine", async () => {
    const { files, adapter } = vaultAdapter();
    const { plugin, dispose } = await pluginOn(adapter);
    try {
      await plugin.settingsChanges.changeValue("llm.baseUrl", "https://new.example/v1");

      const written = JSON.parse(files.get(DATA_JSON)!);
      expect(written.settings.llm.baseUrl).toBe("https://new.example/v1");
      expect(files.get(DATA_JSON)).toBe(JSON.stringify(written, undefined, 2));
      expect(adapter.write).toHaveBeenCalledWith(DATA_JSON, expect.any(String));
    } finally {
      dispose();
    }
  });
});
