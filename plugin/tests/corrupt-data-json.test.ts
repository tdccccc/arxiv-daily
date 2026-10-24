import { beforeEach, describe, expect, it, vi } from "vitest";
import { Notice } from "obsidian";
import { DEFAULT_SETTINGS, type StorageAdapter } from "@arxiv-daily/core";
import * as obsidianHost from "../src/hosts/obsidian";
import ArxivDailyPlugin from "../main.ts";

const PLUGIN_DIR = ".obsidian/plugins/arxiv-daily";
const DATA_JSON = `${PLUGIN_DIR}/data.json`;
const HALF_WRITTEN = '{"settings":{"llm":{"apiKey":"sk-secret","model":"m"},"arxiv":{"cat';

/** The modified time the folder reports for data.json, so a copy's name can be predicted. */
const WRITTEN_AT = Date.UTC(2026, 9, 21, 8, 30, 15, 250);
const COPY_NAME = `${DATA_JSON}.corrupt-2026-10-21T08-30-15-250Z`;

const fsError = (code: string) => Object.assign(new Error(`${code}: data.json`), { code });

/** A plugin folder as the vault adapter sees it. */
function pluginFolder(files: Record<string, string>) {
  const data = new Map(Object.entries(files));
  const adapter = {
    stat: vi.fn(async (path: string) => (
      data.has(path) ? { type: "file", ctime: WRITTEN_AT, mtime: WRITTEN_AT, size: data.get(path)!.length } : null
    )),
    read: vi.fn(async (path: string) => {
      const value = data.get(path);
      if (value === undefined) throw fsError("ENOENT");
      return value;
    }),
    write: vi.fn(async (path: string, content: string) => { data.set(path, content); }),
    exists: vi.fn(async (path: string) => data.has(path)),
  };
  return { data, adapter };
}
type Adapter = ReturnType<typeof pluginFolder>["adapter"];

/**
 * Plugin.loadData as Obsidian ships it (Vault.readJson): the parsed file, null
 * when there is no file, and undefined, never a rejection, for a file it could
 * not read or parse.
 */
function obsidianLoadData(adapter: Adapter) {
  return async (): Promise<unknown> => {
    try {
      return JSON.parse(await adapter.read(DATA_JSON));
    } catch (error) {
      if ((error as { code?: string }).code === "ENOENT") return null;
      return undefined;
    }
  };
}

function pluginOn(adapter: Adapter, loadData: () => Promise<unknown> = obsidianLoadData(adapter)) {
  const plugin = new ArxivDailyPlugin();
  Object.assign(plugin, {
    app: { vault: { configDir: ".obsidian", adapter } },
    manifest: { id: "arxiv-daily", dir: PLUGIN_DIR },
    loadData,
  });
  return plugin as ArxivDailyPlugin & { loadSettingsAndState(): Promise<string[]> };
}

const asideFiles = (data: Map<string, string>) =>
  [...data.keys()].filter((path) => path.startsWith(`${DATA_JSON}.corrupt-`));
const notices = () => Notice.calls.map((call) => call.message);

beforeEach(() => {
  Notice.calls = [];
});

describe("a data.json that Obsidian could not parse", () => {
  it("starts on default settings and keeps the file beside the original", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });
    const plugin = pluginOn(adapter);
    // What Obsidian does with this file: no error, just undefined.
    await expect(obsidianLoadData(adapter)()).resolves.toBeUndefined();

    const warnings = await plugin.loadSettingsAndState();

    expect(plugin.settings).toEqual(expect.objectContaining({
      schedule: DEFAULT_SETTINGS.schedule,
      output: DEFAULT_SETTINGS.output,
    }));
    expect(plugin.settings.llm.apiKey).toBe("");
    expect(asideFiles(data)).toEqual([COPY_NAME]);
    expect(data.get(COPY_NAME)).toBe(HALF_WRITTEN);
    expect(data.get(DATA_JSON)).toBe(HALF_WRITTEN);
    expect(warnings.join("\n")).toContain("data.json.corrupt-2026-10-21T08-30-15-250Z");
  });

  it("tells the user the settings were not loaded and where the file went", async () => {
    const { adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });

    await pluginOn(adapter).loadSettingsAndState();

    expect(notices()).toHaveLength(1);
    expect(notices()[0]).toContain("default settings");
    expect(notices()[0]).toContain(COPY_NAME);
    expect(notices()[0]).not.toContain("sk-secret");
  });

  it("finds its copy again when started again without the file being fixed", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });

    await pluginOn(adapter).loadSettingsAndState();
    adapter.write.mockClear();
    await pluginOn(adapter).loadSettingsAndState();

    expect(asideFiles(data)).toEqual([COPY_NAME]);
    expect(adapter.write).not.toHaveBeenCalled();
    // Still running on defaults, so the user is told again.
    expect(Notice.calls).toHaveLength(2);
  });

  it("keeps a separate copy when the broken file has changed since", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });

    await pluginOn(adapter).loadSettingsAndState();
    const changed = `${HALF_WRITTEN}"x`;
    data.set(DATA_JSON, changed);
    adapter.stat.mockResolvedValue({ type: "file", ctime: WRITTEN_AT, mtime: WRITTEN_AT + 60_000, size: changed.length });
    await pluginOn(adapter).loadSettingsAndState();

    expect(asideFiles(data)).toHaveLength(2);
    expect(data.get(COPY_NAME)).toBe(HALF_WRITTEN);
    expect(data.get(`${DATA_JSON}.corrupt-2026-10-21T08-31-15-250Z`)).toBe(changed);
  });

  it("names the copy without characters a Windows file name cannot hold", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });

    await pluginOn(adapter).loadSettingsAndState();

    const [aside] = asideFiles(data);
    expect(aside!.slice(PLUGIN_DIR.length + 1)).toMatch(/^data\.json\.corrupt-[0-9A-Za-z-]+$/);
  });

  it("still keeps a copy, named by the clock, when the file's time cannot be read", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });
    adapter.stat.mockRejectedValue(new Error("stat failed"));

    await pluginOn(adapter).loadSettingsAndState();

    const [aside] = asideFiles(data);
    expect(asideFiles(data)).toHaveLength(1);
    expect(data.get(aside!)).toBe(HALF_WRITTEN);
  });

  it("keeps an empty file too, since the settings it should have held are gone", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: "" });

    await pluginOn(adapter).loadSettingsAndState();

    expect(asideFiles(data)).toEqual([COPY_NAME]);
    expect(notices()).toHaveLength(1);
  });
});

describe("a data.json that Obsidian could not read", () => {
  it("is used as it reads the second time, with no copy and no notice", async () => {
    const stored = JSON.stringify({ settings: { llm: { apiKey: "kept", model: "m" } } });
    const { data, adapter } = pluginFolder({ [DATA_JSON]: stored });
    // The file was busy for Obsidian's read (a sync tool, a backup) and is fine a moment later.
    adapter.read.mockRejectedValueOnce(fsError("EBUSY"));
    const plugin = pluginOn(adapter);

    const warnings = await plugin.loadSettingsAndState();

    expect(plugin.settings.llm.apiKey).toBe("kept");
    expect(warnings).toEqual([]);
    expect(asideFiles(data)).toEqual([]);
    expect(adapter.write).not.toHaveBeenCalled();
    expect(Notice.calls).toEqual([]);
  });

  it("stops the plugin starting, and says why, when it cannot be read at all", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: '{"settings":{}}' });
    const denied = fsError("EACCES");
    adapter.read.mockRejectedValue(denied);

    await expect(pluginOn(adapter).loadSettingsAndState()).rejects.toBe(denied);

    expect(adapter.write).not.toHaveBeenCalled();
    expect(asideFiles(data)).toEqual([]);
    expect(notices()).toHaveLength(1);
    expect(notices()[0]).toContain("did not start");
    expect(notices()[0]).toContain(DATA_JSON);
  });

  it("stops the plugin starting, and says why, when no copy can be kept", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });
    const full = new Error("disk full");
    adapter.write.mockRejectedValue(full);

    await expect(pluginOn(adapter).loadSettingsAndState()).rejects.toBe(full);

    expect(asideFiles(data)).toEqual([]);
    expect(data.get(DATA_JSON)).toBe(HALF_WRITTEN);
    expect(notices()).toHaveLength(1);
    expect(notices()[0]).toContain("did not start");
  });

  it("treats a data.json that is gone by then as a first run", async () => {
    const { data, adapter } = pluginFolder({});
    const plugin = pluginOn(adapter, async () => undefined);

    const warnings = await plugin.loadSettingsAndState();

    expect(plugin.settings.llm.apiKey).toBe("");
    expect(warnings).toEqual([]);
    expect(asideFiles(data)).toEqual([]);
    expect(Notice.calls).toEqual([]);
  });
});

describe("a host that rejects with the parse error instead", () => {
  it("is handled the same way", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });
    const plugin = pluginOn(adapter, async () => { throw new SyntaxError("Unexpected end of JSON input"); });

    await plugin.loadSettingsAndState();

    expect(plugin.settings.llm.apiKey).toBe("");
    expect(asideFiles(data)).toEqual([COPY_NAME]);
    expect(notices()).toHaveLength(1);
  });

  it("still stops the plugin starting when no copy can be kept", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });
    const full = new Error("disk full");
    adapter.write.mockRejectedValue(full);
    const plugin = pluginOn(adapter, async () => { throw new SyntaxError("Unexpected end of JSON input"); });

    await expect(plugin.loadSettingsAndState()).rejects.toBe(full);

    expect(asideFiles(data)).toEqual([]);
  });

  it("does not guess when the cause is not a parse error", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: '{"settings":{}}' });
    const denied = fsError("EACCES");
    const plugin = pluginOn(adapter, async () => { throw denied; });

    await expect(plugin.loadSettingsAndState()).rejects.toBe(denied);

    expect(asideFiles(data)).toEqual([]);
    expect(Notice.calls).toEqual([]);
  });
});

describe("a data.json that loads", () => {
  it("is used as before, without touching the folder", async () => {
    const { data, adapter } = pluginFolder({
      [DATA_JSON]: JSON.stringify({ settings: { llm: { apiKey: "kept", model: "m" } } }),
    });
    const plugin = pluginOn(adapter);

    const warnings = await plugin.loadSettingsAndState();

    expect(plugin.settings.llm.apiKey).toBe("kept");
    expect(warnings).toEqual([]);
    expect(asideFiles(data)).toEqual([]);
    expect(Notice.calls).toEqual([]);
    expect(adapter.exists).not.toHaveBeenCalled();
    expect(adapter.write).not.toHaveBeenCalled();
  });

  it("starts on defaults, quietly, when there is no data.json yet", async () => {
    const { data, adapter } = pluginFolder({});
    const plugin = pluginOn(adapter);

    const warnings = await plugin.loadSettingsAndState();

    expect(plugin.settings.llm.apiKey).toBe("");
    expect(warnings).toEqual([]);
    expect(asideFiles(data)).toEqual([]);
    expect(Notice.calls).toEqual([]);
    // null from Obsidian is a first run only if the file is really absent.
    expect(adapter.exists).toHaveBeenCalledTimes(1);
    expect(adapter.write).not.toHaveBeenCalled();
  });
});

describe("a data.json that is valid JSON but not a settings object", () => {
  it.each([
    ["an array", "[]", "an array"],
    ["a number", "42", "a number"],
    ["a string", '"x"', "a string"],
    ["a boolean", "true", "a boolean"],
    ["a literal null", "null", "null"],
  ])("keeps %s aside, starts on defaults and says so", async (_name, text, described) => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: text });
    const plugin = pluginOn(adapter);

    const warnings = await plugin.loadSettingsAndState();

    expect(plugin.settings.llm.apiKey).toBe("");
    expect(asideFiles(data)).toEqual([COPY_NAME]);
    expect(data.get(COPY_NAME)).toBe(text);
    expect(data.get(DATA_JSON)).toBe(text);
    expect(warnings.join("\n")).toContain(COPY_NAME);
    expect(notices()).toHaveLength(1);
    expect(notices()[0]).toContain("default settings");
    expect(notices()[0]).toContain(COPY_NAME);
    expect(notices()[0]).toContain(described);
  });

  it("finds its copy again when started again", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: "[]" });

    await pluginOn(adapter).loadSettingsAndState();
    adapter.write.mockClear();
    await pluginOn(adapter).loadSettingsAndState();

    expect(asideFiles(data)).toEqual([COPY_NAME]);
    expect(adapter.write).not.toHaveBeenCalled();
    expect(Notice.calls).toHaveLength(2);
  });

  it("stops the plugin starting when no copy can be kept", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: "[]" });
    const full = new Error("disk full");
    adapter.write.mockRejectedValue(full);

    await expect(pluginOn(adapter).loadSettingsAndState()).rejects.toBe(full);

    expect(asideFiles(data)).toEqual([]);
    expect(data.get(DATA_JSON)).toBe("[]");
  });

  it("uses an object that Obsidian failed to deliver the first time", async () => {
    const text = JSON.stringify({ settings: { llm: { apiKey: "kept", model: "m" } } });
    const { data, adapter } = pluginFolder({ [DATA_JSON]: text });
    const plugin = pluginOn(adapter, async () => null);

    await plugin.loadSettingsAndState();

    expect(plugin.settings.llm.apiKey).toBe("kept");
    expect(asideFiles(data)).toEqual([]);
    expect(Notice.calls).toEqual([]);
  });

  it("treats null with no file as a first run", async () => {
    const { data, adapter } = pluginFolder({});

    const warnings = await pluginOn(adapter, async () => null).loadSettingsAndState();

    expect(warnings).toEqual([]);
    expect(asideFiles(data)).toEqual([]);
    expect(Notice.calls).toEqual([]);
  });
});

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
    rename: async (from, to) => { data.set(to, data.get(from)!); data.delete(from); },
    writeTextAtomic: async () => {},
    acquireLock: vi.fn(async () => null),
    list: vi.fn(async () => []),
  };
}

describe("loading the plugin with a data.json that is not valid JSON", () => {
  async function onloadWith(adapter: Adapter) {
    const plugin = pluginOn(adapter);
    const saveData = vi.fn();
    const buildHost = vi.spyOn(obsidianHost, "buildObsidianHostAdapters")
      .mockReturnValue({ storage: memoryStorage(), http: {}, markupParser: {} } as never);
    Object.assign(plugin, {
      cleanupCachesIfDue: vi.fn(),
      saveData,
      addStatusBarItem: () => document.createElement("div"),
      addSettingTab: () => { throw new Error("stop once the settings are loaded"); },
    });
    return { plugin, saveData, done: () => { plugin.onunload(); buildHost.mockRestore(); } };
  }

  it("keeps the file, tells the user and runs on defaults, without saving anything", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });
    const { plugin, saveData, done } = await onloadWith(adapter);
    try {
      // Reaching the settings tab means loading the settings did not fail.
      await expect(plugin.onload()).rejects.toThrow("stop once the settings are loaded");

      expect(plugin.settings.schedule.enabled).toBe(false);
      expect(plugin.settings.llm.apiKey).toBe("");
      expect(asideFiles(data)).toEqual([COPY_NAME]);
      expect(data.get(DATA_JSON)).toBe(HALF_WRITTEN);
      expect(notices().join("\n")).toContain("default settings");
      expect(saveData).not.toHaveBeenCalled();
    } finally {
      done();
    }
  });

  it("does not start at all when the file can be neither read nor kept", async () => {
    const { data, adapter } = pluginFolder({ [DATA_JSON]: HALF_WRITTEN });
    adapter.write.mockRejectedValue(new Error("disk full"));
    const { plugin, saveData, done } = await onloadWith(adapter);
    try {
      await expect(plugin.onload()).rejects.toThrow("disk full");

      expect(data.get(DATA_JSON)).toBe(HALF_WRITTEN);
      expect(notices().join("\n")).toContain("did not start");
      expect(saveData).not.toHaveBeenCalled();
      expect(plugin.settings).toBeUndefined();
    } finally {
      done();
    }
  });
});
