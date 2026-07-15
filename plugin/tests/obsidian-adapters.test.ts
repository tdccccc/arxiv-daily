import { describe, expect, it, vi } from "vitest";
import { buildObsidianHostAdapters } from "../src/hosts/obsidian";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import type { PluginSettings } from "@arxiv-daily/core";

function testSettings(): PluginSettings {
  return {
    ...DEFAULT_SETTINGS,
    llm: {
      ...DEFAULT_SETTINGS.llm,
      apiKey: "stored-key",
    },
    arxiv: { ...DEFAULT_SETTINGS.arxiv },
    output: { ...DEFAULT_SETTINGS.output },
    schedule: { ...DEFAULT_SETTINGS.schedule },
    advanced: { ...DEFAULT_SETTINGS.advanced },
  };
}

function testApp() {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const workspace = {
    openLinkText: vi.fn().mockResolvedValue(undefined),
  };
  const app = {
    vault: {
      adapter: {
        read: vi.fn(async (path: string) => files[path]),
        write: vi.fn(async (path: string, content: string) => {
          files[path] = content;
        }),
        exists: vi.fn(async (path: string) => path in files || dirs.has(path)),
        mkdir: vi.fn(async (path: string) => {
          dirs.add(path);
        }),
        rename: vi.fn(async (from: string, to: string) => {
          files[to] = files[from];
          delete files[from];
        }),
        remove: vi.fn(async (path: string) => {
          delete files[path];
          dirs.delete(path);
        }),
        readBinary: vi.fn(async () => new Uint8Array([1, 2]).buffer),
        writeBinary: vi.fn(async () => {}),
      },
    },
    workspace,
  };
  return { app, files, workspace };
}

describe("Obsidian host adapters", () => {
  it("assembles vault storage, settings secrets, and resource opening", async () => {
    const { app, files, workspace } = testApp();
    const settings = testSettings();
    const persistSettings = vi.fn();

    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: () => settings,
      persistSettings,
    });

    await host.storage.mkdir("arxiv-daily/daily");
    await host.storage.writeText("arxiv-daily/daily/today.md", "daily");
    await host.storage.writeTextAtomic?.("arxiv-daily/daily/atomic.md", "atomic");
    expect(files["arxiv-daily/daily/today.md"]).toBe("daily");
    expect(files["arxiv-daily/daily/atomic.md"]).toBe("atomic");
    expect(files["arxiv-daily/daily/atomic.md.tmp"]).toBeUndefined();
    expect(await host.storage.exists("arxiv-daily/daily")).toBe(true);

    expect(await host.secrets.getSecret("llm.apiKey")).toBe("stored-key");
    await host.secrets.setSecret?.("apiKey", "next-key");
    expect(settings.llm.apiKey).toBe("next-key");
    expect(persistSettings).toHaveBeenCalledTimes(1);

    await host.opener.openNote("arxiv-daily/papers/2606.12345.md", {
      newLeaf: true,
    });
    expect(workspace.openLinkText).toHaveBeenCalledWith(
      "arxiv-daily/papers/2606.12345.md",
      "",
      true,
    );
  });

  it("opens URLs through the host window", async () => {
    const { app } = testApp();
    const open = vi.fn();
    vi.stubGlobal("open", open);
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: testSettings,
    });

    await host.opener.openUrl("https://arxiv.org/abs/2606.12345");

    expect(open).toHaveBeenCalledWith(
      "https://arxiv.org/abs/2606.12345",
      "_blank",
      "noopener",
    );
    vi.unstubAllGlobals();
  });
});
