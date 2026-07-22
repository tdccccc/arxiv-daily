import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { buildObsidianHostAdapters } from "../src/hosts/obsidian";
import { ObsidianMarkupParser } from "../src/hosts/obsidian/markup-parser";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import type { PluginSettings } from "@arxiv-daily/core";

const resourceOpenerSource = readFileSync(
  resolve(process.cwd(), "src/hosts/obsidian/resource-opener.ts"),
  "utf-8",
);

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
  it("uses Obsidian's active window without globalThis", () => {
    expect(resourceOpenerSource).toContain("window.activeWindow.open");
    expect(resourceOpenerSource).not.toContain("globalThis");
  });

  it("uses the host DOMParser for markup parsing", () => {
    const parseFromString = vi.fn(() => ({}) as Document);
    const DomParser = vi.fn(function (this: { parseFromString: typeof parseFromString }) {
      this.parseFromString = parseFromString;
    });
    vi.stubGlobal("DOMParser", DomParser);

    new ObsidianMarkupParser().parseFromString("<feed />", "text/xml");

    expect(DomParser).toHaveBeenCalledTimes(1);
    expect(parseFromString).toHaveBeenCalledWith("<feed />", "text/xml");
    vi.unstubAllGlobals();
  });

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

  it("atomically preserves scientific Markdown bytes without touching plugin data", async () => {
    const { app, files } = testApp();
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: testSettings,
    });
    const body = String.raw`- **Core results**: $\mathrm{NMAD}$ and $\eta$ hold for \(r_{\rm cut}/R_{\rm vir}\), z<0.1, z>3.5, and A & B.`;

    await host.storage.writeTextAtomic?.("arxiv-daily/daily/science.md", body);

    expect(files["arxiv-daily/daily/science.md"]).toBe(body);
    expect(files["arxiv-daily/daily/science.md.tmp"]).toBeUndefined();
    expect(files["arxiv-daily/daily/science.md.bak"]).toBeUndefined();
    expect(Object.keys(files)).toEqual(["arxiv-daily/daily/science.md"]);
  });

  it("opens URLs through Obsidian's active window", async () => {
    const { app } = testApp();
    const activeWindow = { open: vi.fn(() => null) } as unknown as Window;
    Object.defineProperty(window, "activeWindow", {
      configurable: true,
      value: activeWindow,
    });
    const open = activeWindow.open as ReturnType<typeof vi.fn>;
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
  });
});
