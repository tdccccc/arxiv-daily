import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  type LibraryInventory,
} from "@arxiv-daily/core";
import type { OpenedScopedLibrarySource } from "@arxiv-daily/node-runtime/scoped-library-source";
import ArxivDailyPlugin from "../main.ts";
import {
  createLibraryConnection,
  libraryAuthorizationDisclosure,
} from "../src/library/connection";

function makePlugin() {
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  const saveData = vi.fn().mockResolvedValue(undefined);
  const setSensitiveValues = vi.fn();
  const files = new Map<string, string>();
  Object.assign(plugin, {
    settings: structuredClone(DEFAULT_SETTINGS),
    logger: { setSensitiveValues, error: vi.fn(), warn: vi.fn() },
    host: { storage: {
      normalizePath: (path: string) => path,
      exists: async (path: string) => files.has(path),
      readText: async (path: string) => files.get(path)!,
      writeText: async (path: string, value: string) => { files.set(path, value); },
      writeTextAtomic: async (path: string, value: string) => { files.set(path, value); },
      mkdir: async () => undefined,
      rename: async () => undefined,
      remove: async () => undefined,
    } },
    saveData,
    libraryDirectoryPicker: { select: vi.fn() },
    openLibrarySource: vi.fn(),
    operations: { cancelAll: vi.fn() },
    librarySelectionRevision: 0,
    libraryConnectionRevision: 0,
    libraryOutputRevision: 0,
    libraryMutationQueue: Promise.resolve(),
  });
  return {
    plugin,
    internals: plugin as unknown as Record<string, any>,
    saveData,
    setSensitiveValues,
  };
}

function sourceWithInventory(
  inventory: LibraryInventory,
  canonicalRoot = "/papers",
): OpenedScopedLibrarySource {
  return {
    canonicalRoot,
    rootIdentity: canonicalRoot === "/previous" ? "1:3" : "1:2",
    inventory: vi.fn().mockResolvedValue(inventory),
    readBinary: vi.fn(),
  };
}

describe("personal library plugin lifecycle", () => {
  it("loads valid sibling state and ignores malformed persisted connections", async () => {
    const valid = createLibraryConnection("/papers", "1:2");
    const first = makePlugin();
    first.internals.loadData = vi.fn().mockResolvedValue({
      settings: structuredClone(DEFAULT_SETTINGS),
      libraryConnection: valid,
    });
    await first.internals.loadSettingsAndState();
    expect(first.plugin.getLibraryConnectionStatus().kind)
      .toBe("authorization-required");

    const malformed = makePlugin();
    malformed.internals.loadData = vi.fn().mockResolvedValue({
      settings: structuredClone(DEFAULT_SETTINGS),
      libraryConnection: { schemaVersion: 1, selectedRoot: "../bad" },
    });
    await expect(malformed.internals.loadSettingsAndState()).resolves.toContain(
      "ignored invalid personal library connection metadata",
    );
    expect(malformed.plugin.getLibraryConnectionStatus()).toEqual({
      kind: "disconnected",
    });
  });

  it("selects a validated source, persists sibling state, and refreshes redaction", async () => {
    const { plugin, internals, saveData, setSensitiveValues } = makePlugin();
    const source = sourceWithInventory(
      { entries: [], truncated: false },
      "/private/papers",
    );
    internals.libraryDirectoryPicker.select.mockResolvedValue({
      kind: "selected",
      path: "/private/papers",
    });
    internals.openLibrarySource.mockResolvedValue(source);

    await expect(plugin.selectLibraryRoot()).resolves.toBe("selected");

    expect(internals.openLibrarySource).toHaveBeenCalledWith("/private/papers");
    expect(saveData).toHaveBeenCalledWith(expect.objectContaining({
      settings: plugin.settings,
      libraryConnection: expect.objectContaining({
        selectedRoot: "/private/papers",
      }),
    }));
    expect(setSensitiveValues).toHaveBeenLastCalledWith(
      expect.arrayContaining(["/private/papers"]),
    );
  });

  it("ignores an older folder selection that resolves after a newer one", async () => {
    const { plugin, internals, saveData } = makePlugin();
    let resolveOlderSource!: (source: OpenedScopedLibrarySource) => void;
    const olderSource = new Promise<OpenedScopedLibrarySource>((resolve) => {
      resolveOlderSource = resolve;
    });
    internals.libraryDirectoryPicker.select
      .mockResolvedValueOnce({ kind: "selected", path: "/older" })
      .mockResolvedValueOnce({ kind: "selected", path: "/newer" });
    internals.openLibrarySource.mockImplementation((root: string) =>
      root === "/older"
        ? olderSource
        : Promise.resolve(sourceWithInventory(
          { entries: [], truncated: false },
          "/newer",
        )),
    );

    const olderSelection = plugin.selectLibraryRoot();
    await vi.waitFor(() => expect(internals.openLibrarySource).toHaveBeenCalledWith("/older"));
    await expect(plugin.selectLibraryRoot()).resolves.toBe("selected");
    resolveOlderSource(sourceWithInventory(
      { entries: [], truncated: false },
      "/older",
    ));
    await expect(olderSelection).resolves.toBe("cancelled");

    expect(internals.libraryConnection.selectedRoot).toBe("/newer");
    expect(saveData).toHaveBeenCalledTimes(1);
  });

  it("retains the previous connection, source, and redaction when selection persistence fails", async () => {
    const { plugin, internals, saveData, setSensitiveValues } = makePlugin();
    const previous = createLibraryConnection("/previous", "1:3");
    const previousSource = sourceWithInventory(
      { entries: [], truncated: false },
      "/previous",
    );
    internals.libraryConnection = previous;
    internals.librarySource = previousSource;
    internals.libraryDirectoryPicker.select.mockResolvedValue({
      kind: "selected",
      path: "/next",
    });
    internals.openLibrarySource.mockResolvedValue(
      sourceWithInventory({ entries: [], truncated: false }, "/next"),
    );
    saveData.mockRejectedValueOnce(new Error("disk full"));

    await expect(plugin.selectLibraryRoot()).rejects.toThrow("disk full");

    expect(internals.libraryConnection).toBe(previous);
    expect(internals.librarySource).toBe(previousSource);
    expect(setSensitiveValues).toHaveBeenLastCalledWith(
      expect.arrayContaining(["/previous"]),
    );
  });

  it("authorizes only the exact terms the user reviewed", async () => {
    const { plugin, internals } = makePlugin();
    internals.libraryConnection = createLibraryConnection("/papers", "1:2");
    const reviewed = plugin.getLibraryAuthorizationDisclosure();
    expect(reviewed).not.toBeNull();

    plugin.settings.llm.baseUrl = "https://changed.example/v1";
    await expect(plugin.authorizeLibraryProcessing(
      reviewed!.authorizationFingerprint,
    )).rejects.toThrow("terms changed");
    expect(plugin.getLibraryConnectionStatus().kind)
      .toBe("authorization-required");

    const current = plugin.getLibraryAuthorizationDisclosure();
    await plugin.authorizeLibraryProcessing(current!.authorizationFingerprint);
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorized");
  });

  it("serializes ordinary settings saves behind library mutation rollback", async () => {
    const { plugin, internals, saveData } = makePlugin();
    internals.libraryConnection = createLibraryConnection("/papers", "1:2");
    let rejectAuthorizationSave!: (error: Error) => void;
    saveData.mockImplementationOnce(() => new Promise<void>((_resolve, reject) => {
      rejectAuthorizationSave = reject;
    }));

    const authorization = plugin.authorizeLibraryProcessing();
    await vi.waitFor(() => expect(saveData).toHaveBeenCalledTimes(1));
    plugin.settings.llm.model = "new-model";
    const settingsSave = plugin.saveSettings();
    expect(saveData).toHaveBeenCalledTimes(1);
    rejectAuthorizationSave(new Error("authorization save failed"));

    await expect(authorization).rejects.toThrow("authorization save failed");
    await expect(settingsSave).resolves.toBeUndefined();
    expect(saveData).toHaveBeenCalledTimes(2);
    expect(saveData.mock.calls[1]?.[0].libraryConnection.authorization)
      .toBeUndefined();
  });

  it("invalidates authorization after endpoint changes but not model or API-key changes", async () => {
    const { plugin, internals } = makePlugin();
    internals.libraryConnection = createLibraryConnection("/papers", "1:2");
    const disclosure = libraryAuthorizationDisclosure(
      internals.libraryConnection,
      { llmBaseUrl: plugin.settings.llm.baseUrl },
    );
    await plugin.authorizeLibraryProcessing(disclosure.authorizationFingerprint);

    plugin.settings.llm.model = "another-model";
    plugin.settings.llm.apiKey = "another-key";
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorized");
    plugin.settings.llm.baseUrl = "https://other.example/v1";
    expect(plugin.getLibraryConnectionStatus().kind)
      .toBe("authorization-invalidated");
  });

  it("previews a bounded local classification without model authorization", async () => {
    const { plugin, internals } = makePlugin();
    const source = sourceWithInventory({
      truncated: false,
      entries: [
        { path: "one.pdf", type: "file", size: 10 },
        { path: "draft.md", type: "file", size: 20 },
      ],
    });
    internals.libraryConnection = createLibraryConnection("/papers", "1:2");
    internals.librarySource = source;

    await expect(plugin.previewLibraryInventory()).resolves.toEqual({
      eligible: [{ path: "one.pdf", size: 10 }],
      ignored: [{ path: "draft.md", reason: "Unsupported file type" }],
      folders: 0,
      truncated: false,
    });
    expect(source.inventory).toHaveBeenCalledWith({
      signal: expect.any(AbortSignal),
    });
  });

  it("keeps local inventory after model revocation and aborts it on unload", async () => {
    const { plugin, internals } = makePlugin();
    let receivedSignal: AbortSignal | undefined;
    const source: OpenedScopedLibrarySource = {
      canonicalRoot: "/papers",
      rootIdentity: "1:2",
      inventory: vi.fn(({ signal } = {}) => new Promise((_resolve, reject) => {
        receivedSignal = signal;
        signal?.addEventListener("abort", () => reject(
          signal.reason instanceof Error
            ? signal.reason
            : new DOMException("Aborted", "AbortError"),
        ), { once: true });
      })),
      readBinary: vi.fn(),
    };
    internals.libraryConnection = createLibraryConnection("/papers", "1:2");
    internals.librarySource = source;
    await plugin.authorizeLibraryProcessing();

    const preview = plugin.previewLibraryInventory();
    await vi.waitFor(() => expect(receivedSignal).toBeDefined());
    await plugin.revokeLibraryProcessing();
    expect(receivedSignal!.aborted).toBe(false);
    plugin.onunload();
    await expect(preview).rejects.toThrow();
  });

  it("rejects a re-opened folder when its filesystem identity changed", async () => {
    const { plugin, internals } = makePlugin();
    internals.libraryConnection = createLibraryConnection("/papers", "1:2");
    internals.openLibrarySource.mockResolvedValue({
      ...sourceWithInventory({ entries: [], truncated: false }),
      rootIdentity: "1:99",
    });

    await expect(plugin.previewLibraryInventory()).rejects.toThrow(
      "folder identity changed",
    );
    expect(internals.librarySource).toBeUndefined();
  });

  it("does not open a source for an already-aborted preview", async () => {
    const { plugin, internals } = makePlugin();
    internals.libraryConnection = createLibraryConnection("/papers", "1:2");
    await plugin.authorizeLibraryProcessing();
    const controller = new AbortController();
    controller.abort("cancelled before start");

    await expect(plugin.previewLibraryInventory(controller.signal)).rejects.toBe(
      "cancelled before start",
    );
    expect(internals.openLibrarySource).not.toHaveBeenCalled();
  });
});
