import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { Notice, type App } from "obsidian";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import type ArxivDailyPlugin from "../main";
import { ArxivDailySettingTab } from "../src/settings/tab";
import { SettingsChangeService } from "../src/settings/change-service";
import {
  confirmEmbeddingMode,
  confirmLibraryAuthorization,
  confirmLibraryRevocation,
} from "../src/library/modal";

vi.mock("../src/library/modal", () => ({
  confirmEmbeddingMode: vi.fn(),
  confirmLibraryAuthorization: vi.fn(),
  confirmLibraryRevocation: vi.fn(),
  showLibraryInventoryPreview: vi.fn(),
  showPersonalLibraryCatalogSummary: vi.fn(),
}));

beforeAll(() => {
  const proto = HTMLElement.prototype as HTMLElement & {
    empty?: () => void;
    addClass?: (...classes: string[]) => void;
    setText?: (text: string) => void;
    createEl?: (tag: string, options?: Record<string, unknown>) => HTMLElement;
    createDiv?: (options?: Record<string, unknown>) => HTMLElement;
    createSpan?: (options?: Record<string, unknown>) => HTMLElement;
  };
  proto.empty ??= function () { this.replaceChildren(); };
  proto.addClass ??= function (...classes: string[]) { this.classList.add(...classes); };
  proto.setText ??= function (text: string) { this.textContent = text; };
  proto.createEl ??= function (tag: string, options: any = {}) {
    const element = document.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text) element.textContent = options.text;
    this.appendChild(element);
    return element;
  };
  proto.createDiv ??= function (options: any = {}) { return this.createEl!("div", options); };
  proto.createSpan ??= function (options: any = {}) { return this.createEl!("span", options); };
});

/**
 * A tab over a plugin fake whose authorization status is derived the same way
 * the real plugin derives it: a grant is a fingerprint over the folder plus
 * the endpoint currently in scope, so moving the endpoint invalidates it.
 */
function makeTab() {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.llm.baseUrl = "https://llm.example.com/v1";
  const state: { root?: string; grant?: string } = {};
  const saveSettings = vi.fn(() => Promise.resolve());

  const fingerprintFor = (mode: "local" | "remote") =>
    mode === "remote" && settings.embedding.baseUrl.trim()
      ? `sha256:remote:${settings.embedding.baseUrl.trim()}`
      : "sha256:local";

  const disclosureFor = (mode: "local" | "remote") => {
    if (!state.root) return null;
    const remote = mode === "remote" && Boolean(settings.embedding.baseUrl.trim());
    return {
      selectedRoot: state.root,
      eligibleExtensions: [".pdf"],
      processingDepth: remote ? "full-text" : "metadata-and-abstracts",
      endpoint: "https://llm.example.com/v1/chat/completions",
      ...(remote
        ? { embeddingEndpoint: `${settings.embedding.baseUrl.trim()}/embeddings` }
        : {}),
      authorizationFingerprint: fingerprintFor(mode),
    };
  };

  const plugin = {
    settings,
    saveSettings,
    app: {},
    manifest: { version: "0.0.0-test" },
    stateStore: { snapshot: () => ({}) },
    logger: {
      error: vi.fn(),
      setLevel: vi.fn(),
      setTimezone: vi.fn(),
      setSensitiveValues: vi.fn(),
    },
    refreshSensitiveValues: vi.fn(),
    restartScheduler: vi.fn(),
    selectLibraryRoot: vi.fn(async () => {
      state.root = "/private/papers";
      return "selected" as const;
    }),
    getLibraryConnectionStatus: vi.fn(() => {
      if (!state.root) return { kind: "disconnected" as const };
      if (!state.grant) {
        return { kind: "authorization-required" as const, rootLabel: "papers" };
      }
      if (state.grant !== fingerprintFor(settings.embedding.mode)) {
        return { kind: "authorization-invalidated" as const, rootLabel: "papers" };
      }
      return {
        kind: "authorized" as const,
        rootLabel: "papers",
        grantedAt: "2026-08-30T00:00:00.000Z",
      };
    }),
    getLibraryAuthorizationDisclosure: vi.fn(
      (options?: { embeddingMode?: "local" | "remote" }) =>
        disclosureFor(options?.embeddingMode ?? settings.embedding.mode),
    ),
    authorizeLibraryProcessing: vi.fn(async (expected?: string) => {
      const current = fingerprintFor(settings.embedding.mode);
      if (expected && expected !== current) {
        throw new Error("Library authorization terms changed; review them again");
      }
      state.grant = current;
    }),
    revokeLibraryProcessing: vi.fn(async () => { state.grant = undefined; }),
    indexPersonalLibraryFullText: vi.fn(async () => ({
      indexed: 1,
      reused: 0,
      failed: 0,
      pruned: 0,
      titlesRefreshed: 0,
    })),
  } as unknown as ArxivDailyPlugin;

  (plugin as unknown as { settingsChanges: SettingsChangeService }).settingsChanges =
    new SettingsChangeService({
      settings,
      persistSettings: async (candidate) => { await saveSettings(candidate); },
      setLoggerLevel: () => {},
      setLoggerTimezone: () => {},
      restartScheduler: () => {},
      refreshSensitiveValues: () => {},
    });

  const tab = new ArxivDailySettingTab({} as App, plugin);
  vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
  return { tab, plugin, settings, state, saveSettings };
}

const confirmAuthorization = vi.mocked(confirmLibraryAuthorization);
const confirmRevocation = vi.mocked(confirmLibraryRevocation);
const confirmMode = vi.mocked(confirmEmbeddingMode);

beforeEach(() => {
  vi.clearAllMocks();
  Notice.calls.length = 0;
});

describe("switching the embedding mode asks in place (ADR 0008)", () => {
  it("discloses full text, then switches and authorizes in one confirmed step", async () => {
    const { tab, plugin, settings, state } = makeTab();
    state.root = "/private/papers";
    settings.embedding.baseUrl = "https://embed.example.com/v1";
    confirmAuthorization.mockResolvedValue(true);

    await tab.applyEmbeddingModeChange("remote");

    expect(confirmAuthorization).toHaveBeenCalledTimes(1);
    const disclosure = confirmAuthorization.mock.calls[0]?.[1];
    expect(disclosure).toMatchObject({
      selectedRoot: "/private/papers",
      processingDepth: "full-text",
      embeddingEndpoint: "https://embed.example.com/v1/embeddings",
    });
    expect(settings.embedding.mode).toBe("remote");
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorized");
    expect(plugin.authorizeLibraryProcessing).toHaveBeenCalledWith(
      disclosure?.authorizationFingerprint,
    );
  });

  it("changes nothing when the disclosure is declined", async () => {
    const { tab, plugin, settings, state, saveSettings } = makeTab();
    state.root = "/private/papers";
    settings.embedding.baseUrl = "https://embed.example.com/v1";
    confirmAuthorization.mockResolvedValue(false);

    await tab.applyEmbeddingModeChange("remote");

    expect(settings.embedding.mode).toBe("local");
    expect(plugin.authorizeLibraryProcessing).not.toHaveBeenCalled();
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorization-required");
    expect(saveSettings).not.toHaveBeenCalled();
  });

  it("defers the disclosure to folder selection when no folder is chosen yet", async () => {
    const { tab, plugin, settings } = makeTab();
    settings.embedding.baseUrl = "https://embed.example.com/v1";
    confirmAuthorization.mockResolvedValue(true);

    await tab.applyEmbeddingModeChange("remote");

    // Nothing concrete to disclose without a folder: switch now, ask later.
    expect(confirmAuthorization).not.toHaveBeenCalled();
    expect(settings.embedding.mode).toBe("remote");
    expect(settings.embedding.initialChoiceDone).toBe(true);

    await tab.chooseLibraryRoot();

    expect(confirmMode).not.toHaveBeenCalled();
    expect(confirmAuthorization).toHaveBeenCalledTimes(1);
    expect(confirmAuthorization.mock.calls[0]?.[1]).toMatchObject({
      selectedRoot: "/private/papers",
      processingDepth: "full-text",
    });
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorized");
  });
});

describe("moving the authorized endpoint re-asks", () => {
  async function authorizedRemoteTab() {
    const made = makeTab();
    made.state.root = "/private/papers";
    made.settings.embedding.baseUrl = "https://embed.example.com/v1";
    confirmAuthorization.mockResolvedValue(true);
    await made.tab.applyEmbeddingModeChange("remote");
    confirmAuthorization.mockReset();
    return made;
  }

  it("restores the authorized endpoint when the new one is declined", async () => {
    const { tab, plugin, settings } = await authorizedRemoteTab();
    confirmAuthorization.mockResolvedValue(false);

    const displayed = await tab.saveEmbeddingEndpointField(
      "embedding.baseUrl",
      "https://elsewhere.example.com/v1",
    );

    expect(confirmAuthorization).toHaveBeenCalledTimes(1);
    expect(displayed).toBe("https://embed.example.com/v1");
    expect(settings.embedding.baseUrl).toBe("https://embed.example.com/v1");
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorized");
  });

  it("re-authorizes the new endpoint when the disclosure is confirmed", async () => {
    const { tab, plugin, settings } = await authorizedRemoteTab();
    confirmAuthorization.mockResolvedValue(true);

    const displayed = await tab.saveEmbeddingEndpointField(
      "embedding.baseUrl",
      "https://elsewhere.example.com/v1",
    );

    expect(confirmAuthorization.mock.calls[0]?.[1]).toMatchObject({
      embeddingEndpoint: "https://elsewhere.example.com/v1/embeddings",
    });
    expect(displayed).toBe("https://elsewhere.example.com/v1");
    expect(settings.embedding.baseUrl).toBe("https://elsewhere.example.com/v1");
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorized");
  });

  it("saves without asking when the endpoint identity does not move", async () => {
    const { tab, settings } = await authorizedRemoteTab();

    await tab.saveEmbeddingEndpointField("embedding.model", "text-embedding-3-small");

    expect(confirmAuthorization).not.toHaveBeenCalled();
    expect(settings.embedding.model).toBe("text-embedding-3-small");
  });
});

describe("revocation switches back to local (A1)", () => {
  async function authorizedRemoteTab() {
    const made = makeTab();
    made.state.root = "/private/papers";
    made.settings.embedding.baseUrl = "https://embed.example.com/v1";
    confirmAuthorization.mockResolvedValue(true);
    await made.tab.applyEmbeddingModeChange("remote");
    return made;
  }

  it("revokes and returns to local embedding when confirmed", async () => {
    const { tab, plugin, settings } = await authorizedRemoteTab();
    confirmRevocation.mockResolvedValue(true);

    await tab.revokeLibraryAuthorization();

    expect(confirmRevocation).toHaveBeenCalledTimes(1);
    expect(confirmRevocation.mock.calls[0]?.[1]).toMatchObject({ switchesToLocal: true });
    expect(plugin.revokeLibraryProcessing).toHaveBeenCalledTimes(1);
    expect(settings.embedding.mode).toBe("local");
  });

  it("changes neither the grant nor the mode when cancelled", async () => {
    const { tab, plugin, settings } = await authorizedRemoteTab();
    confirmRevocation.mockResolvedValue(false);

    await tab.revokeLibraryAuthorization();

    expect(plugin.revokeLibraryProcessing).not.toHaveBeenCalled();
    expect(settings.embedding.mode).toBe("remote");
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorized");
  });
});

describe("legacy remote configurations without a grant", () => {
  function legacyTab() {
    const made = makeTab();
    made.state.root = "/private/papers";
    made.settings.embedding.mode = "remote";
    made.settings.embedding.initialChoiceDone = true;
    made.settings.embedding.baseUrl = "https://embed.example.com/v1";
    return made;
  }

  it("confirms before building the index and then indexes", async () => {
    const { tab, plugin } = legacyTab();
    confirmAuthorization.mockResolvedValue(true);

    await tab.indexPersonalLibraryFullText();

    expect(confirmAuthorization).toHaveBeenCalledTimes(1);
    expect(plugin.authorizeLibraryProcessing).toHaveBeenCalledTimes(1);
    expect(plugin.indexPersonalLibraryFullText).toHaveBeenCalledTimes(1);
  });

  it("aborts indexing and changes nothing when the disclosure is declined", async () => {
    const { tab, plugin, settings } = legacyTab();
    confirmAuthorization.mockResolvedValue(false);

    await tab.indexPersonalLibraryFullText();

    expect(plugin.indexPersonalLibraryFullText).not.toHaveBeenCalled();
    expect(plugin.authorizeLibraryProcessing).not.toHaveBeenCalled();
    expect(settings.embedding.mode).toBe("remote");
    expect(plugin.getLibraryConnectionStatus().kind).toBe("authorization-required");
  });

  it("does not re-ask an already authorized library", async () => {
    const { tab, plugin, state } = legacyTab();
    state.grant = "sha256:remote:https://embed.example.com/v1";

    await tab.indexPersonalLibraryFullText();

    expect(confirmAuthorization).not.toHaveBeenCalled();
    expect(plugin.indexPersonalLibraryFullText).toHaveBeenCalledTimes(1);
  });

  it("never asks in local mode", async () => {
    const { tab, plugin } = makeTab();
    plugin.settings.embedding.mode = "local";

    await tab.indexPersonalLibraryFullText();

    expect(confirmAuthorization).not.toHaveBeenCalled();
    expect(plugin.indexPersonalLibraryFullText).toHaveBeenCalledTimes(1);
  });
});
