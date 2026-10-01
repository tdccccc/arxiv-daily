import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import * as obsidian from "obsidian";
import { MenuItem, Modal, Notice, Setting, ToggleComponent, type App } from "obsidian";
import {
  DEFAULT_SETTINGS, LlmClient, PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION, createEmptyPersonalLibraryCatalog,
  createPersonalLibraryCatalogInputManifestFingerprint, createPersonalLibraryRepresentativeSetFingerprint,
  normalizeTopic, type PersonalLibraryDirectionProposal, type PluginSettings, type ProposalAcceptanceReceipt,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { LibraryIndexStatusStore } from "../src/library/index-status";
import {
  ArxivDailySettingTab,
  LIBRARY_INDEX_PROGRESS_INTERVAL_MS,
} from "../src/settings/tab";
import {
  allSettingKeys,
  buildSettingDefinitions,
  readSettingValue,
  SETTING_KEYS,
  writeSettingValue,
} from "../src/settings/definitions";
import {
  renderApiKeyRow,
  renderCategoryRow,
  renderEmailApiKeyRow,
  renderEmailModeRow,
  renderEmailToRow,
  renderEmbeddingBaseUrlRow,
  renderEmbeddingModeRow,
  renderHostedTokenRow,
  renderLibraryDirectionsRow,
  renderLlmBaseUrlRow,
  renderModelRow,
  renderPdfParserSidecarCapabilitiesUrlRow,
  renderReasoningEffortRow,
  renderRunWindowRow,
  renderLibraryGuideRow,
  renderScheduleEnabledRow,
  renderSetupGuideRow,
  renderTickIntervalRow,
  renderTimezoneRow,
} from "../src/settings/declarative-rows";
import { SettingsChangeService } from "../src/settings/change-service";

beforeAll(() => {
  type CreateOptions = {
    cls?: string;
    text?: string;
    type?: string;
    value?: string;
    attr?: Record<string, string>;
  };
  const proto = HTMLElement.prototype as HTMLElement & {
    empty?: () => void;
    addClass?: (...classes: string[]) => void;
    removeClass?: (...classes: string[]) => void;
    toggleClass?: (className: string, force?: boolean) => void;
    setText?: (text: string) => void;
    appendText?: (text: string) => void;
    detach?: () => void;
    createEl?: (tag: string, options?: CreateOptions) => HTMLElement;
    createDiv?: (options?: CreateOptions) => HTMLElement;
    createSpan?: (options?: CreateOptions) => HTMLElement;
  };
  proto.empty ??= function () { this.replaceChildren(); };
  proto.addClass ??= function (...classes) { this.classList.add(...classes); };
  proto.removeClass ??= function (...classes) { this.classList.remove(...classes); };
  proto.toggleClass ??= function (className, force) {
    this.classList.toggle(className, force);
  };
  proto.setText ??= function (text) { this.textContent = text; };
  proto.appendText ??= function (text) { this.append(text); };
  proto.detach ??= function () { this.remove(); };
  proto.createEl ??= function (tag, options = {}) {
    const element = document.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text) element.textContent = options.text;
    if (options.type) element.setAttribute("type", options.type);
    if (options.value !== undefined) {
      (element as HTMLInputElement | HTMLOptionElement).value = options.value;
    }
    for (const [key, value] of Object.entries(options.attr ?? {})) {
      element.setAttribute(key, value);
    }
    this.appendChild(element);
    return element;
  };
  proto.createDiv ??= function (options = {}) {
    return this.createEl("div", options);
  };
  proto.createSpan ??= function (options = {}) {
    return this.createEl("span", options);
  };
});

function deferred(): {
  promise: Promise<void>;
  resolve: () => void;
  reject: (error: Error) => void;
} {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function renderSetting() {
  const settingEl = document.createElement("div") as HTMLElement & {
    createEl: typeof HTMLElement.prototype.createEl;
  };
  const controlEl = document.createElement("div") as HTMLElement & {
    createEl: typeof HTMLElement.prototype.createEl;
    empty: typeof HTMLElement.prototype.empty;
  };
  settingEl.appendChild(controlEl);
  return { settingEl, controlEl };
}

/** Real settings queue; saveSettings is the mocked persistence boundary. */
function makeTab() {
  const settings = structuredClone(DEFAULT_SETTINGS);
  const saveSettings = vi.fn((_candidate?: PluginSettings) => Promise.resolve());
  const plugin = {
    settings,
    saveSettings,
    setLlmBaseUrl: vi.fn(async (value: string) => {
      settings.llm.baseUrl = value.trim();
      await saveSettings();
    }),
    manifest: { version: "0.0.0-test" },
    app: {},
    stateStore: { snapshot: () => ({}) },
    logger: {
      setSensitiveValues: vi.fn(),
      setLevel: vi.fn(),
      setTimezone: vi.fn(),
      error: vi.fn(),
      warn: vi.fn(),
    },
    refreshSensitiveValues: vi.fn(),
    restartScheduler: vi.fn(),
    sendHostedVerificationEmail: vi.fn().mockResolvedValue("Verification sent"),
    sendTestEmail: vi.fn().mockResolvedValue("Test sent"),
    automaticEmailSupported: vi.fn().mockReturnValue(true),
    getLibraryConnectionStatus: vi.fn().mockReturnValue({ kind: "disconnected" }),
    libraryIndexStatus: new LibraryIndexStatusStore(),
    selectLibraryRoot: vi.fn().mockResolvedValue("cancelled"),
    getLibraryAuthorizationDisclosure: vi.fn().mockReturnValue(null),
    authorizeLibraryProcessing: vi.fn().mockResolvedValue(undefined),
    previewLibraryInventory: vi.fn().mockResolvedValue({
      eligible: [],
      ignored: [],
      folders: 0,
      truncated: false,
    }),
    scanPersonalLibrary: vi.fn().mockResolvedValue({}),
    reloadPersonalLibraryCatalog: vi.fn().mockResolvedValue({}),
    revokeLibraryProcessing: vi.fn().mockResolvedValue(undefined),
    cancelPersonalLibraryIndexing: vi.fn().mockReturnValue(true),
    openPersonalLibraryDirectionReview: vi.fn(),
    getLastFullTextIndexLibraryContext: vi.fn().mockReturnValue(undefined),
    getPersonalLibraryInterestProfile: vi.fn().mockReturnValue(null),
  } as unknown as ArxivDailyPlugin;
  (plugin as unknown as { settingsChanges: SettingsChangeService }).settingsChanges =
    new SettingsChangeService({
      settings,
      persistSettings: async (candidate) => saveSettings(candidate),
      setLoggerLevel: (level) => plugin.logger.setLevel(level),
      setLoggerTimezone: (timezone) => plugin.logger.setTimezone(timezone),
      restartScheduler: () => plugin.restartScheduler(),
      refreshSensitiveValues: () => plugin.refreshSensitiveValues(),
    });
  plugin.saveSettings = () => plugin.settingsChanges.persistCurrent();
  plugin.setScheduleEnabled = vi.fn(async (enabled: boolean) => {
    await plugin.settingsChanges.changeValue("schedule.enabled", enabled);
    return true;
  });
  const tab = new ArxivDailySettingTab({} as App, plugin);
  return { tab, plugin, settings, saveSettings };
}

/** Walk nested declarative items, invoking fn for every item and list. */
function walkItems(
  items: readonly unknown[],
  fn: (item: Record<string, unknown>) => void,
): void {
  for (const item of items as ReadonlyArray<Record<string, unknown>>) {
    fn(item);
    if (Array.isArray(item.items)) walkItems(item.items as unknown[], fn);
  }
}

describe("declarative daily paper limit", () => {
  function renderLimit(tab: ArxivDailySettingTab): HTMLInputElement {
    let render: ((setting: Setting) => void) | undefined;
    walkItems(tab.getSettingDefinitions(), (item) => {
      if (item.name === "Daily paper limit" && typeof item.render === "function") {
        render = item.render as (setting: Setting) => void;
      }
    });
    expect(render).toBeTypeOf("function");
    const setting = new Setting(document.createElement("div"));
    render!(setting);
    return setting.controlEl.querySelector<HTMLInputElement>("input")!;
  }

  it("shows 20 and persists a whole-number edit as a number", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const input = renderLimit(tab);
    expect(input.value).toBe("20");
    expect(input.min).toBe("1");
    expect(input.step).toBe("1");
    input.value = "35";
    input.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(settings.output.maxDailyPapers).toBe(35));
    expect(saveSettings).toHaveBeenCalledWith(expect.objectContaining({
      output: expect.objectContaining({ maxDailyPapers: 35 }),
    }));
  });

  it.each(["0", "2.5", ""])("rejects the invalid draft %j without persistence", async (draft) => {
    const { tab, settings, saveSettings } = makeTab();
    const input = renderLimit(tab);
    input.value = draft;
    input.dispatchEvent(new Event("change"));
    await Promise.resolve();
    expect(input.validationMessage).toContain("positive whole number");
    expect(saveSettings).not.toHaveBeenCalled();
    expect(settings.output.maxDailyPapers).toBe(20);
  });

  it("restores the current limit when persistence fails", async () => {
    const { tab, settings, saveSettings } = makeTab();
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const input = renderLimit(tab);
    input.value = "35";
    input.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(input.value).toBe("20"));
    expect(saveSettings).toHaveBeenCalledOnce();
    expect(settings.output.maxDailyPapers).toBe(20);
  });
});

describe("wired getSettingDefinitions", () => {
  function libraryEntry(tab: ArxivDailySettingTab) {
    const row = tab.getSettingDefinitions().find((item) => "name" in item && item.name === "Topics from your library");
    expect(row).toBeDefined();
    const setting = new Setting(document.createElement("div"));
    if (row && "render" in row) row.render?.(setting);
    const button = setting.controlEl.querySelector<HTMLButtonElement>("button");
    expect(button).not.toBeNull();
    return button!;
  }

  it("opens direction review from research settings when an index is ready", async () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({ kind: "authorized", rootLabel: "papers", grantedAt: "2026-09-08T00:00:00.000Z" });
    plugin.libraryIndexStatus.setLastRun({ updatedAt: "2026-09-08T00:00:00.000Z", papers: 20 });
    libraryEntry(tab).click();
    await vi.waitFor(() => expect(plugin.openPersonalLibraryDirectionReview).toHaveBeenCalledOnce());
    expect(plugin.selectLibraryRoot).not.toHaveBeenCalled();
  });

  it("starts with choosing a library and stops when selection is cancelled", async () => {
    const { tab, plugin } = makeTab();
    libraryEntry(tab).click();
    await vi.waitFor(() => expect(plugin.selectLibraryRoot).toHaveBeenCalledOnce());
    expect(plugin.openPersonalLibraryDirectionReview).not.toHaveBeenCalled();
  });

  it("opens review only after a successful index and stops on failure", async () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({ kind: "authorized", rootLabel: "papers", grantedAt: "2026-09-08T00:00:00.000Z" });
    plugin.indexPersonalLibraryFullText = vi.fn(async () => { throw new Error("index failed"); });
    const button = libraryEntry(tab);
    button.click();
    await vi.waitFor(() => expect(plugin.logger.error).toHaveBeenCalled());
    expect(plugin.openPersonalLibraryDirectionReview).not.toHaveBeenCalled();
    plugin.indexPersonalLibraryFullText = vi.fn(async () => {
      plugin.libraryIndexStatus.setLastRun({ updatedAt: "2026-09-08T00:00:00.000Z", papers: 20 });
      return { indexed: 20, reused: 0, failed: 0, pruned: 0, titlesRefreshed: 0, outcomes: [] } as Awaited<ReturnType<ArxivDailyPlugin["indexPersonalLibraryFullText"]>>;
    });
    button.click();
    await vi.waitFor(() => expect(plugin.openPersonalLibraryDirectionReview).toHaveBeenCalledOnce());
  });

  it("returns non-empty definitions with section groups", () => {
    const { tab } = makeTab();
    const items = tab.getSettingDefinitions();
    expect(items.length).toBeGreaterThanOrEqual(4);
    const groups = items.filter((item) => item.type === "group");
    expect(groups.map((g) => g.heading)).toEqual(
      expect.arrayContaining([
        "LLM",
        "Personal library",
        "Output & schedule",
        "Email delivery",
        "Advanced",
        "Help & feedback",
      ]),
    );
    expect(groups.map((g) => g.heading)).not.toContain("Embedding");
    expect(groups.map((g) => g.heading)).not.toContain("PDF parsing");
    const llmIndex = groups.findIndex((g) => g.heading === "LLM");
    const libraryIndex = groups.findIndex((g) => g.heading === "Personal library");
    expect(libraryIndex).toBeGreaterThan(llmIndex);
  });

  it("keeps the personal library section after Output & schedule and before Email delivery", () => {
    const { tab } = makeTab();
    const headings = tab
      .getSettingDefinitions()
      .filter((item) => item.type === "group" || item.type === "list")
      .map((item) => item.heading);
    expect(headings).toEqual([
      "LLM",
      "arXiv categories",
      "Research topics",
      "Output & schedule",
      "Personal library",
      "Email delivery",
      "Advanced",
      "Help & feedback",
    ]);
    const scheduleIndex = headings.indexOf("Output & schedule");
    expect(headings[scheduleIndex + 1]).toBe("Personal library");
    expect(headings[scheduleIndex + 2]).toBe("Email delivery");
  });

  it("contains the api key row, model row, topics list, and email to control", () => {
    const { tab } = makeTab();
    const items = tab.getSettingDefinitions();
    const names: string[] = [];
    walkItems(items, (item) => {
      if (typeof item.name === "string") names.push(item.name);
    });
    expect(names).toContain("API key");
    expect(names).toContain("Model");
    const topicsList = items.find(
      (item) => item.type === "list" && item.heading === "Research topics",
    );
    expect(topicsList).toBeDefined();
    expect(topicsList?.addItem?.name).toBe("Add topic");

    const emailGroup = items.find(
      (item) => item.type === "group" && item.heading === "Email delivery",
    );
    const emailTo = emailGroup?.items.find((item) => item.name === "Your email");
    expect(emailTo).toHaveProperty("render");
    expect(emailTo).not.toHaveProperty("action");
  });

  it("wires every render callback so complex rows are present", () => {
    const { tab } = makeTab();
    const names = tab
      .getSettingDefinitions()
      .flatMap((item) => {
        if (item.type === "group") {
          return [item, ...item.items].map((sub) => sub.name);
        }
        return [item.name];
      });
    expect(names).toEqual(
      expect.arrayContaining([
        "Enable · Paused",
        "Library",
        "Embedding",
        "Timezone",
        "Run window",
        "Check every (minutes)",
        "Resend API key",
        "From email",
        "From name",
        "Daily auto-send",
      ]),
    );
    expect(names).not.toContain("Embedding API base URL");
    expect(names).not.toContain("Better PDF parser");
    expect(names).not.toContain("Sidecar capability URL");
    expect(names).not.toContain("Library connection");
    expect(names).not.toContain("Embedding mode");
    expect(names).not.toContain("Use local parser sidecar");
  });

  it("renders the setup guide as a full-width row without a second title", () => {
    const { tab } = makeTab();
    const guideRow = tab.getSettingDefinitions()[0] as Record<string, unknown>;
    expect(guideRow.name).toBe("");
    expect(guideRow).toHaveProperty("render");
    const setting = new Setting(tab.containerEl);
    renderSetupGuideRow(tab, setting);
    expect(setting.settingEl.classList).toContain("arxiv-daily-settings__setup-guide-host");
    expect(setting.settingEl.querySelectorAll(".arxiv-daily-setup")).toHaveLength(1);
  });

  it("offers category deletion only while more than one category remains", () => {
    const { tab, settings } = makeTab();
    const categoriesList = () =>
      tab.getSettingDefinitions().find(
        (item) => item.type === "list" && item.heading === "arXiv categories",
      ) as { onDelete?: unknown } | undefined;
    settings.arxiv.categories = ["astro-ph"];
    expect(categoriesList()?.onDelete).toBeUndefined();
    settings.arxiv.categories = ["astro-ph", "gr-qc"];
    expect(categoriesList()?.onDelete).toEqual(expect.any(Function));
  });

  it("names guide sections as they appear on the page", () => {
    const { tab, settings } = makeTab();
    const text = tab.createSetupGuide()?.textContent ?? "";
    expect(text).toContain("under LLM");
    settings.llm.apiKey = "key";
    settings.arxiv.category = "";
    settings.arxiv.categories = [];
    expect(tab.createSetupGuide()?.textContent).toContain("under arXiv categories");
    expect(text).not.toContain("under AI model");
  });

  it("routes list mutations and actions to the tab's public methods", async () => {
    const { tab, settings, saveSettings } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});

    const topicsList = tab
      .getSettingDefinitions()
      .find(
        (item) => item.type === "list" && item.heading === "Research topics",
      );
    await topicsList?.addItem?.action(document.createElement("button"));
    await vi.waitFor(() => expect(tab.refreshSettings).toHaveBeenCalledTimes(1));
    expect(settings.arxiv.topics).toHaveLength(1);
    expect(saveSettings).toHaveBeenCalledTimes(1);

    expect(topicsList?.onReorder).toBeUndefined();
  });
});

describe("personal library guide row", () => {
  function libraryGroupItems(tab: ArxivDailySettingTab) {
    const group = tab.getSettingDefinitions().find(
      (item) => item.type === "group" && item.heading === "Personal library",
    ) as { items: Array<Record<string, unknown>> } | undefined;
    return group?.items ?? [];
  }

  it("shows the intro box as the first item while no library is connected", () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({ kind: "disconnected" });
    const items = libraryGroupItems(tab);
    expect(items[0]?.name).toBe("");
    expect(items[0]).toHaveProperty("render");
    expect(items[1]?.name).toBe("Library");

    const setting = new Setting(tab.containerEl);
    renderLibraryGuideRow(tab, setting);
    expect(setting.settingEl.classList).toContain("arxiv-daily-settings__library-guide-host");
    expect(setting.settingEl.querySelectorAll(".arxiv-daily-settings__library-guide")).toHaveLength(1);
    expect(setting.settingEl.textContent).toContain("Choose a folder of PDFs");
  });

  it("keeps the intro box as the first item once a folder is chosen (always visible, like the email guide)", () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    const items = libraryGroupItems(tab);
    expect(items[0]?.name).toBe("");
    expect(items[1]?.name).toBe("Library");
  });

  it("keeps the intro box as the first item once authorized (always visible, like the email guide)", () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: new Date().toISOString(),
    });
    const items = libraryGroupItems(tab);
    expect(items[0]?.name).toBe("");
    expect(items[1]?.name).toBe("Library");
  });
});

describe("personal library directions row", () => {
  function libraryGroupItems(tab: ArxivDailySettingTab) {
    const group = tab.getSettingDefinitions().find(
      (item) => item.type === "group" && item.heading === "Personal library",
    ) as { items: Array<Record<string, unknown>> } | undefined;
    return group?.items ?? [];
  }

  it("is hidden while no library folder is chosen", () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({ kind: "disconnected" });
    const items = libraryGroupItems(tab);
    expect(items.some((item) => item.name === "Research directions")).toBe(false);
  });

  it("shows directly below Library once a folder is chosen, before any remote authorization", () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    const items = libraryGroupItems(tab);
    const libraryIndex = items.findIndex((item) => item.name === "Library");
    const directionsIndex = items.findIndex((item) => item.name === "Research directions");
    expect(libraryIndex).toBeGreaterThanOrEqual(0);
    expect(directionsIndex).toBe(libraryIndex + 1);
    expect(items[directionsIndex]?.desc).toEqual(expect.stringContaining("daily reports"));
  });

  it("also shows once fully authorized", () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: new Date().toISOString(),
    });
    const items = libraryGroupItems(tab);
    expect(items.some((item) => item.name === "Research directions")).toBe(true);
  });

  it("reflects saved topic directions without loading a retired profile", () => {
    const { tab, plugin } = makeTab();
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: new Date().toISOString(),
    });
    plugin.settings.arxiv.topics = [normalizeTopic({
      id: "topic", name: "Agents", tag: "agents", directions: [
        { id: "a", text: "Reliable agents", origin: "manual" },
        { id: "b", text: "Agent evaluation", origin: "library" },
      ],
    })];
    const items = libraryGroupItems(tab);
    const row = items.find((item) => item.name === "Research directions");
    expect(row?.desc).toContain("2 saved directions");
  });

  it("opens the direction review modal when its button is clicked", async () => {
    const { tab, plugin } = makeTab();
    const setting = new Setting(tab.containerEl);
    renderLibraryDirectionsRow(tab, setting);
    const buttons = [...setting.settingEl.querySelectorAll("button")];
    expect(buttons).toHaveLength(1);
    expect(buttons[0]?.textContent).toBe("Review directions");
    buttons[0]?.click();
    await vi.waitFor(() => expect(plugin.openPersonalLibraryDirectionReview).toHaveBeenCalled());
  });
});

describe("personal library settings row", () => {
  function renderLibraryButtons(tab: ArxivDailySettingTab) {
    const buttons: Array<{
      text: string;
      cta: boolean;
      warning: boolean;
      disabled: boolean;
      click?: () => void;
    }> = [];
    // Attached to the document because the row refuses to write progress into
    // a description that is no longer on screen, and a detached node is exactly
    // what that check is for.
    const descEl = document.createElement("div");
    document.body.appendChild(descEl);
    const setting = {
      controlEl: { addClass: vi.fn() },
      descEl,
      setDesc: vi.fn().mockReturnThis(),
      addButton(callback: (button: any) => void) {
        const state = { text: "", cta: false, warning: false, disabled: false } as {
          text: string;
          cta: boolean;
          warning: boolean;
          disabled: boolean;
          click?: () => void;
        };
        const button = {
          buttonEl: { getBoundingClientRect: vi.fn(() => ({ left: 0, bottom: 0 })) },
          setButtonText(text: string) { state.text = text; return button; },
          setCta() { state.cta = true; return button; },
          setWarning() { state.warning = true; return button; },
          setDisabled(disabled: boolean) { state.disabled = disabled; return button; },
          onClick(click: () => void) { state.click = click; return button; },
        };
        callback(button);
        buttons.push(state);
        return setting;
      },
    };
    tab.renderLibraryConnectionControls(setting as never);
    return { buttons, setting };
  }

  it("keeps the library buttons on one right-aligned row", () => {
    const { tab, plugin } = makeTab();
    plugin.settings.embedding.mode = "local";
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    const { setting } = renderLibraryButtons(tab);
    expect(setting.controlEl.addClass).toHaveBeenCalledWith(
      "arxiv-daily-settings__library-controls",
    );
  });

  it("shows only folder selection while disconnected", () => {
    const { tab } = makeTab();
    const { buttons, setting } = renderLibraryButtons(tab);
    expect(buttons.map((button) => button.text)).toEqual(["Choose folder"]);
    expect(setting.setDesc).toHaveBeenCalledWith(
      expect.stringMatching(/Choose a folder of PDFs/i),
    );
  });

  it("offers Build index after a local folder is selected without requiring authorization", () => {
    const { tab, plugin } = makeTab();
    plugin.settings.embedding.mode = "local";
    const runAction = vi.spyOn(tab, "runAction").mockImplementation(() => {});
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    const { buttons, setting } = renderLibraryButtons(tab);
    expect(buttons.map((button) => button.text)).toEqual([
      "Change folder",
      "Build index",
    ]);
    expect(buttons[1]?.cta).toBe(true);
    expect(setting.setDesc).toHaveBeenCalledWith(
      expect.stringMatching(/Local embedding stays on this device/i),
    );
    buttons[1]?.click?.();
    expect(runAction).toHaveBeenCalledWith(
      "index personal library titles and abstracts",
      expect.any(Function),
    );
  });

  it("keeps Build index as the main action while remote consent is still pending", () => {
    const { tab, plugin } = makeTab();
    plugin.settings.embedding.mode = "remote";
    const runAction = vi.spyOn(tab, "runAction").mockImplementation(() => {});
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    const pending = renderLibraryButtons(tab);
    expect(pending.buttons.map((button) => button.text)).toEqual([
      "Change folder",
      "Build index",
    ]);
    expect(pending.buttons[1]?.cta).toBe(true);
    // The row still says the remote grant is missing; only the button changed.
    expect(pending.setting.setDesc).toHaveBeenCalledWith(
      expect.stringMatching(/Remote embedding sends titles and abstracts/i),
    );
    pending.buttons[1]?.click?.();
    expect(runAction).toHaveBeenCalledWith(
      "index personal library titles and abstracts",
      expect.any(Function),
    );

    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: "2026-08-02T12:00:00.000Z",
    });
    const authorized = renderLibraryButtons(tab);
    expect(authorized.buttons.map((button) => button.text)).toEqual([
      "Change folder",
      "Build index",
      "Revoke",
    ]);
    expect(authorized.buttons[1]?.cta).toBe(true);
    authorized.buttons[1]?.click?.();
    expect(runAction).toHaveBeenCalledWith(
      "index personal library titles and abstracts",
      expect.any(Function),
    );
  });

  it("exposes Revoke as a direct button only while authorized", () => {
    const { tab, plugin } = makeTab();
    plugin.settings.embedding.mode = "local";
    const runAction = vi.spyOn(tab, "runAction").mockImplementation(() => {});
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    const pending = renderLibraryButtons(tab).buttons;
    expect(pending.map((button) => button.text)).not.toContain("Revoke");

    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: "2026-08-02T12:00:00.000Z",
    });
    const authorized = renderLibraryButtons(tab).buttons;
    const revoke = authorized.find((button) => button.text === "Revoke");
    expect(revoke).toBeDefined();
    expect(revoke?.cta).toBe(false);
    revoke?.click?.();
    expect(runAction).toHaveBeenCalledWith(
      "revoke personal library",
      expect.any(Function),
    );
  });

  it("never renders a Manage menu or an authorization button on the library row", () => {
    const { tab, plugin } = makeTab();
    const openMenu = vi.spyOn(MenuItem.prototype, "setTitle");
    for (const status of [
      { kind: "disconnected" } as const,
      { kind: "authorization-required", rootLabel: "papers" } as const,
      { kind: "authorization-invalidated", rootLabel: "papers" } as const,
      {
        kind: "authorized",
        rootLabel: "papers",
        grantedAt: "2026-08-02T12:00:00.000Z",
      } as const,
    ]) {
      for (const mode of ["local", "remote"] as const) {
        plugin.settings.embedding.mode = mode;
        vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue(status);
        const { buttons } = renderLibraryButtons(tab);
        const labels = buttons.map((button) => button.text);
        expect(labels).not.toContain("Manage…");
        expect(buttons.length).toBeLessThanOrEqual(3);
        for (const label of labels) {
          expect(label.toLowerCase()).not.toContain("authoriz");
        }
        // The main action is always folder selection or indexing.
        expect(["Choose folder", "Build index"]).toContain(labels.at(-1) === "Revoke"
          ? labels.at(-2)
          : labels.at(-1));
      }
    }
    expect(openMenu).not.toHaveBeenCalled();
    openMenu.mockRestore();
  });

  it("shows Build index for a legacy remote library that was never granted", () => {
    const { tab, plugin } = makeTab();
    plugin.settings.embedding.mode = "remote";
    const runAction = vi.spyOn(tab, "runAction").mockImplementation(() => {});
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-invalidated",
      rootLabel: "papers",
    });
    const { buttons, setting } = renderLibraryButtons(tab);

    expect(buttons.map((button) => button.text)).toEqual([
      "Change folder",
      "Build index",
    ]);
    expect(setting.setDesc).toHaveBeenCalledWith(
      expect.stringMatching(/changed/i),
    );
    buttons[1]?.click?.();
    expect(runAction).toHaveBeenCalledWith(
      "index personal library titles and abstracts",
      expect.any(Function),
    );
  });

  /**
   * Indexing already reported itself to the status bar, which the settings
   * modal covers. These say the row that started the run is now the row that
   * shows it, and that it can stop it.
   */
  describe("while an index run is in flight", () => {
    function connectedTab() {
      const made = makeTab();
      made.plugin.settings.embedding.mode = "local";
      vi.mocked(made.plugin.getLibraryConnectionStatus).mockReturnValue({
        kind: "authorization-required",
        rootLabel: "papers",
      });
      vi.spyOn(made.tab, "refreshSettings").mockImplementation(() => {});
      return made;
    }

    it("shows the run on the row and stops it from there", () => {
      const { tab, plugin } = connectedTab();
      plugin.libraryIndexStatus.beginRun("personal-library-fulltext-index:1", "indexing");
      plugin.libraryIndexStatus.report({
        phase: "extracting and embedding PDF text",
        completed: 4,
        total: 40,
      });

      const { buttons } = renderLibraryButtons(tab);
      expect(buttons.map((button) => button.text)).toEqual([
        "Change folder",
        "Indexing… (4/40)",
        "Cancel",
      ]);
      expect(buttons[1]?.disabled).toBe(true);
      expect(buttons[0]?.disabled).toBe(true);

      buttons[2]?.click?.();
      expect(plugin.cancelPersonalLibraryIndexing).toHaveBeenCalled();
    });

    /**
     * A run reports once per paper. Re-rendering the settings page at that rate
     * would throw away whatever else is being edited on it, so a report that
     * only changes text writes straight into the row that is already drawn —
     * and is paced, because the reports arrive faster than they can be read.
     */
    it("rewrites the row in place rather than re-rendering the page", async () => {
      vi.useFakeTimers();
      try {
        const { tab, plugin } = connectedTab();
        plugin.libraryIndexStatus.beginRun("personal-library-fulltext-index:1", "indexing");
        const { setting } = renderLibraryButtons(tab);
        vi.mocked(tab.refreshSettings).mockClear();

        for (let paper = 1; paper <= 20; paper += 1) {
          plugin.libraryIndexStatus.report({
            phase: "extracting and embedding PDF text",
            completed: paper,
            total: 40,
          });
        }
        expect(tab.refreshSettings).not.toHaveBeenCalled();
        expect(setting.descEl.textContent).toBe("");

        vi.advanceTimersByTime(LIBRARY_INDEX_PROGRESS_INTERVAL_MS);
        // One write, carrying the newest report rather than the first.
        expect(setting.descEl.textContent).toContain("extracting and embedding PDF text");
        expect(tab.refreshSettings).not.toHaveBeenCalled();
      } finally {
        vi.useRealTimers();
      }
    });

    /** Starting and stopping change which buttons exist, which text cannot do. */
    it("re-renders when the run appears and when it is over", () => {
      const { tab, plugin } = connectedTab();
      renderLibraryButtons(tab);
      vi.mocked(tab.refreshSettings).mockClear();

      plugin.libraryIndexStatus.beginRun("personal-library-fulltext-index:1", "indexing");
      expect(tab.refreshSettings).toHaveBeenCalledTimes(1);

      // The re-render the tab would have done is stubbed out here, so tell the
      // row what it would have learned: the run is on it now.
      renderLibraryButtons(tab);
      vi.mocked(tab.refreshSettings).mockClear();
      plugin.libraryIndexStatus.endRun();
      expect(tab.refreshSettings).toHaveBeenCalledTimes(1);
    });

    it("stops following the run once the tab is closed", () => {
      const { tab, plugin } = connectedTab();
      renderLibraryButtons(tab);
      vi.mocked(tab.refreshSettings).mockClear();
      tab.hide();
      plugin.libraryIndexStatus.beginRun("personal-library-fulltext-index:1", "indexing");
      expect(tab.refreshSettings).not.toHaveBeenCalled();
    });
  });
});

describe("declarative embedding rows delegate the consent flow", () => {
  it("routes the mode dropdown through the shared in-place consent", async () => {
    const { tab, plugin } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    const apply = vi
      .spyOn(tab, "applyEmbeddingModeChange")
      .mockImplementation(async () => false);
    const setting = renderSetting();
    renderEmbeddingModeRow(tab, setting as never);
    const select = setting.controlEl.querySelector("select") as HTMLSelectElement;

    select.value = "remote";
    select.dispatchEvent(new Event("change"));

    await vi.waitFor(() => expect(apply).toHaveBeenCalledWith("remote"));
    // A declined switch leaves the dropdown showing the unchanged mode.
    await vi.waitFor(() => expect(select.value).toBe(plugin.settings.embedding.mode));
    expect(select.value).toBe("local");
  });

  it("labels the local option honestly about the one-time model download, not as always-offline", () => {
    const { tab } = makeTab();
    const setting = renderSetting();
    renderEmbeddingModeRow(tab, setting as never);
    const options = [...setting.controlEl.querySelectorAll("option")] as HTMLOptionElement[];
    const local = options.find((option) => option.value === "local");
    expect(local?.textContent).toBe("Local (default, one-time model download)");
    expect(local?.textContent).not.toMatch(/offline, default/);
  });

  it("describes the local embedding row's one-time download and approximate size, not a bundled model", () => {
    const { tab } = makeTab();
    const group = tab.getSettingDefinitions().find(
      (item) => item.type === "group" && item.heading === "Personal library",
    ) as { items: Array<{ name: string; desc?: string }> } | undefined;
    const embeddingItem = group?.items.find((item) => item.name === "Embedding");
    expect(embeddingItem?.desc).toContain("downloads its model once");
    expect(embeddingItem?.desc).toContain("130 MB");
    expect(embeddingItem?.desc).not.toContain("bundled");
  });

  it("routes the endpoint field through the shared re-ask on change", async () => {
    const { tab, plugin } = makeTab();
    plugin.settings.embedding.mode = "remote";
    plugin.settings.embedding.baseUrl = "https://embed.example.com/v1";
    const save = vi
      .spyOn(tab, "saveEmbeddingEndpointField")
      .mockResolvedValue("https://embed.example.com/v1");
    const setting = renderSetting();
    renderEmbeddingBaseUrlRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "https://elsewhere.example.com/v1";
    input.dispatchEvent(new Event("change"));

    await vi.waitFor(() => expect(save).toHaveBeenCalledWith(
      "embedding.baseUrl",
      "https://elsewhere.example.com/v1",
    ));
    // A declined change puts the authorized endpoint back on screen.
    await vi.waitFor(() => expect(input.value).toBe("https://embed.example.com/v1"));
  });
});

describe("declarative LLM and category rows", () => {
  it("persists a base URL candidate and restores the input when persistence fails", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const original = settings.llm.baseUrl;
    saveSettings.mockImplementationOnce(async (candidate: unknown) => {
      expect(candidate).not.toBe(settings);
      expect((candidate as typeof settings).llm.baseUrl).toBe("https://candidate.example/v1");
      expect(settings.llm.baseUrl).toBe(original);
      throw new Error("disk full");
    });
    const setting = renderSetting();
    renderLlmBaseUrlRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = " https://candidate.example/v1 ";
    input.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(saveSettings).toHaveBeenCalledTimes(1);
      expect(settings.llm.baseUrl).toBe(original);
      expect(input.value).toBe(original);
    });
  });

  it("does not let an earlier failed base URL save overwrite a later draft or commit", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const firstSave = deferred();
    const secondSave = deferred();
    saveSettings
      .mockImplementationOnce(() => firstSave.promise)
      .mockImplementationOnce(() => secondSave.promise);
    vi.spyOn(tab, "refreshDeclarativeSetupGuide").mockImplementation(() => {});
    const setting = renderSetting();
    renderLlmBaseUrlRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "https://rejected.example/v1";
    input.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    input.value = "https://accepted.example/v1";
    input.dispatchEvent(new Event("change"));
    firstSave.reject(new Error("first save failed"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));
    expect(input.value).toBe("https://accepted.example/v1");

    secondSave.resolve();
    await vi.waitFor(() => {
      expect(settings.llm.baseUrl).toBe("https://accepted.example/v1");
      expect(input.value).toBe("https://accepted.example/v1");
    });
  });

  it("saves a newer sensitive draft after an earlier save fails without stale restoration", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const firstSave = deferred();
    const secondSave = deferred();
    saveSettings
      .mockImplementationOnce(() => firstSave.promise)
      .mockImplementationOnce(() => secondSave.promise);
    vi.spyOn(tab, "refreshDeclarativeSetupGuide").mockImplementation(() => {});
    const setting = renderSetting();
    renderApiKeyRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "rejected-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    input.value = "accepted-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    firstSave.reject(new Error("first save failed"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));
    expect(input.value).toBe("accepted-secret");

    secondSave.resolve();
    await vi.waitFor(() => {
      expect(settings.llm.apiKey).toBe("accepted-secret");
      expect(input.value).toBe("accepted-secret");
    });
  });

  it("captures a rejected sensitive save triggered by Enter", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const setting = renderSetting();
    renderApiKeyRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "rejected-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

    await vi.waitFor(() => expect(plugin.logger.error).toHaveBeenCalledWith(
      "settings: save LLM API key failed",
      expect.any(Error),
    ));
    expect(settings.llm.apiKey).toBe("");
    expect(input.value).toBe("");
  });

  it.each([
    {
      name: "LLM API key",
      key: "llm.apiKey",
      configure: (settings: typeof DEFAULT_SETTINGS) => { settings.llm.apiKey = "stored-llm-secret"; },
      current: (settings: typeof DEFAULT_SETTINGS) => settings.llm.apiKey,
      render: renderApiKeyRow,
      draft: "new-llm-secret",
    },
    {
      name: "Resend API key",
      key: "email.apiKey",
      configure: (settings: typeof DEFAULT_SETTINGS) => { settings.email.apiKey = "stored-resend-secret"; },
      current: (settings: typeof DEFAULT_SETTINGS) => settings.email.apiKey,
      render: renderEmailApiKeyRow,
      draft: "new-resend-secret",
    },
    {
      name: "verification code",
      key: "email.hostedToken",
      configure: (settings: typeof DEFAULT_SETTINGS) => {
        settings.email.mode = "hosted";
        settings.email.hostedToken = "stored-hosted-secret";
      },
      current: (settings: typeof DEFAULT_SETTINGS) => settings.email.hostedToken,
      render: renderHostedTokenRow,
      draft: "new-hosted-secret",
    },
  ])(
    "reveals the persisted $name, masks it by default, and saves drafts transactionally",
    async ({ key, configure, current, render, draft }) => {
      const { tab, settings, saveSettings } = makeTab();
      configure(settings);
      vi.spyOn(tab, "refreshDeclarativeSetupGuide").mockImplementation(() => {});
      const setting = renderSetting();
      render(tab, setting as never);

      const input = setting.controlEl.querySelector("input") as HTMLInputElement;
      const buttons = Array.from(setting.controlEl.querySelectorAll("button"));
      const replace = buttons.find((button) => button.textContent === "Replace");
      const clear = buttons.find((button) => button.textContent === "Clear");
      const reveal = buttons.find((button) =>
        button.getAttribute("aria-label")?.startsWith("Show"),
      ) as HTMLButtonElement;

      expect(input.type).toBe("password");
      expect(input.value).toContain("stored-");
      expect(input.value).not.toBe("Configured");
      expect(replace).toBeUndefined();
      expect(clear).toBeUndefined();
      expect(reveal).toBeDefined();

      reveal.click();
      expect(input.type).toBe("text");
      expect(input.value).toContain("stored-");
      reveal.click();
      expect(input.type).toBe("password");

      input.value = draft;
      input.dispatchEvent(new Event("input"));
      input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

      await vi.waitFor(() => expect(current(settings)).toBe(draft));
      expect(input.value).toBe(draft);
      expect(saveSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          [key.split(".")[0]!]: expect.objectContaining({
            [key.split(".")[1]!]: draft,
          }),
        }),
      );
    },
  );

  it("maps None and Medium to thinking mode plus effort", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const setting = renderSetting();
    renderReasoningEffortRow(tab, setting as never);
    const select = setting.controlEl.querySelector("select") as HTMLSelectElement;

    select.value = "none";
    select.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(settings.llm.thinkingMode).toBe(false));

    select.value = "medium";
    select.dispatchEvent(new Event("change"));
    await vi.waitFor(() => {
      expect(settings.llm.thinkingMode).toBe(true);
      expect(settings.llm.reasoningEffort).toBe("medium");
    });
    expect(saveSettings).toHaveBeenCalledTimes(2);
  });

  it("renders each category as a fixed dropdown without custom input", () => {
    const { tab } = makeTab();
    const setting = renderSetting();
    renderCategoryRow(tab, setting as never, 0);
    expect(setting.controlEl.querySelector("select")).not.toBeNull();
    expect(setting.controlEl.querySelector("input")).toBeNull();
  });

  it("integrates hosted verification into the email row", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    settings.email.mode = "hosted";
    const setting = renderSetting();
    renderEmailToRow(tab, setting as never);

    const input = setting.controlEl.querySelector("input") as HTMLInputElement;
    const button = Array.from(setting.controlEl.querySelectorAll("button"))
      .find((item) => item.textContent === "Send verification") as HTMLButtonElement;
    expect(button).toBeDefined();
    input.value = "  me@example.com  ";
    button.click();

    await vi.waitFor(() => {
      expect(settings.email.to).toBe("me@example.com");
      expect(saveSettings).toHaveBeenCalledTimes(1);
      expect(plugin.sendHostedVerificationEmail).toHaveBeenCalledTimes(1);
    });
  });

  it("waits for email persistence before sending verification", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    settings.email.mode = "hosted";
    let finishSave!: () => void;
    saveSettings.mockImplementationOnce(() => new Promise<void>((resolve) => {
      finishSave = resolve;
    }));
    const setting = renderSetting();
    renderEmailToRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;
    const button = Array.from(setting.controlEl.querySelectorAll("button"))
      .find((item) => item.textContent === "Send verification") as HTMLButtonElement;

    input.value = "new@example.com";
    input.dispatchEvent(new Event("change"));
    button.click();
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    expect(plugin.sendHostedVerificationEmail).not.toHaveBeenCalled();

    finishSave();
    await vi.waitFor(() => {
      expect(plugin.sendHostedVerificationEmail).toHaveBeenCalledTimes(1);
    });
  });

  it("rolls back email and skips verification when persistence fails", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    settings.email.mode = "hosted";
    settings.email.to = "old@example.com";
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const setting = renderSetting();
    renderEmailToRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;
    const button = Array.from(setting.controlEl.querySelectorAll("button"))
      .find((item) => item.textContent === "Send verification") as HTMLButtonElement;

    input.value = "new@example.com";
    input.dispatchEvent(new Event("change"));
    button.click();
    await vi.waitFor(() => {
      expect(settings.email.to).toBe("old@example.com");
      expect(input.value).toBe("old@example.com");
    });
    expect(plugin.sendHostedVerificationEmail).not.toHaveBeenCalled();
  });

  it("uses a masked verification code and saves it before sending a test", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    settings.email.mode = "hosted";
    settings.email.hostedToken = "old-token";
    const setting = renderSetting();
    renderHostedTokenRow(tab, setting as never);

    const input = setting.controlEl.querySelector("input") as HTMLInputElement;
    const buttons = Array.from(setting.controlEl.querySelectorAll("button"));
    const reveal = buttons.find((item) =>
      item.getAttribute("aria-label") === "Show verification code",
    ) as HTMLButtonElement;
    const sendTest = buttons.find((item) => item.textContent === "Send test") as HTMLButtonElement;
    expect(input.type).toBe("password");
    expect(input.value).toBe("old-token");
    expect(setting.controlEl.textContent).not.toContain("Replace");
    expect(setting.controlEl.textContent).not.toContain("Clear");

    reveal.click();
    expect(input.type).toBe("text");
    input.value = " new token \n value ";
    input.dispatchEvent(new Event("input"));
    sendTest.click();

    await vi.waitFor(() => {
      expect(settings.email.hostedToken).toBe("newtokenvalue");
      expect(saveSettings).toHaveBeenCalledTimes(1);
      expect(plugin.sendTestEmail).toHaveBeenCalledTimes(1);
    });
  });

  it("rolls back a verification code when persistence fails", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    settings.email.mode = "hosted";
    settings.email.hostedToken = "old-token";
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const setting = renderSetting();
    renderHostedTokenRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "new-token";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await vi.waitFor(() => {
      expect(settings.email.hostedToken).toBe("old-token");
      expect(input.value).toBe("old-token");
    });
    expect(plugin.refreshSensitiveValues).not.toHaveBeenCalled();
  });

  it("puts the self-mode test action inside the Resend API key row", () => {
    const { tab } = makeTab();
    const setting = renderSetting();
    renderEmailApiKeyRow(tab, setting as never);
    expect(
      Array.from(setting.controlEl.querySelectorAll("button"))
        .some((button) => button.textContent === "Send test"),
    ).toBe(true);
  });

  it("restores the hidden Resend key when candidate persistence fails", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    settings.email.apiKey = "old-resend-secret";
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const setting = renderSetting();
    renderEmailApiKeyRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "new-resend-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

    await vi.waitFor(() => expect(plugin.logger.error).toHaveBeenCalled());
    expect(settings.email.apiKey).toBe("old-resend-secret");
    expect(plugin.refreshSensitiveValues).not.toHaveBeenCalled();
    expect(input.value).toBe("old-resend-secret");
  });

  it("keeps a custom timezone as a local draft and rejects invalid commits", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const setting = renderSetting();
    renderTimezoneRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "Mars/Olympus_Mons";
    input.dispatchEvent(new Event("input"));
    expect(settings.arxiv.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);
    expect(saveSettings).not.toHaveBeenCalled();

    input.dispatchEvent(new Event("change"));
    await Promise.resolve();
    expect(input.validationMessage).toMatch(/timezone/i);
    expect(input.value).toBe("Mars/Olympus_Mons");
    expect(settings.arxiv.timezone).toBe(DEFAULT_SETTINGS.arxiv.timezone);
    expect(saveSettings).not.toHaveBeenCalled();
  });

  it.each(["change", "blur"])(
    "commits a valid custom timezone on %s",
    async (eventName) => {
      const { tab, settings, saveSettings } = makeTab();
      const setting = renderSetting();
      renderTimezoneRow(tab, setting as never);
      const input = setting.controlEl.querySelector("input") as HTMLInputElement;

      input.value = "Europe/Paris";
      input.dispatchEvent(new Event(eventName));

      await vi.waitFor(() => {
        expect(settings.arxiv.timezone).toBe("Europe/Paris");
        expect(saveSettings).toHaveBeenCalledTimes(1);
      });
    },
  );

  it("commits a valid custom timezone on Enter", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();
    const setting = renderSetting();
    renderTimezoneRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "Europe/Paris";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

    await vi.waitFor(() => {
      expect(settings.arxiv.timezone).toBe("Europe/Paris");
      expect(saveSettings).toHaveBeenCalledTimes(1);
    });
    expect(plugin.logger.setTimezone).toHaveBeenCalledWith("Europe/Paris");
  });

  it.each(["reject", "resolve"] as const)(
    "queues a newer declarative timezone draft when the older draft will %s",
    async (firstOutcome) => {
      const { tab, settings, saveSettings } = makeTab();
      const firstSave = deferred();
      const secondSave = deferred();
      saveSettings
        .mockImplementationOnce(() => firstSave.promise)
        .mockImplementationOnce(() => secondSave.promise);
      const setting = renderSetting();
      renderTimezoneRow(tab, setting as never);
      const input = setting.controlEl.querySelector("input") as HTMLInputElement;

      input.value = "Europe/Paris";
      input.dispatchEvent(new Event("change"));
      await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
      input.value = "Europe/Berlin";
      input.dispatchEvent(new Event("blur"));
      if (firstOutcome === "reject") firstSave.reject(new Error("first failed"));
      else firstSave.resolve();

      await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));
      expect(input.value).toBe("Europe/Berlin");
      secondSave.resolve();
      await vi.waitFor(() => {
        expect(settings.arxiv.timezone).toBe("Europe/Berlin");
        expect(input.value).toBe("");
      });
    },
  );

  it("coalesces duplicate timezone events but keeps a distinct later draft queued", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const firstSave = deferred();
    const secondSave = deferred();
    saveSettings
      .mockImplementationOnce(() => firstSave.promise)
      .mockImplementationOnce(() => secondSave.promise);
    const setting = renderSetting();
    renderTimezoneRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "Europe/Paris";
    input.dispatchEvent(new Event("change"));
    input.dispatchEvent(new Event("blur"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    input.value = "Europe/Berlin";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new Event("change"));
    firstSave.resolve();

    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));
    expect(input.value).toBe("Europe/Berlin");
    secondSave.resolve();
    await vi.waitFor(() => expect(settings.arxiv.timezone).toBe("Europe/Berlin"));
  });

  it("restores the latest successful timezone when a newer distinct draft fails", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const firstSave = deferred();
    const secondSave = deferred();
    saveSettings
      .mockImplementationOnce(() => firstSave.promise)
      .mockImplementationOnce(() => secondSave.promise);
    const setting = renderSetting();
    renderTimezoneRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;
    const select = setting.controlEl.querySelector("select") as HTMLSelectElement;

    input.value = "Europe/Paris";
    input.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    firstSave.resolve();
    await vi.waitFor(() => expect(settings.arxiv.timezone).toBe("Europe/Paris"));

    input.value = "Europe/Berlin";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));
    secondSave.reject(new Error("second failed"));

    await vi.waitFor(() => expect(input.value).toBe("Europe/Paris"));
    expect(select.value).toBe("Europe/Paris");
    expect(settings.arxiv.timezone).toBe("Europe/Paris");
  });

  it("restores the tick interval input after a rejected transaction", async () => {
    const { tab, settings, saveSettings } = makeTab();
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const setting = renderSetting();
    renderTickIntervalRow(tab, setting as never);
    const input = setting.controlEl.querySelector("input") as HTMLInputElement;

    input.value = "5";
    input.dispatchEvent(new Event("change"));
    input.dispatchEvent(new Event("blur"));

    await vi.waitFor(() => expect(input.value).toBe(
      String(DEFAULT_SETTINGS.schedule.tickIntervalMin),
    ));
    expect(saveSettings).toHaveBeenCalledTimes(1);
    expect(settings.schedule.tickIntervalMin).toBe(DEFAULT_SETTINGS.schedule.tickIntervalMin);
  });

  it("restores a run-window select after a rejected transaction", async () => {
    const { tab, settings, saveSettings } = makeTab();
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const setting = renderSetting();
    renderRunWindowRow(tab, setting as never);
    const start = setting.controlEl.querySelector("select") as HTMLSelectElement;

    start.value = "10:00";
    start.dispatchEvent(new Event("change"));

    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    expect(settings.schedule.runAtLocal).toBe(DEFAULT_SETTINGS.schedule.runAtLocal);
    expect(start.value).toBe(DEFAULT_SETTINGS.schedule.runAtLocal);
  });

  it("keeps the newer run-window draft when an older save fails and the newer save succeeds", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const firstSave = deferred();
    const secondSave = deferred();
    saveSettings
      .mockImplementationOnce(() => firstSave.promise)
      .mockImplementationOnce(() => secondSave.promise);
    const setting = renderSetting();
    renderRunWindowRow(tab, setting as never);
    const start = setting.controlEl.querySelector("select") as HTMLSelectElement;

    start.value = "10:00";
    start.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    start.value = "11:00";
    start.dispatchEvent(new Event("change"));
    firstSave.reject(new Error("first failed"));

    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));
    expect(start.value).toBe("11:00");
    secondSave.resolve();
    await vi.waitFor(() => expect(settings.schedule.runAtLocal).toBe("11:00"));
    expect(start.value).toBe("11:00");
  });

  it("restores the latest successful run-window value when a newer save fails", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const firstSave = deferred();
    const secondSave = deferred();
    saveSettings
      .mockImplementationOnce(() => firstSave.promise)
      .mockImplementationOnce(() => secondSave.promise);
    const setting = renderSetting();
    renderRunWindowRow(tab, setting as never);
    const start = setting.controlEl.querySelector("select") as HTMLSelectElement;

    start.value = "10:00";
    start.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    firstSave.resolve();
    await vi.waitFor(() => expect(settings.schedule.runAtLocal).toBe("10:00"));

    start.value = "11:00";
    start.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));
    secondSave.reject(new Error("second failed"));

    await vi.waitFor(() => expect(start.value).toBe("10:00"));
    expect(settings.schedule.runAtLocal).toBe("10:00");
  });

  it("restores the schedule toggle after a rejected transaction", async () => {
    ToggleComponent.reset();
    const { tab, settings, saveSettings } = makeTab();
    settings.llm.apiKey = "configured";
    settings.arxiv.topics.push(normalizeTopic({
      id: "topic-1",
      name: "Language models",
      tag: "language-models",
      description: "Research about language models",
      detail: false,
    }));
    saveSettings.mockRejectedValueOnce(new Error("disk full"));
    const refresh = vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    const setting = renderSetting();
    renderScheduleEnabledRow(tab, setting as never);
    const toggle = ToggleComponent.instances.at(-1)!;

    await expect(toggle.trigger(true)).resolves.toBeUndefined();

    expect(settings.schedule.enabled).toBe(false);
    expect(toggle.value).toBe(false);
    expect(refresh).toHaveBeenCalledTimes(1);
  });
});

describe("declarative topic cards", () => {
  it("preserves the settings viewport across a declarative refresh", () => {
    const { tab } = makeTab();
    const viewport = document.createElement("div");
    viewport.style.overflowY = "auto";
    viewport.appendChild(tab.containerEl);
    document.body.appendChild(viewport);
    viewport.scrollTop = 180;
    vi.spyOn(tab, "update").mockImplementation(() => {});

    tab.refreshSettings();

    expect(viewport.scrollTop).toBe(180);
    viewport.remove();
  });

  it("reveals and focuses a newly added topic after the list refreshes", async () => {
    const { tab, settings } = makeTab();
    document.body.appendChild(tab.containerEl);
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {
      const topic = settings.arxiv.topics.at(-1)!;
      const card = document.createElement("div");
      card.className = "arxiv-daily-settings__topic-card";
      card.dataset.arxivDailyTopicId = topic.id;
      const scrollIntoView = vi.fn();
      Object.defineProperty(card, "scrollIntoView", { value: scrollIntoView });
      const input = document.createElement("input");
      input.className = "arxiv-daily-settings__topic-name-input";
      card.appendChild(input);
      tab.containerEl.appendChild(card);
    });

    await tab.addTopic();
    await new Promise<void>((resolve) => queueMicrotask(resolve));

    expect(document.activeElement).toBe(
      tab.containerEl.querySelector(".arxiv-daily-settings__topic-name-input"),
    );
    expect(
      tab.containerEl.querySelector<HTMLElement>(
        ".arxiv-daily-settings__topic-card",
      )?.scrollIntoView,
    ).toHaveBeenCalledWith({ block: "nearest", behavior: "auto" });
    tab.containerEl.remove();
  });

  it("keeps the next topic fixed in the viewport when deleting a topic", async () => {
    const { tab, settings } = makeTab();
    settings.arxiv.topics.push(
      normalizeTopic({ id: "first", name: "First", tag: "first", description: "First", detail: false }),
      normalizeTopic({ id: "next", name: "Next", tag: "next", description: "Next", detail: false }),
    );
    const viewport = document.createElement("div");
    viewport.style.overflowY = "auto";
    viewport.scrollTop = 500;
    viewport.appendChild(tab.containerEl);
    document.body.appendChild(viewport);
    const makeCard = (topicId: string, top: number) => {
      const card = document.createElement("div");
      card.className = "arxiv-daily-settings__topic-card";
      card.dataset.arxivDailyTopicId = topicId;
      Object.defineProperty(card, "getBoundingClientRect", {
        value: () => ({ top }),
      });
      return card;
    };
    tab.containerEl.append(
      makeCard("first", 0),
      makeCard("next", 200),
    );
    vi.spyOn(tab, "confirmReplace").mockResolvedValue(true);
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {
      tab.containerEl.replaceChildren(makeCard("next", 0));
    });

    await tab.deleteTopic(0);

    expect(viewport.scrollTop).toBe(300);
    viewport.remove();
  });

  it("keeps topic fields focused while updating the setup guide", async () => {
    const { tab, settings, saveSettings } = makeTab();
    settings.arxiv.topics.push(
      normalizeTopic({
        id: "topic-1",
        name: "",
        tag: "topic-1",
        detail: false,
        directions: [{ id: "d1", text: "Existing.", origin: "manual" }],
      }),
    );
    const refresh = vi.spyOn(tab, "refreshSettings");
    document.body.appendChild(tab.containerEl);
    const guideSetting = new Setting(tab.containerEl);
    renderSetupGuideRow(tab, guideSetting);
    const topicSetting = new Setting(tab.containerEl);
    tab.renderTopicRow(topicSetting, 0);

    const header = topicSetting.settingEl.querySelector(
      ".arxiv-daily-settings__topic-header",
    ) as HTMLButtonElement;
    header.click();
    const fields = [
      topicSetting.settingEl.querySelector(
        ".arxiv-daily-settings__topic-name-input",
      ),
      topicSetting.settingEl.querySelector(
        ".arxiv-daily-settings__topic-direction-input",
      ),
    ] as Array<HTMLInputElement | HTMLTextAreaElement>;

    for (const [index, field] of fields.entries()) {
      field.focus();
      field.value = `draft-${index}`;
      field.dispatchEvent(new Event("input"));
      await vi.waitFor(() => {
        expect(saveSettings).toHaveBeenCalledTimes(index + 1);
      });
      expect(document.activeElement).toBe(field);
    }

    expect(refresh).not.toHaveBeenCalled();
    tab.containerEl.remove();
  });

  it("hides the tag chip while collapsed so a long name keeps the header width", () => {
    const { tab, settings } = makeTab();
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Something", origin: "manual" as const }],
      id: "topic-1",
      name: "A very long research topic name that would otherwise get truncated",
      tag: "long-topic-tag",
      description: "Something",
      detail: false,
    });
    const topicSetting = new Setting(tab.containerEl);
    tab.renderTopicRow(topicSetting, 0);

    expect(
      topicSetting.settingEl.querySelector(".arxiv-daily-settings__topic-tag"),
    ).toBeNull();

    const header = topicSetting.settingEl.querySelector(
      ".arxiv-daily-settings__topic-header",
    ) as HTMLButtonElement;
    header.click();

    expect(topicSetting.settingEl.querySelector(".arxiv-daily-settings__topic-tag")).toBeNull();

    header.click();

    expect(
      topicSetting.settingEl.querySelector(".arxiv-daily-settings__topic-tag"),
    ).toBeNull();
  });
});

describe("topic tags and blocked first report", () => {
  it("never gives a new topic a tag another topic already has", async () => {
    const { tab, settings } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    vi.spyOn(tab, "confirmReplace").mockResolvedValue(true);
    await tab.addTopic();
    await tab.addTopic();
    await tab.deleteTopic(0);

    await tab.addTopic();

    const tags = settings.arxiv.topics.map((topic) => topic.tag);
    expect(new Set(tags).size).toBe(tags.length);
  });

  it("derives a new topic's tag from its name and avoids collisions", async () => {
    const { tab, settings } = makeTab();
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Existing", origin: "manual" as const }],
      id: "existing",
      name: "Dark matter",
      tag: "dark-matter",
      description: "Existing",
      detail: false,
    });
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    await tab.addTopic();
    const setting = new Setting(tab.containerEl);
    tab.renderTopicRow(setting, 1);
    const name = setting.settingEl.querySelector(
      ".arxiv-daily-settings__topic-name-input",
    ) as HTMLInputElement;

    name.value = "Dark matter";
    name.dispatchEvent(new Event("input"));

    await vi.waitFor(() => expect(settings.arxiv.topics[1]!.tag).toBe("dark-matter-2"));
  });

  it("names what blocks the first report when the earlier steps look done", () => {
    const { tab, settings } = makeTab();
    settings.llm.apiKey = "sk-test";
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Galaxy evolution", origin: "manual" as const }],
      id: "t1",
      name: "Galaxies",
      tag: "galaxies",
      description: "Galaxy evolution",
      detail: false,
    });
    settings.arxiv.categories = ["astro-ph", "astro-ph"];

    const guide = tab.createSetupGuide();
    const firstReportStep = Array.from(
      guide?.querySelectorAll(".arxiv-daily-setup__item") ?? [],
    ).find((item) => item.textContent?.includes("Generate your first report"));

    expect(
      firstReportStep?.querySelector(".arxiv-daily-setup__description")?.textContent,
    ).toMatch(/Duplicate arXiv category: astro-ph/);
  });
});

describe("declarative text rows commit when editing ends", () => {
  function rowFor(tab: ArxivDailySettingTab, name: string): Setting {
    let found: Setting | undefined;
    walkItems(tab.getSettingDefinitions(), (item) => {
      if (item.name !== name) return;
      expect(item, `${name} renders its own input`).toHaveProperty("render");
      const setting = new Setting(tab.containerEl);
      (item.render as (setting: Setting) => void)(setting);
      found = setting;
    });
    expect(found, name).toBeDefined();
    return found!;
  }

  function type(input: HTMLInputElement, text: string): void {
    for (const character of text) {
      input.value += character;
      input.dispatchEvent(new Event("input"));
    }
  }

  it.each([
    ["Daily reports folder", "output.dailyDir", "notes/daily", "notes/daily"],
    ["Paper notes folder", "output.papersDir", "notes/papers", "notes/papers"],
    ["From email", "email.fromEmail", " me@example.com ", "me@example.com"],
    ["From name", "email.fromName", "Papers", "Papers"],
  ])("%s saves once on change, not per keystroke", async (name, key, typed, saved) => {
    const { tab, settings } = makeTab();
    // Output folders need store preparation the mock plugin lacks; record the value.
    const change = vi
      .spyOn(tab.plugin.settingsChanges, "changeValue")
      .mockImplementation(async (changedKey, value) => {
        writeSettingValue(settings, changedKey, value);
      });
    const input = rowFor(tab, name).controlEl.querySelector("input")!;

    input.value = "";
    type(input, typed);
    expect(change).not.toHaveBeenCalled();

    input.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(change).toHaveBeenCalledTimes(1));
    expect(readSettingValue(settings, key)).toBe(saved);
  });

  it("marks an unsafe folder draft invalid and does not save it", async () => {
    const { tab, settings } = makeTab();
    const change = vi.spyOn(tab.plugin.settingsChanges, "changeValue");
    const input = rowFor(tab, "Daily reports folder").controlEl.querySelector("input")!;

    input.value = "../outside";
    input.dispatchEvent(new Event("input"));
    expect(input.classList).toContain("is-invalid");
    input.dispatchEvent(new Event("change"));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(change).not.toHaveBeenCalled();
    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
  });
});

describe("topic and category saves that fail", () => {
  async function noticesDuring(run: () => Promise<unknown>): Promise<string> {
    const { Notice } = await import("obsidian");
    const calls = (Notice as unknown as { calls: Array<{ message: string }> }).calls;
    calls.length = 0;
    await run();
    return calls.map((call) => call.message).join("\n");
  }

  it("rolls back an added topic and reports", async () => {
    const { tab, settings, saveSettings } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    saveSettings.mockRejectedValueOnce(new Error("disk full"));

    const notices = await noticesDuring(() => tab.addTopic());

    expect(settings.arxiv.topics).toEqual([]);
    expect(notices).toMatch(/disk full/);
  });

  it("rolls back a deleted topic and reports", async () => {
    const { tab, settings, saveSettings } = makeTab();
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "A", origin: "manual" as const }], id: "t1", name: "A", tag: "a", description: "A", detail: false });
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    vi.spyOn(tab, "confirmReplace").mockResolvedValue(true);
    saveSettings.mockRejectedValueOnce(new Error("disk full"));

    const notices = await noticesDuring(() => tab.deleteTopic(0));

    expect(settings.arxiv.topics.map((topic) => topic.id)).toEqual(["t1"]);
    expect(notices).toMatch(/disk full/);
  });

  it("rolls back an added category and reports", async () => {
    const { tab, settings, saveSettings } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    saveSettings.mockRejectedValueOnce(new Error("disk full"));

    const notices = await noticesDuring(() => tab.addCategory());

    expect(settings.arxiv.categories).toEqual(["astro-ph"]);
    expect(notices).toMatch(/disk full/);
  });

  it("reports a failed topic field save", async () => {
    const { tab, settings, saveSettings } = makeTab();
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "A", origin: "manual" as const }], id: "t1", name: "A", tag: "a", description: "A", detail: false });
    const setting = new Setting(tab.containerEl);
    tab.renderTopicRow(setting, 0);
    const input = setting.settingEl.querySelector(
      ".arxiv-daily-settings__topic-direction-input",
    ) as HTMLTextAreaElement;
    saveSettings.mockRejectedValueOnce(new Error("disk full"));

    const notices = await noticesDuring(async () => {
      input.value = "B";
      input.dispatchEvent(new Event("input"));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(notices).toMatch(/disk full/);
  });
});

describe("category rows", () => {
  it("refuses a category another row already has instead of dropping a row", async () => {
    const { Notice } = await import("obsidian");
    const notices = (Notice as unknown as { calls: Array<{ message: string }> }).calls;
    notices.length = 0;
    const { tab, settings } = makeTab();
    settings.arxiv.categories = ["astro-ph", "gr-qc"];
    const refresh = vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    const setting = new Setting(tab.containerEl);
    renderCategoryRow(tab, setting, 1);
    const select = setting.controlEl.querySelector("select")!;

    select.value = "astro-ph";
    select.dispatchEvent(new Event("change"));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(settings.arxiv.categories).toEqual(["astro-ph", "gr-qc"]);
    expect(select.value).toBe("gr-qc");
    expect(notices.map((call) => call.message).join("\n")).toMatch(/already/i);
    expect(refresh).not.toHaveBeenCalled();
  });
});

describe("model field", () => {
  it("accepts a typed model name and saves it when editing ends", async () => {
    const { tab, settings } = makeTab();
    const setting = new Setting(tab.containerEl);
    renderModelRow(tab, setting);
    const input = setting.controlEl.querySelector<HTMLInputElement>(
      "input.arxiv-daily-settings__model-input",
    );
    expect(input).not.toBeNull();

    input!.value = "my-";
    input!.dispatchEvent(new Event("input"));
    input!.value = "my-model";
    input!.dispatchEvent(new Event("input"));
    expect(settings.llm.model).toBe(DEFAULT_SETTINGS.llm.model);
    input!.dispatchEvent(new Event("change"));

    await vi.waitFor(() => expect(settings.llm.model).toBe("my-model"));
  });

  it("offers fetched models as suggestions without replacing the current one", async () => {
    const { tab, plugin, settings } = makeTab();
    (plugin as unknown as { getHttpClient: () => unknown }).getHttpClient = () => ({});
    const fetchModels = vi
      .spyOn(LlmClient.prototype, "fetchModels")
      .mockResolvedValue(["provider-a", "provider-b"]);
    document.body.appendChild(tab.containerEl);
    const setting = new Setting(tab.containerEl);
    renderModelRow(tab, setting);
    const button = Array.from(setting.controlEl.querySelectorAll("button"))
      .find((candidate) => candidate.textContent === "Get models")!;

    button.click();

    await vi.waitFor(() => expect(fetchModels).toHaveBeenCalled());
    await vi.waitFor(() => {
      const options = Array.from(
        setting.settingEl.querySelectorAll<HTMLOptionElement>("datalist option"),
      ).map((option) => option.value);
      expect(options).toEqual(["provider-a", "provider-b"]);
    });
    expect(settings.llm.model).toBe(DEFAULT_SETTINGS.llm.model);
    fetchModels.mockRestore();
    tab.containerEl.remove();
  });
});

describe("local parser sidecar address", () => {
  it("moves both endpoints when one moves to another port", async () => {
    const { tab, settings } = makeTab();
    settings.pdfParserSidecar.enabled = true;
    const setting = new Setting(tab.containerEl);
    renderPdfParserSidecarCapabilitiesUrlRow(tab, setting);
    const input = setting.controlEl.querySelector("input")!;

    input.value = "http://127.0.0.1:5002/v1/capabilities";
    input.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(settings.pdfParserSidecar.capabilitiesUrl).toBe(
        "http://127.0.0.1:5002/v1/capabilities",
      );
      expect(settings.pdfParserSidecar.parseUrl).toBe("http://127.0.0.1:5002/v1/parse");
    });
  });
});

describe("first report date", () => {
  function readyTab(recentDates: unknown) {
    const made = makeTab();
    made.settings.llm.apiKey = "sk-test";
    made.settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Galaxy evolution", origin: "manual" as const }],
      id: "t1",
      name: "Galaxies",
      tag: "galaxies",
      description: "Galaxy evolution",
      detail: false,
    });
    const runForDateNow = vi.fn(async () => ({
      kind: "completed" as const,
      papersWritten: 1,
    }));
    const plugin = made.plugin as unknown as Record<string, unknown>;
    plugin.scheduler = { runForDateNow };
    plugin.recentDates = recentDates;
    (made.plugin.logger as unknown as { info: () => void }).info = vi.fn();
    vi.spyOn(made.tab, "refreshSetupGuide").mockImplementation(() => {});
    return { ...made, runForDateNow };
  }

  function recentDates(dates: string[]) {
    return {
      refresh: vi.fn().mockResolvedValue(undefined),
      snapshot: () => ({ status: "ready", dates: new Set(dates), refreshedAt: 1 }),
    };
  }

  afterEach(() => {
    vi.useRealTimers();
  });

  it("uses the latest announced day on a weekend", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-09-26T04:00:00Z")); // Saturday noon in Shanghai
    const { tab, runForDateNow } = readyTab(
      recentDates(["2026-09-22", "2026-09-25", "2026-09-24"]),
    );

    await tab.generateFirstReport();

    expect(runForDateNow).toHaveBeenCalledWith("2026-09-25");
  });

  it("ignores announced days after today", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-09-24T04:00:00Z"));
    const { tab, runForDateNow } = readyTab(
      recentDates(["2026-09-23", "2026-09-25"]),
    );

    await tab.generateFirstReport();

    expect(runForDateNow).toHaveBeenCalledWith("2026-09-23");
  });

  it("refreshes open dashboards once the first report finishes", async () => {
    const { tab, plugin } = readyTab(recentDates([]));
    const refreshFromVault = vi.fn(async () => undefined);
    (plugin as unknown as { app: unknown }).app = {
      workspace: { getLeavesOfType: vi.fn(() => [{ view: { refreshFromVault } }]) },
    };

    await tab.generateFirstReport();

    expect(refreshFromVault).toHaveBeenCalledOnce();
  });

  it("falls back to today when the announced days cannot be loaded", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-09-26T04:00:00Z"));
    const { tab, runForDateNow } = readyTab({
      refresh: vi.fn().mockRejectedValue(new Error("offline")),
      snapshot: () => ({ status: "failed", dates: new Set(), refreshedAt: 1 }),
    });

    await tab.generateFirstReport();

    expect(runForDateNow).toHaveBeenCalledWith("2026-09-26");
  });
});

describe("declarative setup guide refresh after setup", () => {
  function completeSetup() {
    const made = makeTab();
    made.settings.llm.apiKey = "sk-test";
    made.settings.schedule.enabled = true;
    made.settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Galaxy evolution", origin: "manual" as const }],
      id: "t1",
      name: "Galaxies",
      tag: "galaxies",
      description: "Galaxy evolution",
      detail: false,
    });
    (made.plugin as unknown as { stateStore: unknown }).stateStore = {
      snapshot: () => ({
        "2026-09-23": { status: "completed", lastAttempt: 1, attempts: 1 },
      }),
    };
    return made;
  }

  it("shows the completion summary once, persists the marker, then leaves the page alone", () => {
    const { tab, settings } = completeSetup();
    // First evaluation: every milestone just became true at once, so the
    // one-time completion summary still shows and the marker gets set.
    expect(tab.shouldShowSetupGuide()).toBe(true);
    const guide = tab.createSetupGuide();
    expect(guide?.className).toContain("arxiv-daily-setup--complete");
    expect(settings.onboarding.guideCompleted).toBe(true);

    const refresh = vi.spyOn(tab, "refreshSettings");
    expect(tab.shouldShowSetupGuide()).toBe(false);

    settings.arxiv.topics[0]!.name = "Galaxies and clusters";
    tab.refreshSetupGuide();

    expect(refresh).not.toHaveBeenCalled();
  });

  it("does not bring the full guide back once completed; a compact warning explains a later break instead", () => {
    const { tab, settings } = completeSetup();
    tab.createSetupGuide(); // completes and persists the marker
    expect(settings.onboarding.guideCompleted).toBe(true);

    settings.arxiv.topics[0]!.directions = [];
    const guide = tab.createSetupGuide();

    expect(guide).not.toBeNull();
    expect(guide?.querySelector(".arxiv-daily-setup__list")).toBeNull();
    expect(guide?.textContent).toMatch(/has no directions/i);
  });
});

describe("declarative setup guide actions", () => {
  /** A rendered 1.13 group: Obsidian puts the definition's `cls` on it. */
  function renderGroup(tab: ArxivDailySettingTab, cls: string): HTMLElement {
    const group = document.createElement("div");
    group.className = `setting-group ${cls}`;
    Object.defineProperty(group, "scrollIntoView", { value: vi.fn() });
    tab.containerEl.appendChild(group);
    return group;
  }

  function clickGuideButton(tab: ArxivDailySettingTab, label: string): void {
    const button = Array.from(
      tab.containerEl.querySelectorAll<HTMLButtonElement>(".arxiv-daily-setup button"),
    ).find((candidate) => candidate.textContent === label);
    expect(button, `guide button ${label}`).toBeDefined();
    button!.click();
  }

  function sectionClass(tab: ArxivDailySettingTab, heading: string): string {
    const section = tab
      .getSettingDefinitions()
      .find((item) => (item.type === "group" || item.type === "list") && item.heading === heading);
    const cls = section && "cls" in section ? section.cls : undefined;
    expect(cls, `${heading} section class`).toEqual(expect.any(String));
    return cls!;
  }

  it("gives the sections the guide points at a stable class", () => {
    const { tab } = makeTab();
    const classes = ["LLM", "arXiv categories", "Research topics"].map(
      (heading) => sectionClass(tab, heading),
    );
    expect(new Set(classes).size).toBe(3);
  });

  it("scrolls to the LLM group and focuses the empty API key on Connect AI", () => {
    const { tab } = makeTab();
    document.body.appendChild(tab.containerEl);
    const guideSetting = new Setting(tab.containerEl);
    renderSetupGuideRow(tab, guideSetting);
    const group = renderGroup(tab, sectionClass(tab, "LLM"));
    const apiKeySetting = new Setting(group);
    renderApiKeyRow(tab, apiKeySetting);

    clickGuideButton(tab, "Connect AI");

    expect(group.scrollIntoView).toHaveBeenCalled();
    expect(document.activeElement).toBe(
      group.querySelector('input[aria-label="LLM API key"]'),
    );
    tab.containerEl.remove();
  });

  it("opens the first incomplete topic and focuses its first empty field", () => {
    const { tab, settings } = makeTab();
    settings.llm.apiKey = "sk-test";
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Complete topic", origin: "manual" as const }],
      id: "done",
      name: "Done",
      tag: "done",
      description: "Complete topic",
      detail: false,
    }, { directions: [],
      id: "draft",
      name: "Draft",
      tag: "draft",
      description: "",
      detail: false,
    });
    document.body.appendChild(tab.containerEl);
    renderSetupGuideRow(tab, new Setting(tab.containerEl));
    const group = renderGroup(tab, sectionClass(tab, "Research topics"));
    tab.renderTopicRow(new Setting(group), 0);
    tab.renderTopicRow(new Setting(group), 1);

    clickGuideButton(tab, "Describe interests");

    const draftCard = group.querySelector<HTMLElement>(
      '[data-arxiv-daily-topic-id="draft"]',
    )!;
    expect(group.scrollIntoView).toHaveBeenCalled();
    expect(draftCard.querySelector<HTMLElement>(".arxiv-daily-settings__topic-form")?.hidden)
      .toBe(false);
    expect(document.activeElement).toBe(
      draftCard.querySelector(".arxiv-daily-settings__topic-direction-add"),
    );
    tab.containerEl.remove();
  });

  it("adds a first topic when Describe interests has none to open", () => {
    const { tab, settings } = makeTab();
    settings.llm.apiKey = "sk-test";
    document.body.appendChild(tab.containerEl);
    renderSetupGuideRow(tab, new Setting(tab.containerEl));
    renderGroup(tab, sectionClass(tab, "Research topics"));
    const addTopic = vi.spyOn(tab, "addTopic").mockResolvedValue(undefined);

    clickGuideButton(tab, "Describe interests");

    expect(addTopic).toHaveBeenCalledTimes(1);
    tab.containerEl.remove();
  });

  it("turns on daily runs from the last step", async () => {
    const { tab, plugin, settings } = makeTab();
    settings.llm.apiKey = "sk-test";
    plugin.stateStore.snapshot = () => ({ "2026-10-01": { status: "completed", attempts: 1, lastAttempt: 1 } });
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Galaxy evolution", origin: "manual" as const }],
      id: "t1",
      name: "Galaxies",
      tag: "galaxies",
      description: "Galaxy evolution",
      detail: false,
    });
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    document.body.appendChild(tab.containerEl);
    renderSetupGuideRow(tab, new Setting(tab.containerEl));
    expect(tab.containerEl.textContent).toMatch(/4 of 5 complete/);

    clickGuideButton(tab, "Turn on daily reports");

    await vi.waitFor(() => {
      expect(plugin.setScheduleEnabled).toHaveBeenCalledWith(true);
      expect(settings.schedule.enabled).toBe(true);
    });
    tab.containerEl.remove();
  });

  it("shows the first report as running and ignores repeat clicks", async () => {
    const { tab, plugin, settings } = makeTab();
    settings.llm.apiKey = "sk-test";
    settings.arxiv.topics.push({ directions: [{ id: "fixture-direction", text: "Galaxy evolution", origin: "manual" as const }],
      id: "t1",
      name: "Galaxies",
      tag: "galaxies",
      description: "Galaxy evolution",
      detail: false,
    });
    const run = deferred();
    const runForDateNow = vi.fn(async () => {
      await run.promise;
      return { kind: "completed" as const, papersWritten: 3 };
    });
    (plugin as unknown as { scheduler: unknown }).scheduler = { runForDateNow };
    (plugin.logger as unknown as { info: () => void }).info = vi.fn();
    document.body.appendChild(tab.containerEl);
    renderSetupGuideRow(tab, new Setting(tab.containerEl));

    clickGuideButton(tab, "Generate first report");
    await vi.waitFor(() => expect(runForDateNow).toHaveBeenCalledTimes(1));
    const busy = Array.from(
      tab.containerEl.querySelectorAll<HTMLButtonElement>(".arxiv-daily-setup button"),
    ).find((button) => button.textContent === "Generating…");
    expect(busy?.disabled).toBe(true);
    await tab.generateFirstReport();
    expect(runForDateNow).toHaveBeenCalledTimes(1);

    run.resolve();
    await vi.waitFor(() => {
      expect(tab.containerEl.textContent).not.toContain("Generating…");
    });
    tab.containerEl.remove();
  });

  it("says where to go instead of doing nothing when a section is missing", async () => {
    const { Notice } = await import("obsidian");
    const notices = (Notice as unknown as { calls: Array<{ message: string }> }).calls;
    notices.length = 0;
    const { tab } = makeTab();
    document.body.appendChild(tab.containerEl);
    renderSetupGuideRow(tab, new Setting(tab.containerEl));

    clickGuideButton(tab, "Connect AI");

    expect(notices.map((call) => call.message).join("\n")).toMatch(/LLM/);
    tab.containerEl.remove();
  });
});

describe("next setup step", () => {
  it("offers only the first incomplete task, then advances as settings become ready", () => {
    const { tab, settings } = makeTab();
    settings.llm.apiKey = "";
    settings.arxiv.topics = [];
    let guide = tab.createSetupGuide();
    expect(Array.from(guide.querySelectorAll("ol button"), (button) => button.textContent))
      .toEqual(["Connect AI"]);
    expect(guide.querySelectorAll(".arxiv-daily-setup__description")).toHaveLength(1);

    settings.llm.apiKey = "test-key";
    settings.llm.model = "test-model";
    guide = tab.createSetupGuide();
    expect(Array.from(guide.querySelectorAll("ol button"), (button) => button.textContent))
      .toEqual(["Describe interests"]);

    settings.arxiv.topics = [normalizeTopic({
      id: "research", name: "Research", tag: "research", detail: false,
      directions: [{ id: "direction", text: "Reliable research agents", origin: "manual" }],
    })];
    guide = tab.createSetupGuide();
    expect(Array.from(guide.querySelectorAll("ol button"), (button) => button.textContent))
      .toEqual(["Generate first report"]);
  });

  it("moves focus to the current section on the declarative settings page", () => {
    const { tab, settings } = makeTab();
    settings.llm.apiKey = "test-key";
    settings.llm.model = "test-model";
    settings.arxiv.topics = [];
    const topics = tab.getSettingDefinitions().find((item) =>
      item.type === "list" && item.heading === "Research topics");
    const section = document.createElement("section");
    section.className = topics?.cls ?? "";
    section.scrollIntoView = vi.fn();
    tab.containerEl.appendChild(section);
    document.body.appendChild(tab.containerEl);
    const guide = tab.createSetupGuide();
    tab.containerEl.prepend(guide);
    guide.querySelector<HTMLButtonElement>("ol button")!.click();
    expect(document.activeElement).toBe(section);
    expect(section.scrollIntoView).toHaveBeenCalledOnce();
    tab.containerEl.remove();
  });
});

describe("wired getControlValue", () => {
  it("resolves every registered key against the nested settings", () => {
    const { tab } = makeTab();
    for (const key of allSettingKeys()) {
      expect(tab.getControlValue(key)).toBeDefined();
      expect(tab.getControlValue(key)).toBe(
        readSettingValue(tab.plugin.settings, key),
      );
    }
  });

  it("resolves every control key used by the declarative items", () => {
    const { tab } = makeTab();
    walkItems(tab.getSettingDefinitions(), (item) => {
      const control = item.control as { key?: string } | undefined;
      if (control?.key) {
        expect(tab.getControlValue(control.key)).toBe(
          readSettingValue(tab.plugin.settings, control.key),
        );
      }
    });
  });

  it("is consistent with buildSettingDefinitions on the same settings", () => {
    const { tab, plugin } = makeTab();
    const hostItems = buildSettingDefinitions({
      plugin,
      showSetupGuide: true,
      renderSetupGuideRow: () => {},
      renderLlmBaseUrlRow: () => {},
      renderApiKeyRow: () => {},
      renderModelRow: () => {},
      renderReasoningEffortRow: () => {},
      renderLibraryConnectionRow: () => {},
      renderCategoryRow: () => {},
      renderTopicRow: () => {},
      renderTimezoneRow: () => {},
      addCategory: () => {},
      deleteCategory: () => {},
      addTopic: () => {},
      renderScheduleEnabledRow: () => {},
      renderRunWindowRow: () => {},
      renderTickIntervalRow: () => {},
      renderEmailGuideRow: () => {},
      renderEmailModeRow: () => {},
      renderEmailToRow: () => {},
      renderEmailApiKeyRow: () => {},
      renderHostedTokenRow: () => {},
      renderEmbeddingModeRow: () => {},
      renderEmbeddingBaseUrlRow: () => {},
      renderEmbeddingApiKeyRow: () => {},
      renderEmbeddingModelRow: () => {},
      renderEmbeddingDimensionRow: () => {},
      renderPdfParserSidecarEnabledRow: () => {},
      renderPdfParserSidecarCapabilitiesUrlRow: () => {},
      renderPdfParserSidecarParseUrlRow: () => {},
    });
    const tabItems = tab.getSettingDefinitions();
    expect(tabItems.length).toBe(hostItems.length);
    const keyOf = (item: unknown): string =>
      (item as { name?: string }).name ?? "";
    expect(tabItems.map(keyOf)).toEqual(hostItems.map(keyOf));
  });
});

describe("shared legacy and declarative runtime-coupled changes", () => {
  it("uses the transaction service for timezone, interval, and log level", async () => {
    const { tab, plugin, settings, saveSettings } = makeTab();

    await tab.saveTimezone("Europe/Paris");
    await tab.saveTickInterval("7");
    await tab.saveLogLevel("debug");

    expect(settings.arxiv.timezone).toBe("Europe/Paris");
    expect(settings.schedule.tickIntervalMin).toBe(7);
    expect(settings.advanced.logLevel).toBe("debug");
    expect(saveSettings).toHaveBeenCalledTimes(3);
    expect(plugin.logger.setTimezone).toHaveBeenCalledWith("Europe/Paris");
    expect(plugin.restartScheduler).toHaveBeenCalledTimes(1);
    expect(plugin.logger.setLevel).toHaveBeenCalledWith("debug");
  });
});

describe("wired setControlValue", () => {
  it("writes through the dotted key and persists via saveSettings", async () => {
    const { tab, settings, saveSettings } = makeTab();
    await tab.setControlValue(SETTING_KEYS.llm.model, "deepseek-r1");
    expect(settings.llm.model).toBe("deepseek-r1");
    expect(saveSettings).toHaveBeenCalledTimes(1);

    await tab.setControlValue(SETTING_KEYS.llm.thinkingMode, false);
    expect(settings.llm.thinkingMode).toBe(false);
    expect(saveSettings).toHaveBeenCalledTimes(2);
  });

  it("round-trips every registered key through setControlValue", async () => {
    const { tab, settings } = makeTab();
    for (const key of allSettingKeys()) {
      const value = readSettingValue(settings, key);
      await tab.setControlValue(key, value);
      expect(readSettingValue(settings, key)).toEqual(value);
    }
  });

  it("trims email to and fromEmail but not fromName, mirroring display()", async () => {
    const { tab, settings } = makeTab();
    await tab.setControlValue(SETTING_KEYS.email.to, "  me@example.com  ");
    expect(settings.email.to).toBe("me@example.com");

    await tab.setControlValue(
      SETTING_KEYS.email.fromEmail,
      "  sender@example.com  ",
    );
    expect(settings.email.fromEmail).toBe("sender@example.com");

    await tab.setControlValue(SETTING_KEYS.email.fromName, "  arXiv Daily  ");
    expect(settings.email.fromName).toBe("  arXiv Daily  ");
  });

  it("rejects a declarative daily/papers collision without persistence", async () => {
    const { tab, settings, saveSettings } = makeTab();

    await expect(
      tab.setControlValue(
        SETTING_KEYS.output.dailyDir,
        settings.output.papersDir.toUpperCase(),
      ),
    ).rejects.toThrow(/daily and papers directories/i);

    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(saveSettings).not.toHaveBeenCalled();
  });

  it("does not refresh a stale declarative value over a later queued change", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const firstSave = deferred();
    const secondSave = deferred();
    saveSettings
      .mockImplementationOnce(() => firstSave.promise)
      .mockImplementationOnce(() => secondSave.promise);
    const update = vi.spyOn(tab, "update").mockImplementation(() => {});

    const first = tab.setControlValue(SETTING_KEYS.output.summaryLanguage, "en");
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));
    const second = tab.setControlValue(SETTING_KEYS.output.summaryLanguage, "en");
    firstSave.reject(new Error("first save failed"));
    await expect(first).rejects.toThrow("first save failed");
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(2));

    expect(update).not.toHaveBeenCalled();
    secondSave.resolve();
    await second;
    expect(settings.output.summaryLanguage).toBe("en");
  });

  it("restores the declarative displayed value when persistence fails", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const update = vi.spyOn(tab, "update").mockImplementation(() => {});
    saveSettings.mockRejectedValueOnce(new Error("disk full"));

    const failure = await tab
      .setControlValue(SETTING_KEYS.output.summaryLanguage, "en")
      .catch((error: unknown) => error);

    expect(settings.output.summaryLanguage).toBe("zh");
    expect(tab.restoreControlValue(failure, SETTING_KEYS.output.summaryLanguage)).toBe("zh");
    expect(update).toHaveBeenCalledTimes(1);
  });
});

describe("real acceptance refresh preserves topic editing", () => {
  function openHostReview() {
    // The mock 1.13 renderer is empty. Select the actual legacy renderer for
    // this host integration, including its real guide and refresh behavior.
    vi.spyOn(obsidian, "requireApiVersion").mockReturnValue(false);
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.arxiv.topics = [normalizeTopic({
      id: "existing-topic", name: "Existing topic", tag: "existing-topic", detail: false,
      directions: [{ id: "existing-direction", text: "Original handwritten direction", origin: "manual" }],
    })];
    const fingerprint = `sha256:${"a".repeat(64)}`;
    const evidence = `sha256:${"b".repeat(64)}`;
    const papers = [1, 2].map((number) => ({ paperKey: `arxiv:2608.0000${number}`, evidenceFingerprint: evidence }));
    const proposal: PersonalLibraryDirectionProposal = {
      schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION, revision: 0,
      proposalId: "host-proposal", scopeFingerprint: fingerprint, identificationFingerprint: fingerprint,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: fingerprint, identificationFingerprint: fingerprint, catalogInputPapers: papers,
      }),
      catalogInputPapers: papers, generationContractFingerprint: fingerprint, generatedAt: "2026-09-07T00:00:00.000Z",
      topics: [{
        id: "proposed-topic", suggestedName: "Existing topic", targetTopicId: "existing-topic",
        directions: [{
          id: "accepted-direction", text: "Accepted library direction", discoveryCues: ["library evidence"],
          representatives: papers, representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(papers),
          lineage: { candidateIds: ["accepted-direction"] },
          clusterMembers: papers.map(({ paperKey }) => ({ paperKey, confidence: 1 })),
        }],
      }],
    };
    const saved: Array<{ settings: PluginSettings; libraryProposalAcceptances?: ProposalAcceptanceReceipt[] }> = [];
    const saveData = vi.fn(async (data: typeof saved[number]) => { saved.push(structuredClone(data)); });
    Object.assign(plugin, {
      app: {} as App, settings, saveData,
      automaticEmailSupported: () => true,
      manifest: { id: "arxiv-daily", version: "0.0.0-test" },
      logger: { error: vi.fn(), setSensitiveValues: vi.fn() },
      stateStore: { snapshot: () => ({}) }, libraryIndexStatus: new LibraryIndexStatusStore(),
      libraryCatalog: createEmptyPersonalLibraryCatalog(fingerprint, fingerprint),
      libraryIndexedPapers: papers.map(({ paperKey }) => ({ paperKey, title: `Paper ${paperKey}` })),
      libraryProposal: proposal, librarySuggestions: null, libraryProposalAcceptances: [],
      libraryCatalogLoadError: null, libraryProposalLoadError: null, librarySuggestionsLoadError: null,
      libraryProposalAcceptanceLoadError: null,
      libraryConnection: undefined, libraryConnectionRevision: 0, libraryOutputRevision: 0,
      libraryMutationRevision: 0, libraryMutationQueue: Promise.resolve(),
    });
    // Retain the real host persistence envelope and library queue, the same
    // lock order the real acceptance path uses; only disk I/O is controlled.
    const storage = plugin as unknown as {
      enqueueLibraryMutation<T>(operation: () => Promise<T>): Promise<T>;
      persistSettings(candidate: PluginSettings): Promise<void>;
    };
    Object.assign(plugin, { settingsChanges: new SettingsChangeService({
      settings,
      persistSettings: (candidate) => storage.enqueueLibraryMutation(() => storage.persistSettings(candidate)),
    }) });
    const tab = new ArxivDailySettingTab(plugin.app, plugin);
    Object.assign(plugin, { settingsTab: tab });
    const viewport = document.createElement("div");
    viewport.style.overflowY = "auto";
    viewport.appendChild(tab.containerEl);
    document.body.appendChild(viewport);
    tab.display();
    tab.containerEl.querySelector<HTMLButtonElement>(".arxiv-daily-settings__topic-header")!.click();
    plugin.openPersonalLibraryDirectionReview();
    const modal = Modal.opened.at(-1)!;
    modal.modalEl.appendChild(modal.contentEl);
    document.body.appendChild(modal.modalEl);
    const pauseSave = () => {
      const gate = deferred();
      const persist = saveData.getMockImplementation()!;
      saveData.mockImplementationOnce(async (data) => { await gate.promise; await persist(data); });
      return gate;
    };
    const input = (field: "direction" | "name" = "direction") => tab.containerEl.querySelector<HTMLInputElement | HTMLTextAreaElement>(
      `.arxiv-daily-settings__topic-${field === "direction" ? "direction" : "name"}-input`,
    )!;
    return { plugin, tab, viewport, modal, settings, saved, saveData, pauseSave, input };
  }

  it.each(["direction", "name"] as const)("keeps a focused %s draft until all queued edits finish, then restores selection and scroll", async (field) => {
    const made = openHostReview();
    const acceptanceSave = made.pauseSave();
    const accepting = made.plugin.acceptPersonalLibraryProposedTopics(["proposed-topic"], ["accepted-direction"]);
    await vi.waitFor(() => expect(made.saveData).toHaveBeenCalledTimes(1));
    made.modal.close();
    const input = made.input(field);
    input.focus();
    input.value = "First concurrent draft";
    const firstEditSave = made.pauseSave();
    input.dispatchEvent(new Event("input"));
    acceptanceSave.resolve();
    await vi.waitFor(() => expect(made.saveData).toHaveBeenCalledTimes(2));
    expect(input.isConnected).toBe(true);
    expect(document.activeElement).toBe(input);
    expect(made.input(field).value).toBe("First concurrent draft");

    const latestSave = made.pauseSave();
    input.value = "Latest concurrent draft";
    input.setSelectionRange(3, 10, "backward");
    input.dispatchEvent(new Event("input"));
    made.viewport.scrollTop = 241;
    made.viewport.scrollLeft = 17;
    firstEditSave.resolve();
    await vi.waitFor(() => expect(made.saveData).toHaveBeenCalledTimes(3));
    expect(input.isConnected).toBe(true);
    expect(input.value).toBe("Latest concurrent draft");
    latestSave.resolve();
    await accepting;

    const visible = made.input(field);
    expect(visible.value).toBe("Latest concurrent draft");
    expect(document.activeElement).toBe(visible);
    expect([visible.selectionStart, visible.selectionEnd, visible.selectionDirection]).toEqual([3, 10, "backward"]);
    expect([made.viewport.scrollTop, made.viewport.scrollLeft]).toEqual([241, 17]);
    expect(made.tab.containerEl.querySelector(".arxiv-daily-settings__topic-header")?.getAttribute("aria-expanded")).toBe("true");
    expect(made.saved.at(-1)!.settings.arxiv.topics[0].directions.at(-1)!.text).toBe("Accepted library direction");
    expect(made.settings.arxiv.topics[0][field === "name" ? "name" : "description"]).toBe("Latest concurrent draft");
    visible.value = "Continued after refresh";
    visible.dispatchEvent(new Event("input"));
    await made.plugin.settingsChanges.changeComputed(() => ({ changes: [] }));
    expect(made.saved.at(-1)!.settings.arxiv.topics[0][field === "name" ? "name" : "description"])
      .toBe("Continued after refresh");
    made.tab.hide();
    made.viewport.remove();
  });

  it("refreshes the restored field after a queued edit fails while preserving the accepted direction", async () => {
    const made = openHostReview();
    const acceptanceSave = made.pauseSave();
    const accepting = made.plugin.acceptPersonalLibraryProposedTopics(["proposed-topic"], ["accepted-direction"]);
    await vi.waitFor(() => expect(made.saveData).toHaveBeenCalledTimes(1));
    made.modal.close();
    const input = made.input();
    input.focus();
    input.value = "Rejected concurrent edit";
    const editSave = made.pauseSave();
    input.dispatchEvent(new Event("input"));
    acceptanceSave.resolve();
    await vi.waitFor(() => expect(made.saveData).toHaveBeenCalledTimes(2));
    expect(input.isConnected).toBe(true);
    editSave.reject(new Error("edit disk full"));
    await accepting;
    const visible = made.input();
    expect(visible.value).toBe("Original handwritten direction");
    expect(document.activeElement).toBe(visible);
    expect(made.settings.arxiv.topics[0].directions.map(({ text }) => text))
      .toEqual(["Original handwritten direction", "Accepted library direction"]);
    expect(made.saved.at(-1)!.settings.arxiv.topics[0].directions.map(({ text }) => text))
      .toEqual(["Original handwritten direction", "Accepted library direction"]);
    expect(made.plugin.logger.error).toHaveBeenCalledWith("settings: edit direction failed", expect.any(Error));
    made.tab.hide();
    made.viewport.remove();
  });
});

describe("topic edits share the settings transaction queue", () => {
  function editor() {
    const made = makeTab();
    made.settings.arxiv.topics.push(normalizeTopic({
      id: "manual-topic", name: "Research agents", tag: "research-agents", detail: false,
      directions: [
        { id: "manual-first", text: "Original direction", origin: "manual" },
        { id: "manual-second", text: "Other direction", origin: "manual" },
      ],
    }));
    const saved: PluginSettings[] = [];
    made.saveSettings.mockImplementation(async (candidate) => {
      if (!candidate) throw new Error("Persistence requires the complete private candidate");
      saved.push(structuredClone(candidate));
    });
    const refresh = vi.spyOn(made.tab, "refreshSettings").mockImplementation(() => {});
    document.body.appendChild(made.tab.containerEl);
    renderSetupGuideRow(made.tab, new Setting(made.tab.containerEl));
    made.tab.renderTopicRow(new Setting(made.tab.containerEl), 0);
    made.tab.containerEl.querySelector<HTMLButtonElement>(".arxiv-daily-settings__topic-header")!.click();
    const input = () => made.tab.containerEl.querySelector<HTMLTextAreaElement>(".arxiv-daily-settings__topic-direction-input")!;
    const pauseSave = () => {
      const gate = deferred();
      const persist = made.saveSettings.getMockImplementation()!;
      made.saveSettings.mockImplementationOnce(async (candidate) => {
        await gate.promise;
        await persist(candidate);
      });
      return gate;
    };
    const acceptDirection = () => made.plugin.settingsChanges.changeComputed((current) => {
      current.arxiv.topics[0].directions.push({ id: "accepted", text: "Accepted library direction", origin: "library" });
      current.arxiv.topics = current.arxiv.topics.map(normalizeTopic);
      return { changes: [{ key: "arxiv.topics", value: current.arxiv.topics }] };
    });
    const settle = () => made.plugin.settingsChanges.changeComputed(() => ({ changes: [] }));
    return { ...made, saved, refresh, input, pauseSave, acceptDirection, settle };
  }

  it.each(["edit", "remove"] as const)("preserves a direction %s made while an accepted direction is being saved", async (action) => {
    const made = editor();
    const gate = made.pauseSave();
    const accepting = made.acceptDirection();
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    if (action === "edit") {
      made.input().value = "My reviewed direction";
      made.input().dispatchEvent(new Event("input"));
    } else {
      made.tab.containerEl.querySelector<HTMLButtonElement>('[aria-label="Remove direction 1"]')!.click();
    }
    gate.resolve();
    await accepting;
    await made.settle();
    const expected = action === "edit"
      ? ["My reviewed direction", "Other direction", "Accepted library direction"]
      : ["Other direction", "Accepted library direction"];
    expect(made.settings.arxiv.topics[0].directions.map(({ text }) => text)).toEqual(expected);
    expect(made.saved.at(-1)!.arxiv.topics[0].directions.map(({ text }) => text)).toEqual(expected);
    expect(made.settings.arxiv.topics[0].description).toBe(expected[0]);
    made.tab.containerEl.remove();
  });

  it("retains every later input draft and its focus while preceding saves commit", async () => {
    const made = editor();
    const acceptingSave = made.pauseSave();
    const accepting = made.acceptDirection();
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    const input = made.input();
    input.focus();
    input.value = "Research";
    input.dispatchEvent(new Event("input"));
    input.value = "Research agents";
    input.dispatchEvent(new Event("input"));
    const firstEditSave = made.pauseSave();
    acceptingSave.resolve();
    await accepting;
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(2));
    input.value = "Research agents evaluation";
    input.dispatchEvent(new Event("input"));
    expect(made.settings.arxiv.topics[0].directions[0].text).toBe("Original direction");
    expect(input.value).toBe("Research agents evaluation");
    firstEditSave.resolve();
    await made.settle();
    expect(input.value).toBe("Research agents evaluation");
    expect(document.activeElement).toBe(input);
    expect(made.refresh).not.toHaveBeenCalled();
    expect(made.settings.arxiv.topics[0].directions.map(({ text }) => text))
      .toEqual(["Research agents evaluation", "Other direction", "Accepted library direction"]);
    expect(made.saved.slice(1).map((settings) => settings.arxiv.topics[0].directions[0].text))
      .toEqual(["Research", "Research agents", "Research agents evaluation"]);
    made.tab.containerEl.remove();
  });

  it("adds and edits a new direction during acceptance without replacing the latest direction list", async () => {
    const made = editor();
    const gate = made.pauseSave();
    const accepting = made.acceptDirection();
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    made.tab.containerEl.querySelector<HTMLButtonElement>(".arxiv-daily-settings__topic-direction-add")!.click();
    const added = Array.from(made.tab.containerEl.querySelectorAll<HTMLTextAreaElement>(
      ".arxiv-daily-settings__topic-direction-input",
    )).at(-1)!;
    expect(document.activeElement).toBe(added);
    added.value = "New handwritten direction";
    added.dispatchEvent(new Event("input"));
    gate.resolve();
    await accepting;
    await made.settle();
    expect(made.settings.arxiv.topics[0].directions.map(({ text }) => text)).toEqual([
      "Original direction", "Other direction", "Accepted library direction", "New handwritten direction",
    ]);
    expect(made.saved.at(-1)!.arxiv.topics[0].directions.map(({ text }) => text)).toEqual([
      "Original direction", "Other direction", "Accepted library direction", "New handwritten direction",
    ]);
    made.tab.containerEl.remove();
  });

  it("derives a renamed topic's unique tag from topics committed ahead of its edit", async () => {
    const made = editor();
    const gate = made.pauseSave();
    const preceding = made.plugin.settingsChanges.changeComputed((current) => ({ changes: [{
      key: "arxiv.topics",
      value: [...current.arxiv.topics, normalizeTopic({ id: "new-topic", name: "Evaluation", tag: "evaluation", directions: [] })],
    }] }));
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    const name = made.tab.containerEl.querySelector<HTMLInputElement>(".arxiv-daily-settings__topic-name-input")!;
    name.value = "Evaluation";
    name.dispatchEvent(new Event("input"));
    expect(made.settings.arxiv.topics[0].name).toBe("Research agents");
    gate.resolve();
    await preceding;
    await made.settle();
    expect(made.settings.arxiv.topics.map(({ name, tag }) => ({ name, tag })))
      .toEqual([{ name: "Evaluation", tag: "evaluation-2" }, { name: "Evaluation", tag: "evaluation" }]);
    expect(made.saved.at(-1)!.arxiv.topics[0].tag).toBe("evaluation-2");
    made.tab.containerEl.remove();
  });

  it("commits a detail toggle together with directions accepted ahead of it", async () => {
    const made = editor();
    const gate = made.pauseSave();
    const accepting = made.acceptDirection();
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    const detail = made.tab.containerEl.querySelector<HTMLInputElement>(".arxiv-daily-settings__topic-detail-checkbox")!;
    detail.click();
    expect(made.settings.arxiv.topics[0].detail).toBe(false);
    gate.resolve();
    await accepting;
    await made.settle();
    expect(made.settings.arxiv.topics[0].detail).toBe(true);
    expect(made.saved.at(-1)!.arxiv.topics[0].directions.at(-1)!.text).toBe("Accepted library direction");
    expect(made.saved.at(-1)!.arxiv.topics[0].detail).toBe(true);
    made.tab.containerEl.remove();
  });

  it("deletes the chosen topic by identity after an earlier transaction changes its position", async () => {
    const made = editor();
    vi.spyOn(made.tab, "confirmReplace").mockResolvedValue(true);
    const gate = made.pauseSave();
    const preceding = made.plugin.settingsChanges.changeComputed((current) => ({ changes: [{
      key: "arxiv.topics",
      value: [normalizeTopic({ id: "new-topic", name: "New", tag: "new", directions: [] }), ...current.arxiv.topics],
    }] }));
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    const remove = Array.from(made.tab.containerEl.querySelectorAll<HTMLButtonElement>("button"))
      .find(({ textContent }) => textContent === "Delete")!;
    remove.click();
    await Promise.resolve();
    gate.resolve();
    await preceding;
    await made.settle();
    expect(made.settings.arxiv.topics.map(({ id }) => id)).toEqual(["new-topic"]);
    expect(made.saved.at(-1)!.arxiv.topics.map(({ id }) => id)).toEqual(["new-topic"]);
    made.tab.containerEl.remove();
  });

  it("queues adding a blank topic without losing accepted directions", async () => {
    const made = editor();
    const gate = made.pauseSave();
    const accepting = made.acceptDirection();
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    const adding = made.tab.addTopic();
    expect(made.settings.arxiv.topics).toHaveLength(1);
    gate.resolve();
    await Promise.all([accepting, adding]);
    expect(made.saved.at(-1)!.arxiv.topics).toHaveLength(2);
    expect(made.saved.at(-1)!.arxiv.topics[0].directions.at(-1)!.text).toBe("Accepted library direction");
    made.tab.containerEl.remove();
  });

  it.each(["text", "name", "detail", "remove"] as const)("restores a rejected %s edit without changing live settings", async (action) => {
    const made = editor();
    const original = structuredClone(made.settings);
    const previousNoticeCount = Notice.calls.length;
    made.saveSettings.mockRejectedValueOnce(new Error("disk full"));
    // Calling the registered DOM handler also lets the old uncaught rejection
    // settle, so Red is about the dirty state rather than an unhandled promise.
    if (action === "text" || action === "name") {
      const input = action === "text" ? made.input()
        : made.tab.containerEl.querySelector<HTMLInputElement>(".arxiv-daily-settings__topic-name-input")!;
      input.focus();
      input.value = "Rejected edit";
      await Promise.resolve(input.oninput!.call(input, new Event("input"))).catch(() => undefined);
      expect(input.value).toBe(action === "text" ? "Original direction" : "Research agents");
      expect(document.activeElement).toBe(input);
    } else if (action === "detail") {
      const detail = made.tab.containerEl.querySelector<HTMLInputElement>(".arxiv-daily-settings__topic-detail-checkbox")!;
      detail.checked = true;
      await Promise.resolve(detail.onchange!.call(detail, new Event("change"))).catch(() => undefined);
      expect(detail.checked).toBe(false);
    } else {
      const remove = made.tab.containerEl.querySelector<HTMLButtonElement>('[aria-label="Remove direction 1"]')!;
      await Promise.resolve(remove.onclick!.call(remove, new MouseEvent("click"))).catch(() => undefined);
      expect(made.input().value).toBe("Original direction");
    }
    expect(made.settings).toEqual(original);
    expect(made.saved).toEqual([]);
    expect(Notice.calls.slice(previousNoticeCount).map(({ message }) => message).join(" ")).toContain("failed");
    made.tab.containerEl.remove();
  });

  it("does not restore an older failed draft over a newer queued direction edit", async () => {
    const made = editor();
    const oldSave = made.pauseSave();
    const input = made.input();
    input.value = "Older edit";
    const older = Promise.resolve(input.oninput!.call(input, new Event("input"))).catch(() => undefined);
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(1));
    const newerSave = made.pauseSave();
    input.value = "Newer edit";
    input.dispatchEvent(new Event("input"));
    oldSave.reject(new Error("old edit failed"));
    await older;
    await vi.waitFor(() => expect(made.saveSettings).toHaveBeenCalledTimes(2));
    expect(input.value).toBe("Newer edit");
    expect(made.settings.arxiv.topics[0].directions[0].text).toBe("Original direction");
    newerSave.resolve();
    await made.settle();
    expect(made.saved.at(-1)!.arxiv.topics[0].directions[0].text).toBe("Newer edit");
    expect(made.settings.arxiv.topics[0].directions[0].text).toBe("Newer edit");
    made.tab.containerEl.remove();
  });

  it("keeps the original topics and categories when a template replacement cannot be saved", async () => {
    const made = editor();
    const original = structuredClone(made.settings);
    vi.spyOn(made.tab, "confirmReplace").mockResolvedValue(true);
    made.saveSettings.mockRejectedValueOnce(new Error("disk full"));
    await made.tab.applyTopicTemplate("astro-ml").catch(() => undefined);
    expect(made.settings).toEqual(original);
    expect(made.saved).toEqual([]);
    made.tab.containerEl.remove();
  });
});

describe("topic directions editor", () => {
  function renderTopicWithDirections(texts: string[]) {
    const { tab, settings, saveSettings } = makeTab();
    settings.arxiv.topics.push(
      normalizeTopic({
        id: "t1",
        name: "Photo-z",
        tag: "photo-z",
        detail: false,
        directions: texts.map((text, i) => ({ id: `d${i}`, text, origin: "manual" })),
      }),
    );
    document.body.appendChild(tab.containerEl);
    // refreshSettings() takes the declarative path, whose update() every other
    // test here stubs out, so it renders no cards at all. renderTopicRow is the
    // real entry point for a single topic card.
    tab.renderTopicRow(new Setting(tab.containerEl), 0);
    return { tab, settings, saveSettings };
  }

  function directionInputs(tab: ArxivDailySettingTab) {
    return Array.from(
      tab.containerEl.querySelectorAll<HTMLInputElement>(
        ".arxiv-daily-settings__topic-direction-input",
      ),
    );
  }

  it("renders one single-line input per direction", () => {
    const { tab } = renderTopicWithDirections([
      "Photometric redshift methods.",
      "Catalog cross-matching.",
    ]);

    const inputs = directionInputs(tab);
    expect(inputs).toHaveLength(2);
    expect(inputs.map((i) => i.value)).toEqual([
      "Photometric redshift methods.",
      "Catalog cross-matching.",
    ]);
    // ADR 0012 §2's "one line" is about the stored value, not the rendering:
    // a long direction wraps so it can be read in full, but it is still one
    // string with no newline in it.
    expect(inputs.every((i) => i.tagName === "TEXTAREA")).toBe(true);
    tab.containerEl.remove();
  });

  it("collapses an unfocused direction and expands it while editing", () => {
    const { tab } = renderTopicWithDirections(["A long direction that wraps."]);

    const input = directionInputs(tab)[0];
    // The cap lives on the auto-growing wrapper, not on the textarea.
    const field = input.parentElement!;
    const collapsed = () => field.classList.contains("is-collapsed");
    expect(collapsed()).toBe(true);

    input.dispatchEvent(new Event("focus"));
    expect(collapsed()).toBe(false);

    input.dispatchEvent(new Event("blur"));
    expect(collapsed()).toBe(true);
    tab.containerEl.remove();
  });

  it("counts the hidden lines of a truncated direction", () => {
    const { tab } = renderTopicWithDirections(["A long direction that wraps."]);
    const input = directionInputs(tab)[0];
    const field = input.parentElement!;
    const badge = field.querySelector<HTMLElement>(
      ".arxiv-daily-settings__topic-direction-more",
    )!;
    expect(badge.classList.contains("is-visible")).toBe(false);

    // happy-dom lays nothing out, so the collapsed geometry is stated here:
    // one visible 20px line against three lines of content.
    input.style.lineHeight = "20px";
    Object.defineProperty(field, "clientHeight", { value: 20, configurable: true });
    Object.defineProperty(field, "scrollHeight", { value: 60, configurable: true });
    input.dispatchEvent(new Event("input"));

    expect(badge.textContent).toBe("+2");
    expect(badge.title).toBe("2 more lines");
    expect(badge.classList.contains("is-visible")).toBe(true);

    // One hidden line reads as a singular.
    Object.defineProperty(field, "scrollHeight", { value: 40, configurable: true });
    input.dispatchEvent(new Event("input"));
    expect(badge.textContent).toBe("+1");
    expect(badge.title).toBe("1 more line");

    // Focus lifts the cap, so nothing is hidden while editing.
    input.dispatchEvent(new Event("focus"));
    expect(badge.classList.contains("is-visible")).toBe(false);
    tab.containerEl.remove();
  });

  it("never stores a newline in a direction", async () => {
    const { tab, settings } = renderTopicWithDirections(["One."]);

    const input = directionInputs(tab)[0];
    // A paste can carry newlines; the filter prompt joins topics with "\n",
    // so one inside the text would break that line structure.
    input.value = "Pasted first line\nand a second line";
    input.dispatchEvent(new Event("input"));
    await vi.waitFor(() => expect(settings.arxiv.topics[0].directions[0].text)
      .toBe("Pasted first line and a second line"));

    const stored = settings.arxiv.topics[0].directions[0].text;
    expect(stored).not.toContain("\n");
    expect(stored).toBe("Pasted first line and a second line");
    tab.containerEl.remove();
  });

  it("makes Enter confirm the direction and leave the field", async () => {
    const { tab, settings } = renderTopicWithDirections(["First."]);

    const input = directionInputs(tab)[0];
    const field = input.parentElement!;
    input.focus();
    input.dispatchEvent(new Event("focus"));
    expect(field.classList.contains("is-collapsed")).toBe(false);

    const event = new KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
      cancelable: true,
    });
    input.dispatchEvent(event);
    await new Promise<void>((resolve) => queueMicrotask(resolve));

    // No newline is inserted: a direction is one line, and the filter prompt
    // joins topics with "\n".
    expect(event.defaultPrevented).toBe(true);
    // Enter confirms rather than adding: that stays the Add button's job.
    expect(settings.arxiv.topics[0].directions).toHaveLength(1);
    expect(settings.arxiv.topics[0].directions[0].text).toBe("First.");
    expect(document.activeElement).not.toBe(input);
    expect(field.classList.contains("is-collapsed")).toBe(true);
    tab.containerEl.remove();
  });

  it("no longer offers a free-text description textarea", () => {
    const { tab } = renderTopicWithDirections(["Photometric redshift methods."]);

    expect(
      tab.containerEl.querySelector(".arxiv-daily-settings__topic-description"),
    ).toBeNull();
    tab.containerEl.remove();
  });

  it("offers add and remove controls for directions", () => {
    const { tab } = renderTopicWithDirections(["One.", "Two."]);

    expect(
      tab.containerEl.querySelector(".arxiv-daily-settings__topic-direction-add"),
    ).not.toBeNull();
    expect(
      tab.containerEl.querySelectorAll(
        ".arxiv-daily-settings__topic-direction-remove",
      ),
    ).toHaveLength(2);
    tab.containerEl.remove();
  });

  it("keeps the description shadow in step while editing the first direction", async () => {
    const { tab, settings } = renderTopicWithDirections(["Old text.", "Second."]);

    const first = directionInputs(tab)[0];
    first.value = "New text.";
    first.dispatchEvent(new Event("input"));
    await vi.waitFor(() => expect(settings.arxiv.topics[0].directions[0].text).toBe("New text."));

    const topic = settings.arxiv.topics[0];
    expect(topic.directions[0].text).toBe("New text.");
    expect(topic.description).toBe("New text.");
    tab.containerEl.remove();
  });

  it("leaves the shadow alone when a later direction is edited", async () => {
    const { tab, settings } = renderTopicWithDirections(["First.", "Second."]);

    const second = directionInputs(tab)[1];
    second.value = "Second edited.";
    second.dispatchEvent(new Event("input"));
    await vi.waitFor(() => expect(settings.arxiv.topics[0].directions[1].text).toBe("Second edited."));

    const topic = settings.arxiv.topics[0];
    expect(topic.directions[1].text).toBe("Second edited.");
    expect(topic.description).toBe("First.");
    tab.containerEl.remove();
  });
});

describe("topic editing when the setup guide is not on screen", () => {
  /**
   * A finished setup drops the "Getting started" row from the declarative
   * definitions, so no guide row is ever registered. refreshSetupGuide() then
   * used to fall back to a whole-tab re-render, which replaces the input the
   * user is typing into. Onboarding hid this: while the guide is on screen the
   * cheap path is taken, and that is the only case the 0.4.4 fix covered.
   */
  function completedSetup() {
    const { tab, plugin, settings, saveSettings } = makeTab();
    settings.onboarding.guideCompleted = true;
    (plugin as unknown as { stateStore: { snapshot: () => unknown } }).stateStore = {
      snapshot: () => ({ "2026-09-01": { status: "completed" } }),
    };
    settings.llm.apiKey = "sk-test";
    settings.llm.baseUrl = "https://api.example.com/v1";
    settings.llm.model = "test-model";
    settings.arxiv.topics.push(
      normalizeTopic({
        id: "t1",
        name: "Photo-z",
        tag: "photo-z",
        detail: false,
        directions: [{ id: "d1", text: "Photometric redshift methods.", origin: "manual" }],
      }),
    );
    return { tab, settings, saveSettings };
  }

  it("does not re-render the whole tab while any topic field is edited", async () => {
    const { tab, saveSettings } = completedSetup();
    // Deliberately never render a setup guide row: a finished setup has none.
    expect(tab.shouldShowSetupGuide()).toBe(false);
    const refresh = vi.spyOn(tab, "refreshSettings");
    document.body.appendChild(tab.containerEl);
    const topicSetting = new Setting(tab.containerEl);
    tab.renderTopicRow(topicSetting, 0);

    // Not direction-specific: name and directions lose focus the same way.
    const fields = [
      ".arxiv-daily-settings__topic-name-input",
      ".arxiv-daily-settings__topic-direction-input",
    ].map((selector) =>
      topicSetting.settingEl.querySelector<HTMLInputElement>(selector)!,
    );

    for (const [index, input] of fields.entries()) {
      input.focus();
      input.value = `draft-${index}`;
      input.dispatchEvent(new Event("input"));
      await vi.waitFor(() => {
        expect(saveSettings).toHaveBeenCalledTimes(index + 1);
      });
      expect(document.activeElement).toBe(input);
    }

    expect(refresh).not.toHaveBeenCalled();
    tab.containerEl.remove();
  });

  it("still re-renders when the guide has to appear again", async () => {
    const { tab, settings, saveSettings } = completedSetup();
    const refresh = vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    document.body.appendChild(tab.containerEl);
    const topicSetting = new Setting(tab.containerEl);
    tab.renderTopicRow(topicSetting, 0);

    // Emptying the only direction makes the setup incomplete, so the guide
    // must come back — that transition is worth a full re-render.
    const input = topicSetting.settingEl.querySelector<HTMLInputElement>(
      ".arxiv-daily-settings__topic-direction-input",
    )!;
    input.value = "";
    input.dispatchEvent(new Event("input"));
    await vi.waitFor(() => expect(saveSettings).toHaveBeenCalledTimes(1));

    expect(settings.arxiv.topics[0].description).toBe("");
    expect(refresh).toHaveBeenCalled();
    tab.containerEl.remove();
  });
});

describe("topics created by the plugin", () => {
  it("applies a quick-start template as topics with authored directions", async () => {
    const { tab, settings } = makeTab();
    vi.spyOn(tab, "confirmReplace").mockResolvedValue(true);
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});

    await tab.applyTopicTemplate("astro-ml");

    expect(settings.arxiv.topics).toHaveLength(3);
    for (const topic of settings.arxiv.topics) {
      expect(topic.directions).toHaveLength(1);
      expect(topic.directions[0].origin).toBe("manual");
      // The shadow an older build reads stays in step with the list.
      expect(topic.description).toBe(topic.directions[0].text);
    }
    expect(settings.arxiv.topics[0].name).toBe("Photo-z");
  });

  it("adds a blank topic with no directions and an empty shadow", async () => {
    const { tab, settings } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});

    await tab.addTopic();

    const topic = settings.arxiv.topics.at(-1)!;
    expect(topic.directions).toEqual([]);
    expect(topic.description).toBe("");
    expect(topic.tag).toBe("topic-1");
  });

  it("does not hand out a tag that already exists after topics were deleted and re-added", async () => {
    const { tab, settings } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});
    vi.spyOn(tab, "confirmReplace").mockResolvedValue(true);

    await tab.addTopic();
    await tab.addTopic();
    expect(settings.arxiv.topics.map((t) => t.tag)).toEqual(["topic-1", "topic-2"]);

    // Deleting the first frees "topic-1"; the plain count-based candidate for
    // a third topic would otherwise collide with the survivor's tag.
    await tab.deleteTopic(0);
    await tab.addTopic();

    const tags = settings.arxiv.topics.map((t) => t.tag);
    expect(new Set(tags).size).toBe(tags.length);
  });
});

/**
 * Settings no longer shows a topic's machine tag: the collapsed header used
 * to pair name and tag, and that outran the row. The tag itself still exists
 * (daily reports and the filter prompt key off it), so renaming has to keep
 * deriving it — just without a field on screen to fix a collision by hand.
 */
describe("topic header hides the machine tag", () => {
  function renderTopic(topic: Parameters<typeof normalizeTopic>[0]) {
    const { tab, settings, saveSettings } = makeTab();
    settings.arxiv.topics.push(normalizeTopic(topic));
    document.body.appendChild(tab.containerEl);
    tab.renderTopicRow(new Setting(tab.containerEl), 0);
    return { tab, settings, saveSettings };
  }

  it("shows only the name in the collapsed header, with no tag chip", () => {
    const { tab } = renderTopic({
      id: "t1",
      name: "Photo-z",
      tag: "photo-z",
      detail: false,
      directions: [],
    });

    const header = tab.containerEl.querySelector(
      ".arxiv-daily-settings__topic-header",
    )!;
    const title = header.querySelector(".arxiv-daily-settings__topic-title")!;
    expect(title.textContent).toBe("Photo-z");
    expect(header.textContent).not.toContain("#");
    tab.containerEl.remove();
  });

  it("shows the name and the detail star, still with no tag chip", () => {
    const { tab } = renderTopic({
      id: "t1",
      name: "Photo-z",
      tag: "photo-z",
      detail: true,
      directions: [],
    });

    const header = tab.containerEl.querySelector(
      ".arxiv-daily-settings__topic-header",
    )!;
    expect(header.querySelector(".arxiv-daily-settings__topic-star")).not.toBeNull();
    expect(header.textContent).not.toContain("#");
    tab.containerEl.remove();
  });

  it("renders no tag input in the expanded form", () => {
    const { tab } = renderTopic({
      id: "t1",
      name: "Photo-z",
      tag: "photo-z",
      detail: false,
      directions: [],
    });

    expect(
      tab.containerEl.querySelector(".arxiv-daily-settings__topic-tag-input"),
    ).toBeNull();
    tab.containerEl.remove();
  });

  it("re-derives a tag that was still the machine form of the old name", async () => {
    const { tab, settings } = renderTopic({
      id: "t1",
      name: "Photo-z",
      tag: "photo-z",
      detail: false,
      directions: [],
    });

    const nameInput = tab.containerEl.querySelector<HTMLInputElement>(
      ".arxiv-daily-settings__topic-name-input",
    )!;
    nameInput.value = "Weak lensing";
    nameInput.dispatchEvent(new Event("input"));

    await vi.waitFor(() => expect(settings.arxiv.topics[0].tag).toBe("weak-lensing"));
    tab.containerEl.remove();
  });

  it("leaves a hand-set tag alone when the name is renamed", async () => {
    const { tab, settings } = renderTopic({
      id: "t1",
      name: "Photo-z",
      tag: "custom-tag",
      detail: false,
      directions: [],
    });

    const nameInput = tab.containerEl.querySelector<HTMLInputElement>(
      ".arxiv-daily-settings__topic-name-input",
    )!;
    nameInput.value = "Weak lensing";
    nameInput.dispatchEvent(new Event("input"));

    await vi.waitFor(() => expect(settings.arxiv.topics[0].name).toBe("Weak lensing"));
    expect(settings.arxiv.topics[0].tag).toBe("custom-tag");
    tab.containerEl.remove();
  });

  it("makes a renamed topic's derived tag unique against a collision, not a duplicate", async () => {
    const { tab, settings } = makeTab();
    settings.arxiv.topics.push(
      normalizeTopic({
        id: "t1",
        name: "Photo-z",
        tag: "photo-z",
        detail: false,
        directions: [],
      }),
      normalizeTopic({
        id: "t2",
        name: "Weak lensing",
        tag: "weak-lensing",
        detail: false,
        directions: [],
      }),
    );
    document.body.appendChild(tab.containerEl);
    tab.renderTopicRow(new Setting(tab.containerEl), 0);

    const nameInput = tab.containerEl.querySelector<HTMLInputElement>(
      ".arxiv-daily-settings__topic-name-input",
    )!;
    nameInput.value = "Weak lensing";
    nameInput.dispatchEvent(new Event("input"));

    await vi.waitFor(() => expect(settings.arxiv.topics[0].tag).toBe("weak-lensing-2"));
    expect(settings.arxiv.topics[1].tag).toBe("weak-lensing");
    tab.containerEl.remove();
  });
});

describe("topic editor behavior contracts", () => {
  it("asks about the named topic and preserves it when deletion is declined", async () => {
    const { tab, settings, saveSettings } = makeTab();
    const topic = normalizeTopic({ id: "confirmed-topic", name: "Photo-z", tag: "photo-z", directions: [] });
    settings.arxiv.topics = [topic];
    const confirmation = vi.spyOn(tab, "confirmReplace").mockResolvedValue(false);
    await tab.deleteTopic(0);
    expect(confirmation).toHaveBeenCalledWith(expect.stringContaining('"Photo-z"'), "Delete");
    expect(settings.arxiv.topics).toEqual([topic]);
    expect(saveSettings).not.toHaveBeenCalled();
    confirmation.mockResolvedValue(true);
    await tab.deleteTopic(0);
    expect(settings.arxiv.topics).toEqual([]);
    expect(saveSettings).toHaveBeenCalledOnce();
  });

  it("does not rewrite or persist stored categories when rendering settings", () => {
    const { tab, settings, saveSettings } = makeTab();
    settings.arxiv.categories = ["astro-ph.GA", "astro-ph.GA"];
    const before = structuredClone(settings.arxiv);
    tab.getSettingDefinitions();
    tab.display();
    expect(settings.arxiv).toEqual(before);
    expect(saveSettings).not.toHaveBeenCalled();
  });
});
