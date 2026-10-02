import { beforeAll, describe, expect, it, vi } from "vitest";
import {
  ButtonComponent,
  DropdownComponent,
  Setting,
  TextComponent,
  ToggleComponent,
  type App,
} from "obsidian";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type ArxivDailyPlugin from "../main";
import { LibraryIndexStatusStore } from "../src/library/index-status";
import {
  ArxivDailySettingTab,
  isValidLocalTime,
  llmHttpWarning,
  modelFetchNoticeMessage,
  runWindowTimeOptions,
  validateOutputDirectoryDraft,
} from "../src/settings/tab";
import { confirmEmbeddingMode } from "../src/library/modal";
import { SettingsChangeService } from "../src/settings/change-service";

const settingsTabSource = readFileSync(
  resolve(process.cwd(), "src/settings/tab.ts"),
  "utf-8",
);

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
    detach?: () => void;
    appendText?: (text: string) => void;
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
  proto.detach ??= function () { this.remove(); };
  proto.appendText ??= function (text) { this.append(text); };
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
  proto.createDiv ??= function (options = {}) { return this.createEl("div", options); };
  proto.createSpan ??= function (options = {}) { return this.createEl("span", options); };
});

function makeLegacyApiKeyTab(
  persistSettings: (candidate: typeof DEFAULT_SETTINGS) => Promise<void>,
) {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.llm.apiKey = "old-secret";
  const refreshSensitiveValues = vi.fn();
  const installOutputStores = vi.fn();
  const settingsChanges = new SettingsChangeService({
    settings,
    persistSettings,
    refreshSensitiveValues,
    prepareOutputStores: vi.fn(async () => ({
      stateStore: { name: "candidate-state" },
      runHistoryStore: { name: "candidate-history" },
    } as never)),
    installOutputStores,
  });
  const plugin = {
    settings,
    settingsChanges,
    saveSettings: () => settingsChanges.persistCurrent(),
    setScheduleEnabled: (enabled: boolean) => settingsChanges
      .changeValue("schedule.enabled", enabled)
      .then(() => true),
    logger: { error: vi.fn() },
    stateStore: { snapshot: () => ({}) },
    manifest: { version: "0.0.0-test" },
    getLibraryConnectionStatus: vi.fn().mockReturnValue({ kind: "disconnected" }),
    libraryIndexStatus: new LibraryIndexStatusStore(),
    automaticEmailSupported: () => true,
    openPersonalLibraryDirectionReview: vi.fn(),
    getPersonalLibraryInterestProfile: vi.fn().mockReturnValue(null),
  } as unknown as ArxivDailyPlugin;
  const tab = new ArxivDailySettingTab({} as App, plugin);
  vi.spyOn(tab, "refreshSetupGuide").mockImplementation(() => undefined);
  return { tab, settings, refreshSensitiveValues, installOutputStores };
}

it("offers only one library suggestion entry in legacy settings", async () => {
  const { tab } = makeLegacyApiKeyTab(async () => undefined);
  tab.plugin.getLibraryConnectionStatus = () => ({ kind: "authorized", rootLabel: "papers", grantedAt: "2026-09-08T00:00:00.000Z" });
  tab.plugin.libraryIndexStatus.setLastRun({ updatedAt: "2026-09-08T00:00:00.000Z", papers: 20 });
  tab.plugin.openPersonalLibraryDirectionReview = vi.fn();
  tab.display();
  const entry = Array.from(tab.containerEl.querySelectorAll("button")).find(({ textContent }) => textContent === "Review suggestions");
  expect(entry).toBeDefined();
  expect(Array.from(tab.containerEl.querySelectorAll("button")).filter(({ textContent }) => /Review suggestions|Use my library|Review directions/.test(textContent ?? ""))).toHaveLength(1);
  expect(tab.containerEl.textContent).not.toContain("Topics from your library");
  entry!.click();
  await vi.waitFor(() => expect(tab.plugin.openPersonalLibraryDirectionReview).toHaveBeenCalledOnce());
});

function renderLegacyApiKey(tab: ArxivDailySettingTab) {
  const container = document.createElement("div");
  const render = Reflect.get(tab, "renderApiKeySetting") as (
    containerEl: HTMLElement,
  ) => void;
  render.call(tab, container);
  const input = container.querySelector("input");
  const buttons = [...container.querySelectorAll("button")];
  if (!input || buttons.length !== 1) throw new Error("API key row did not render");
  return {
    input,
    reveal: buttons[0] as HTMLButtonElement,
  };
}

function renderLegacySettings(tab: ArxivDailySettingTab): Map<string, Setting> {
  Setting.reset();
  ToggleComponent.reset();
  tab.display();
  return new Map(
    Setting.instances.map((setting) => [setting.nameEl.textContent ?? "", setting]),
  );
}

function componentOf<T>(setting: Setting | undefined, ctor: new (...args: never[]) => T): T {
  const component = setting?.components.find((item) => item instanceof ctor);
  if (!component) throw new Error(`Missing ${ctor.name} component`);
  return component as T;
}

describe("legacy daily paper limit", () => {
  it("shows 20 and persists a whole-number edit as a number", async () => {
    const persistSettings = vi.fn().mockResolvedValue(undefined);
    const { tab, settings, installOutputStores } = makeLegacyApiKeyTab(persistSettings);
    const row = renderLegacySettings(tab).get("Daily paper limit");
    const input = row?.controlEl.querySelector<HTMLInputElement>("input");
    expect(input).toBeDefined();
    expect(input!.value).toBe("20");
    expect(input!.min).toBe("1");
    expect(input!.step).toBe("1");
    input!.value = "35";
    input!.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(settings.output.maxDailyPapers).toBe(35));
    expect(persistSettings).toHaveBeenCalledWith(expect.objectContaining({
      output: expect.objectContaining({ maxDailyPapers: 35 }),
    }));
    expect(installOutputStores).not.toHaveBeenCalled();
  });

  it.each(["0", "2.5", ""])("rejects the invalid draft %j without persistence", async (draft) => {
    const persistSettings = vi.fn().mockResolvedValue(undefined);
    const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
    const input = renderLegacySettings(tab).get("Daily paper limit")?.controlEl.querySelector<HTMLInputElement>("input");
    expect(input).toBeDefined();
    input!.value = draft;
    input!.dispatchEvent(new Event("change"));
    await Promise.resolve();
    expect(input!.validationMessage).toContain("positive whole number");
    expect(persistSettings).not.toHaveBeenCalled();
    expect(settings.output.maxDailyPapers).toBe(20);
  });

  it("restores the current limit when persistence fails", async () => {
    const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
    const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
    const input = renderLegacySettings(tab).get("Daily paper limit")?.controlEl.querySelector<HTMLInputElement>("input");
    expect(input).toBeDefined();
    input!.value = "35";
    input!.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(input!.value).toBe("20"));
    expect(persistSettings).toHaveBeenCalledOnce();
    expect(settings.output.maxDailyPapers).toBe(20);
  });
});

describe("modelFetchNoticeMessage", () => {
  it("reports a successful model fetch in English", () => {
    expect(modelFetchNoticeMessage({ kind: "success", count: 3 })).toBe(
      "API connection successful. Found 3 models.",
    );
  });

  it("reports an empty model list in English", () => {
    expect(modelFetchNoticeMessage({ kind: "empty" })).toBe(
      "API connection successful, but no available models were found.",
    );
  });

  it("reports a failed model fetch in English", () => {
    expect(
      modelFetchNoticeMessage({ kind: "error", message: "Unauthorized" }),
    ).toBe("API connection failed: Unauthorized");
  });
});

describe("llmHttpWarning", () => {
  it("warns without blocking for non-loopback HTTP endpoints", () => {
    expect(llmHttpWarning("http://59.64.32.247:5001/v1")).toEqual({
      kind: "plaintext",
      message:
        "This address uses plain HTTP. Your API key would be sent without encryption—prefer HTTPS.",
    });
  });

  it("uses a softer warning for local HTTP endpoints", () => {
    expect(llmHttpWarning("http://localhost:5001/v1")).toEqual({
      kind: "local",
      message:
        "This address uses plain HTTP on this computer. Only continue if you meant to use a local AI service.",
    });
    expect(llmHttpWarning("http://127.12.0.1:5001/v1")?.kind).toBe("local");
    expect(llmHttpWarning("http://[::1]:5001/v1")?.kind).toBe("local");
  });

  it("does not warn for HTTPS or invalid partial input", () => {
    expect(llmHttpWarning("https://api.deepseek.com/v1")).toBeNull();
    expect(llmHttpWarning("59.64.32.247:5001/v1")).toBeNull();
  });
});

describe("legacy section order", () => {
  it("renders Personal library after Output & schedule and before Email delivery", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    renderLegacySettings(tab);
    const headings = Setting.instances
      .filter((setting) => setting.settingEl.hasAttribute("data-arxiv-daily-section"))
      .map((setting) => setting.nameEl.textContent ?? "");
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
});

describe("legacy personal library guide box", () => {
  it("shows the intro box while no library is connected", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    vi.mocked(tab.plugin.getLibraryConnectionStatus).mockReturnValue({ kind: "disconnected" });
    renderLegacySettings(tab);
    const boxes = tab.containerEl.querySelectorAll(".arxiv-daily-settings__library-guide");
    expect(boxes).toHaveLength(1);
    expect(boxes[0]?.textContent).toContain("Choose a folder of PDFs");
  });

  it("keeps the intro box concise but complete, and honest about the one-time model download", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    const content = tab.libraryGuideContent();
    expect(content.lines.length).toBeLessThanOrEqual(4);
    const text = content.lines.join(" ");
    expect(text).toMatch(/optional/i);
    expect(text).toContain("daily reports work the same without a library");
    expect(text).toContain("Choose a folder of PDFs");
    expect(text).toContain("automatically");
    expect(text).toContain("130 MB");
    expect(text).toMatch(/model downloads once/);
    expect(text).toMatch(/runs locally/);
    expect(text).toContain("Review suggestions");
    expect(text).toContain("steer daily reports");
    expect(text).toContain("Remote embedding and model processing always ask first");
    expect(text).not.toContain("bundled");
    expect(content.lines.filter((line) => /^\d\./.test(line))).toHaveLength(2);
    const directionsStep = content.lines[2] ?? "";
    expect(directionsStep).toContain("Review suggestions (button below)");
    expect(directionsStep).not.toContain("command palette");

    const indexStep = content.lines[3] ?? "";
    expect(indexStep).toContain("Retry preparation");
    expect(text).toContain("arXiv");
    expect(indexStep).not.toContain("button above");
  });

  it("keeps showing the intro box once a folder is chosen (always visible, like the email guide)", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    vi.mocked(tab.plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    renderLegacySettings(tab);
    expect(tab.containerEl.querySelectorAll(".arxiv-daily-settings__library-guide")).toHaveLength(1);
  });

  it("keeps showing the intro box once authorized (always visible, like the email guide)", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    vi.mocked(tab.plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: new Date().toISOString(),
    });
    renderLegacySettings(tab);
    expect(tab.containerEl.querySelectorAll(".arxiv-daily-settings__library-guide")).toHaveLength(1);
  });
});

describe("legacy personal library directions row", () => {
  it("is hidden while no library folder is chosen", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    vi.mocked(tab.plugin.getLibraryConnectionStatus).mockReturnValue({ kind: "disconnected" });
    const rows = renderLegacySettings(tab);
    expect(rows.has("Topics from library")).toBe(false);
  });

  it("shows below Library once a folder is chosen", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    vi.mocked(tab.plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    renderLegacySettings(tab);
    const names = Setting.instances.map((setting) => setting.nameEl.textContent ?? "");
    const libraryIndex = names.indexOf("Library");
    const directionsIndex = names.indexOf("Topics from library");
    expect(libraryIndex).toBeGreaterThanOrEqual(0);
    expect(directionsIndex).toBe(libraryIndex + 1);
    const row = Setting.instances.find((setting) => setting.nameEl.textContent === "Topics from library")!;
    expect(row.controlEl.querySelector("button")?.disabled).toBe(true);
    expect(row.descEl.textContent).toMatch(/prepar/i);
  });

  it("opens the direction review modal when its button is clicked", async () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    vi.mocked(tab.plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: new Date().toISOString(),
    });
    tab.plugin.libraryIndexStatus.setLastRun({ updatedAt: "2026-10-02T00:00:00.000Z", papers: 3 });
    const rows = renderLegacySettings(tab);
    const button = componentOf(rows.get("Topics from library"), ButtonComponent as never) as ButtonComponent;
    button.buttonEl.click();
    await vi.waitFor(() =>
      expect(tab.plugin.openPersonalLibraryDirectionReview).toHaveBeenCalledWith({ generateIfMissing: true }));
  });

  it("disables review during preparation even with an older usable index", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    vi.mocked(tab.plugin.getLibraryConnectionStatus).mockReturnValue({ kind: "authorization-required", rootLabel: "papers" });
    tab.plugin.libraryIndexStatus.setLastRun({ updatedAt: "2026-10-02T00:00:00.000Z", papers: 3 });
    tab.plugin.libraryIndexStatus.beginRun("preparing", "scanning");
    const row = renderLegacySettings(tab).get("Topics from library")!;
    expect(row.controlEl.querySelector("button")?.disabled).toBe(true);
    tab.plugin.libraryIndexStatus.endRun();
    const restored = renderLegacySettings(tab).get("Topics from library")!;
    expect(restored.controlEl.querySelector("button")?.disabled).toBe(false);
  });
});

describe("legacy embedding rows", () => {
  it("routes the mode dropdown through the same in-place consent as the 1.13+ row", async () => {
    const { tab, settings } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    const apply = vi
      .spyOn(tab, "applyEmbeddingModeChange")
      .mockImplementation(async () => false);
    const rows = renderLegacySettings(tab);
    const dropdown = componentOf(rows.get("Embedding"), DropdownComponent as never) as DropdownComponent;

    await dropdown.trigger("remote");

    expect(apply).toHaveBeenCalledWith("remote");
    // Declined switches leave both the setting and the dropdown on local.
    expect(settings.embedding.mode).toBe("local");
    expect(dropdown.selectEl.value).toBe("local");
  });

  it("labels the local option honestly about the one-time model download, not as always-offline", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    const rows = renderLegacySettings(tab);
    const dropdown = componentOf(rows.get("Embedding"), DropdownComponent as never) as DropdownComponent;
    const local = [...dropdown.selectEl.options].find((option) => option.value === "local");
    expect(local?.textContent).toBe("Local (default, one-time model download)");
    expect(local?.textContent).not.toMatch(/offline, default/);
  });

  it("describes the local embedding row's one-time download and approximate size, not a bundled model", () => {
    const { tab } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    const rows = renderLegacySettings(tab);
    const desc = rows.get("Embedding")?.descEl?.textContent ?? "";
    expect(desc).toContain("downloads its model once");
    expect(desc).toContain("130 MB");
    expect(desc).not.toContain("bundled");
  });

  it("routes the endpoint field through the shared re-ask on change", async () => {
    const { tab, settings } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    settings.embedding.mode = "remote";
    settings.embedding.baseUrl = "https://embed.example.com/v1";
    const save = vi
      .spyOn(tab, "saveEmbeddingEndpointField")
      .mockResolvedValue("https://embed.example.com/v1");
    const rows = renderLegacySettings(tab);
    const input = componentOf(
      rows.get("Embedding API base URL"),
      TextComponent as never,
    ) as TextComponent;

    input.inputEl.value = "https://elsewhere.example.com/v1";
    input.inputEl.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(save).toHaveBeenCalled());

    expect(save).toHaveBeenCalledWith(
      "embedding.baseUrl",
      "https://elsewhere.example.com/v1",
    );
    expect(input.inputEl.value).toBe("https://embed.example.com/v1");
  });
});

describe("legacy reasoning effort", () => {
  it("shows Thinking mode on after choosing an effort turns it on", async () => {
    const { tab, settings } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    settings.llm.thinkingMode = false;
    const rows = renderLegacySettings(tab);
    const thinking = componentOf(rows.get("Thinking mode"), ToggleComponent as never) as ToggleComponent;
    const effort = componentOf(rows.get("Reasoning effort"), DropdownComponent as never) as DropdownComponent;

    await effort.trigger("high");

    expect(settings.llm.thinkingMode).toBe(true);
    expect(thinking.value).toBe(true);
  });
});

describe("legacy model field", () => {
  it("saves a typed model on change and restores it when persistence fails", async () => {
    const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
    const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
    const rows = renderLegacySettings(tab);
    const input = rows.get("Model")?.controlEl.querySelector<HTMLInputElement>(
      "input.arxiv-daily-settings__model-input",
    );
    expect(input).toBeTruthy();

    input!.value = "candidate-model";
    input!.dispatchEvent(new Event("change"));

    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(input!.value).toBe(DEFAULT_SETTINGS.llm.model));
    expect(settings.llm.model).toBe(DEFAULT_SETTINGS.llm.model);
  });
});

describe("legacy text fields act when editing ends", () => {
  function typeInto(input: HTMLInputElement, text: string): void {
    for (const character of text) {
      input.value += character;
      input.dispatchEvent(new Event("input"));
    }
  }

  it("does not turn each keystroke of a custom category into a category", async () => {
    const { tab, settings } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    const rows = renderLegacySettings(tab);
    const display = vi.spyOn(tab, "display");
    const input = componentOf(rows.get("Category 1"), TextComponent as never) as TextComponent;

    typeInto(input.inputEl, "cs.LG");
    expect(settings.arxiv.categories).toEqual(["astro-ph"]);
    expect(display).not.toHaveBeenCalled();

    input.inputEl.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(settings.arxiv.categories).toEqual(["cs.LG"]));
  });

  it.each([
    ["Embedding API base URL", "embedding.baseUrl", "https://embed.example.com/v1"],
    ["Embedding model", "embedding.model", "text-embedding-3-large"],
  ])("saves %s once on change, not per keystroke", async (name, key, typed) => {
    const { tab, settings } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    settings.embedding.mode = "remote";
    const save = vi.spyOn(tab, "saveEmbeddingEndpointField").mockImplementation(
      async (_key, next) => next,
    );
    const rows = renderLegacySettings(tab);
    const input = componentOf(rows.get(name), TextComponent as never) as TextComponent;
    input.inputEl.value = "";

    typeInto(input.inputEl, typed);
    expect(save).not.toHaveBeenCalled();

    input.inputEl.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(save).toHaveBeenCalledTimes(1));
    expect(save).toHaveBeenCalledWith(key, typed);
  });

  it("moves both sidecar URLs on change and not while typing", async () => {
    const { tab, settings } = makeLegacyApiKeyTab(vi.fn().mockResolvedValue(undefined));
    settings.pdfParserSidecar.enabled = true;
    const rows = renderLegacySettings(tab);
    const input = componentOf(
      rows.get("Sidecar capability URL"),
      TextComponent as never,
    ) as TextComponent;
    input.inputEl.value = "";

    typeInto(input.inputEl, "http://127.0.0.1:5002/v1/capabilities");
    expect(settings.pdfParserSidecar.capabilitiesUrl).toBe(
      DEFAULT_SETTINGS.pdfParserSidecar.capabilitiesUrl,
    );

    input.inputEl.dispatchEvent(new Event("change"));
    await vi.waitFor(() =>
      expect(settings.pdfParserSidecar.parseUrl).toBe("http://127.0.0.1:5002/v1/parse"));
  });

  it("restores the sidecar toggle and reports when enabling is rejected", async () => {
    const { tab, settings } = makeLegacyApiKeyTab(vi.fn().mockRejectedValue(new Error("disk full")));
    const rows = renderLegacySettings(tab);
    const toggle = componentOf(rows.get("Better PDF parser"), ToggleComponent as never) as ToggleComponent;

    await expect(toggle.trigger(true)).resolves.toBeUndefined();

    expect(settings.pdfParserSidecar.enabled).toBe(false);
    expect(toggle.value).toBe(false);
  });
});

describe("legacy transactional renderers", () => {
  it.each([
    ["API base URL", TextComponent, "https://candidate.example/v1", "llm", "baseUrl"],
    ["Thinking mode", ToggleComponent, false, "llm", "thinkingMode"],
    ["Reasoning effort", DropdownComponent, "high", "llm", "reasoningEffort"],
    ["Link style", DropdownComponent, "relative", "output", "linkStyle"],
    ["Summary language", DropdownComponent, "en", "output", "summaryLanguage"],
    ["How to send", DropdownComponent, "hosted", "email", "mode"],
    ["Your email", TextComponent, "new@example.com", "email", "to"],
    ["From email", TextComponent, "sender@example.com", "email", "fromEmail"],
    ["From name", TextComponent, "Candidate sender", "email", "fromName"],
    ["Daily auto-send", ToggleComponent, true, "email", "enabled"],
  ] as const)(
    "keeps live and displayed %s unchanged when candidate persistence fails",
    async (name, componentType, next, section, field) => {
      const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
      const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
      const previous = (settings[section] as unknown as Record<string, unknown>)[field];
      const rows = renderLegacySettings(tab);
      const component = componentOf(rows.get(name), componentType as never) as {
        trigger(value: never): Promise<void>;
        inputEl?: HTMLInputElement;
        selectEl?: HTMLSelectElement;
        value?: boolean;
      };

      await component.trigger(next as never).catch(() => undefined);

      expect((settings[section] as unknown as Record<string, unknown>)[field]).toBe(previous);
      const displayed = component.inputEl?.value ?? component.selectEl?.value ?? component.value;
      expect(displayed).toBe(previous);
      expect(persistSettings).toHaveBeenCalledTimes(1);
    },
  );

  it("restores the complete named detail profile when persistence fails", async () => {
    const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
    const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
    const previous = structuredClone(settings.detailSelection);
    const rows = renderLegacySettings(tab);
    const profile = componentOf(
      rows.get("Automatic detail notes"),
      DropdownComponent as never,
    ) as DropdownComponent;

    await profile.trigger("conservative").catch(() => undefined);

    expect(settings.detailSelection).toEqual(previous);
    expect(profile.selectEl.value).toBe(previous.profile);
    expect(persistSettings).toHaveBeenCalledTimes(1);
  });

  it("does not let an earlier failed text save overwrite a later queued draft or commit", async () => {
    let rejectFirst!: (error: Error) => void;
    let resolveSecond!: () => void;
    const persistSettings = vi.fn()
      .mockImplementationOnce(() => new Promise<void>((_resolve, reject) => {
        rejectFirst = reject;
      }))
      .mockImplementationOnce(() => new Promise<void>((resolve) => {
        resolveSecond = resolve;
      }));
    const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
    const rows = renderLegacySettings(tab);
    const input = componentOf(
      rows.get("API base URL"),
      TextComponent as never,
    ) as TextComponent;

    const first = input.trigger("https://rejected.example/v1");
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    const second = input.trigger("https://accepted.example/v1");
    rejectFirst(new Error("first save failed"));
    await first.catch(() => undefined);
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(2));
    expect(input.inputEl.value).toBe("https://accepted.example/v1");

    resolveSecond();
    await second;
    expect(settings.llm.baseUrl).toBe("https://accepted.example/v1");
    expect(input.inputEl.value).toBe("https://accepted.example/v1");
  });

  it("coalesces change and blur into one rejected tick-interval transaction", async () => {
    const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
    const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
    const input = document.createElement("input");
    input.value = "5";

    (tab as unknown as {
      bindTickIntervalInput(inputEl: HTMLInputElement): void;
    }).bindTickIntervalInput(input);
    input.value = "5";
    input.dispatchEvent(new Event("change"));
    input.dispatchEvent(new Event("blur"));

    await vi.waitFor(() => expect(input.value).toBe(
      String(DEFAULT_SETTINGS.schedule.tickIntervalMin),
    ));
    expect(persistSettings).toHaveBeenCalledTimes(1);
    expect(settings.schedule.tickIntervalMin).toBe(
      DEFAULT_SETTINGS.schedule.tickIntervalMin,
    );
  });

  it("persists a candidate before committing the live secret and redaction", async () => {
    let finishSave: (() => void) | undefined;
    const persistSettings = vi.fn(async (candidate: typeof DEFAULT_SETTINGS) => {
      expect(candidate.llm.apiKey).toBe("new-secret");
      await new Promise<void>((resolve) => { finishSave = resolve; });
    });
    const { tab, settings, refreshSensitiveValues } = makeLegacyApiKeyTab(
      persistSettings,
    );
    const { input } = renderLegacyApiKey(tab);

    expect(input.value).toBe(settings.llm.apiKey);
    input.value = "new-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    expect(settings.llm.apiKey).toBe("old-secret");
    expect(refreshSensitiveValues).not.toHaveBeenCalled();
    finishSave?.();

    await vi.waitFor(() => expect(settings.llm.apiKey).toBe("new-secret"));
    expect(refreshSensitiveValues).toHaveBeenCalledTimes(1);
    expect(input.value).toBe("new-secret");
  });

  it("does not let an earlier failed legacy secret save hide a later queued draft", async () => {
    let rejectFirst!: (error: Error) => void;
    let resolveSecond!: () => void;
    const persistSettings = vi.fn()
      .mockImplementationOnce(() => new Promise<void>((_resolve, reject) => {
        rejectFirst = reject;
      }))
      .mockImplementationOnce(() => new Promise<void>((resolve) => {
        resolveSecond = resolve;
      }));
    const { tab, settings } = makeLegacyApiKeyTab(persistSettings);
    const { input } = renderLegacyApiKey(tab);

    input.value = "rejected-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    input.value = "accepted-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    rejectFirst(new Error("first save failed"));
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(2));
    expect(input.value).toBe("accepted-secret");

    resolveSecond();
    await vi.waitFor(() => expect(settings.llm.apiKey).toBe("accepted-secret"));
    expect(input.value).toBe("accepted-secret");
  });

  it("restores the masked value when candidate persistence fails", async () => {
    const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
    const { tab, settings, refreshSensitiveValues } = makeLegacyApiKeyTab(
      persistSettings,
    );
    const { input } = renderLegacyApiKey(tab);

    input.value = "new-secret";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

    await vi.waitFor(() => expect(tab.plugin.logger.error).toHaveBeenCalled());
    expect(settings.llm.apiKey).toBe("old-secret");
    expect(refreshSensitiveValues).not.toHaveBeenCalled();
    expect(input.value).toBe("old-secret");
  });

  it.each([
    ["renderEmailApiKeySetting", "apiKey"],
    ["renderHostedTokenSetting", "hostedToken"],
  ] as const)(
    "restores the masked %s secret when candidate persistence fails",
    async (renderMethod, settingKey) => {
      const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
      const { tab, settings, refreshSensitiveValues } = makeLegacyApiKeyTab(
        persistSettings,
      );
      settings.email[settingKey] = "old-email-secret";
      const container = document.createElement("div");
      const render = Reflect.get(tab, renderMethod) as (
        containerEl: HTMLElement,
      ) => void;
      render.call(tab, container);
      const input = container.querySelector("input") as HTMLInputElement;

      input.value = "new-email-secret";
      input.dispatchEvent(new Event("input"));
      input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

      await vi.waitFor(() => expect(tab.plugin.logger.error).toHaveBeenCalled());
      expect(settings.email[settingKey]).toBe("old-email-secret");
      expect(refreshSensitiveValues).not.toHaveBeenCalled();
      expect(input.value).toBe("old-email-secret");
    },
  );
});

describe("settings tab regressions", () => {
  it("uses Obsidian 1.4-compatible title and ARIA help text", () => {
    const attachHelpBody = settingsTabSource.match(
      /private attachHelp[\s\S]*?\n  private reportActionError/,
    )?.[0];
    expect(attachHelpBody).toContain('title: text, "aria-label": text');
    expect(settingsTabSource).not.toContain("setTooltip");
  });

  it("uses scoped element creation in production settings code", () => {
    expect(settingsTabSource).not.toContain("document.createElement");
  });

  it("reports focused fire-and-forget failures instead of swallowing them", () => {
    expect(settingsTabSource).toContain("this.plugin.logger.error(`settings: ${action} failed`");
    expect(settingsTabSource).toContain("new Notice(`arXiv Daily: ${action} failed:");
    expect(settingsTabSource).not.toContain(".catch(() => {})");
    expect(settingsTabSource).toContain('this.runAction("update daily path"');
    expect(settingsTabSource).toContain('this.runAction("generate first report"');
    expect(settingsTabSource).toContain('this.runAction("open dashboard"');
    expect(settingsTabSource).toContain('this.reportActionError("save run window"');
  });

  it("renders an accessible five-step first-report guide without duplicate inputs", () => {
    const guideBody = settingsTabSource.match(
      /public createSetupGuide\(\)[\s\S]*?\n  private renderSetupItem/,
    )?.[0];
    expect(guideBody).toBeDefined();
    expect(guideBody).toContain('createEl("ol"');
    expect(settingsTabSource).toContain('parent.createEl("li"');
    expect(guideBody).toContain('text: `${completedCount} of 5 complete`');
    expect(guideBody).toContain('"Connect AI"');
    expect(guideBody).toContain('"Choose paper sources"');
    expect(guideBody).toContain('"Describe your research interests"');
    expect(guideBody).toContain('"Generate your first report"');
    expect(settingsTabSource).toContain('text: done ? "Complete" : "Next"');
    expect(settingsTabSource).not.toContain('text: done ? "Done"');
    expect(guideBody).not.toContain("new Setting(");
    expect(guideBody).not.toContain("PROVIDER_PRESETS");
  });

  it("uses run-state completion, awaits the first report, and renders compact completion", () => {
    const guideBody = settingsTabSource.match(
      /public createSetupGuide\(\)[\s\S]*?\n  private renderSetupItem/,
    )?.[0];
    const firstReportBody = settingsTabSource.match(
      /public async generateFirstReport\(\)[\s\S]*?\n  private renderTopicCard/,
    )?.[0];
    expect(guideBody).toContain("this.plugin.stateStore.snapshot()");
    expect(guideBody).toContain("status.firstReportComplete");
    expect(guideBody).toContain('"Generate first report"');
    expect(guideBody).toContain('this.runAction("generate first report"');
    expect(firstReportBody).toContain("await this.plugin.scheduler.runForDateNow(date)");
    expect(firstReportBody).toContain("this.refreshSetupGuide()");
    expect(settingsTabSource).not.toContain('this.executeCommand("run-now")');
    expect(guideBody).toContain('guide.addClass("arxiv-daily-setup--complete")');
    expect(guideBody).toContain('text: "Setup complete"');
    expect(guideBody).toContain("status.latestCompletedReportDate");
    expect(settingsTabSource).toContain('text: "Open dashboard"');
  });

  it("keeps validation reasons in guide details and removes the duplicate banner", () => {
    expect(settingsTabSource).toContain('details.createEl("summary", { text: "Configuration details" })');
    expect(settingsTabSource).toContain("status.schedulerReasons");
    expect(settingsTabSource).toContain("for (const reason of reasons)");
    expect(settingsTabSource).not.toContain('text: "Configuration incomplete"');
    expect(settingsTabSource).not.toContain("arxiv-daily-settings__invalid-banner");
  });

  it("focuses setup targets and respects reduced motion through ownerDocument", () => {
    const scrollBody = settingsTabSource.match(
      /private scrollToSection\([\s\S]*?\n  public async generateFirstReport/,
    )?.[0];
    expect(scrollBody).toContain("target.ownerDocument.defaultView");
    expect(scrollBody).toContain('matchMedia?.("(prefers-reduced-motion: reduce)")');
    expect(scrollBody).toContain('target.setAttribute("tabindex", "-1")');
    expect(scrollBody).toContain('behavior: reduceMotion ? "auto" : "smooth"');
    expect(scrollBody).toContain("focus({ preventScroll: true })");
  });

  it("uses clear sentence-case labels", () => {
    expect(settingsTabSource).toContain('"Paper categories"');
    expect(settingsTabSource).toContain('"Research topics"');
    expect(settingsTabSource).toContain('"Output & schedule"');
    expect(settingsTabSource).toContain('"API key"');
    expect(settingsTabSource).not.toContain('"+ Add Category"');
  });

  it("renders the saved API key masked with a Show/Hide toggle, no replace/clear actions", () => {
    const apiKeyBody = settingsTabSource.match(
      /private renderApiKeySetting\([\s\S]*?\n  private renderSetupGuide/,
    )?.[0];
    expect(apiKeyBody).toBeDefined();
    expect(apiKeyBody).toContain('renderSensitiveInput(this, setting, {');
    expect(apiKeyBody).toContain('value: this.plugin.settings.llm.apiKey');
    expect(apiKeyBody).not.toContain("API_KEY_CONFIGURED_SENTINEL");
    expect(apiKeyBody).not.toContain('text: configured ? "Replace" : "Save"');
    expect(apiKeyBody).not.toContain('text: "Cancel"');
    expect(apiKeyBody).not.toContain('text: "Clear"');
  });

  it("warns that quick-start templates replace categories", () => {
    expect(settingsTabSource).toContain("and arXiv categories");
  });

  it("uses accessible topic disclosure controls and associated field labels", () => {
    expect(settingsTabSource).toContain('card.createEl("button"');
    expect(settingsTabSource).toContain('"aria-expanded": String(isExpanded)');
    expect(settingsTabSource).toContain('"aria-controls": formId');
    expect(settingsTabSource).toContain("form.hidden = !isExpanded");
    expect(settingsTabSource).toContain('attr: { for: nameId }');
    expect(settingsTabSource).toContain('attr: { for: dirId }');
    expect(settingsTabSource).toContain('"aria-describedby": nameHintId');
  });



  it("renders one understandable automatic detail-note setting near topics", () => {
    const headingIndex = settingsTabSource.indexOf('"Research topics"');
    const policyIndex = settingsTabSource.indexOf('"Automatic detail notes"');
    const timezoneIndex = settingsTabSource.indexOf('.setName("Timezone")');
    expect(policyIndex).toBeGreaterThan(headingIndex);
    expect(policyIndex).toBeLessThan(timezoneIndex);
    expect(settingsTabSource).toContain(
      "Only topics with detail report turned on are considered",
    );
    expect(settingsTabSource).toContain(
      'Manual “summarize paper” is unchanged',
    );
    expect(settingsTabSource).toContain('.addOption("conservative", "Fewer")');
    expect(settingsTabSource).toContain('.addOption("balanced", "Recommended")');
    expect(settingsTabSource).toContain('.addOption("broad", "More")');
    expect(settingsTabSource).toContain('d.addOption("custom", "Custom (current values)")');
    expect(settingsTabSource).toContain('s.detailSelection.profile === "custom"');
    expect(settingsTabSource).toContain("detailSelectionPreset(profile)");
    expect(settingsTabSource).toContain("await this.plugin.saveSettings()");
  });

  it("does not expose automatic detail thresholds or numeric controls", () => {
    expect(settingsTabSource).not.toContain('"Normal threshold"');
    expect(settingsTabSource).not.toContain('"Exceptional threshold"');
    expect(settingsTabSource).not.toContain('"Soft limit"');
    expect(settingsTabSource).not.toContain("renderDetailSelectionNumber");
    expect(settingsTabSource).not.toContain("detail-selection-number");
  });

  it("uses explicit Start and End labels with non-cyclic select controls", () => {
    expect(settingsTabSource).toContain('"Start"');
    expect(settingsTabSource).toContain('"End"');
    expect(settingsTabSource).toContain('field.createEl("select"');
    expect(settingsTabSource).not.toContain('inputEl.type = "time"');
  });

  it("does not normalize or persist categories merely while displaying them", () => {
    expect(settingsTabSource).toContain("const categories = arxivCategories(s.arxiv);");
    expect(settingsTabSource).toContain("arxiv.categories = normalized;");
    expect(settingsTabSource).toMatch(
      /const apply = async \(\) => \{[\s\S]*?arxiv\.categories = \[tpl\.category\];/,
    );
  });
});

describe("output path drafts", () => {
  it("normalizes safe vault-relative directories", () => {
    expect(validateOutputDirectoryDraft(" arxiv\\papers/details ")).toEqual({
      ok: true,
      value: "arxiv/papers/details",
    });
  });

  it("rejects a sibling directory collision portably", () => {
    expect(validateOutputDirectoryDraft("cafe\u0301/NOTES", "Café/notes")).toEqual({
      ok: false,
      reason: "Daily and papers directories must be different",
    });
  });

  it("rejects empty, absolute, traversal, and configuration paths", () => {
    expect(validateOutputDirectoryDraft("").ok).toBe(false);
    expect(validateOutputDirectoryDraft("/tmp/papers").ok).toBe(false);
    expect(validateOutputDirectoryDraft("C:/papers").ok).toBe(false);
    expect(validateOutputDirectoryDraft("arxiv/../notes").ok).toBe(false);
    expect(validateOutputDirectoryDraft(".obsidian/plugins").ok).toBe(false);
  });

  it("persists a legacy output candidate before committing and installing stores", async () => {
    let finishSave: (() => void) | undefined;
    const persistSettings = vi.fn(async (candidate: typeof DEFAULT_SETTINGS) => {
      expect(candidate.output.dailyDir).toBe("reports/daily");
      await new Promise<void>((resolve) => { finishSave = resolve; });
    });
    const { tab, settings, installOutputStores } = makeLegacyApiKeyTab(
      persistSettings,
    );
    const input = document.createElement("input");
    const apply = Reflect.get(tab, "applyOutputDirectoryDraft") as (
      key: "dailyDir" | "papersDir",
      draft: string,
      inputEl: HTMLInputElement,
    ) => Promise<void>;

    const changing = apply.call(tab, "dailyDir", " reports\\daily ", input);
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledOnce());
    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(installOutputStores).not.toHaveBeenCalled();
    finishSave?.();
    await changing;

    expect(settings.output.dailyDir).toBe("reports/daily");
    expect(input.value).toBe("reports/daily");
    expect(installOutputStores).toHaveBeenCalledTimes(1);
  });

  it("restores the legacy output input without a second persistence rollback", async () => {
    const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
    const { tab, settings, installOutputStores } = makeLegacyApiKeyTab(
      persistSettings,
    );
    const input = document.createElement("input");
    input.value = "reports/daily";
    const apply = Reflect.get(tab, "applyOutputDirectoryDraft") as (
      key: "dailyDir" | "papersDir",
      draft: string,
      inputEl: HTMLInputElement,
    ) => Promise<void>;

    await apply.call(tab, "dailyDir", input.value, input);

    expect(settings.output.dailyDir).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(input.value).toBe(DEFAULT_SETTINGS.output.dailyDir);
    expect(persistSettings).toHaveBeenCalledTimes(1);
    expect(installOutputStores).not.toHaveBeenCalled();
  });
});

describe("run window time options", () => {
  it("renders standard 24-hour quarter-hour values without 24:00", () => {
    const options = runWindowTimeOptions("09:00");
    expect(options).toHaveLength(96);
    expect(options[0]).toMatchObject({ value: "00:00", label: "00:00" });
    expect(options.at(-1)).toMatchObject({ value: "23:45", label: "23:45" });
    expect(options.some((option) => option.value === "24:00")).toBe(false);
  });

  it("preserves arbitrary valid minutes as a selectable value", () => {
    const options = runWindowTimeOptions("09:07");
    expect(options).toContainEqual({ value: "09:07", label: "09:07", valid: true });
    expect(isValidLocalTime("09:07")).toBe(true);
  });

  it("displays invalid and legacy values without treating them as persistable", () => {
    expect(runWindowTimeOptions("24:00")).toContainEqual({
      value: "24:00",
      label: "24:00 — invalid",
      valid: false,
    });
    expect(isValidLocalTime("24:00")).toBe(false);
    expect(isValidLocalTime("9:00 AM")).toBe(false);
  });
});

describe("confirmEmbeddingMode", () => {
  beforeAll(() => {
    const proto = HTMLElement.prototype as any;
    proto.empty ??= function () { this.replaceChildren(); };
    proto.setText ??= function (text: string) { this.textContent = text; };
    proto.addClass ??= function (...classes: string[]) { this.classList.add(...classes); };
    proto.createEl ??= function (tag: string, options: { cls?: string; text?: string; attr?: Record<string, string> } = {}) {
      const element = document.createElement(tag);
      if (options.cls) element.className = options.cls;
      if (options.text !== undefined) element.textContent = options.text;
      for (const [key, value] of Object.entries(options.attr ?? {})) element.setAttribute(key, value);
      this.appendChild(element);
      return element;
    };
    proto.createDiv ??= function (options: { cls?: string; text?: string } = {}) {
      return this.createEl!("div", options);
    };
  });

  it("returns remote when the remote button is clicked", async () => {
    const { Modal } = await import("obsidian");
    Modal.opened.length = 0;
    const promise = confirmEmbeddingMode({} as any);
    const modal = Modal.opened.at(-1)!;
    [...modal.contentEl.querySelectorAll<HTMLButtonElement>("button")]
      .find((button) => button.textContent === "Remote")!.click();
    await expect(promise).resolves.toBe("remote");
  });

  it("keeps the local default when the modal is dismissed", async () => {
    const { Modal } = await import("obsidian");
    Modal.opened.length = 0;
    const promise = confirmEmbeddingMode({} as any);
    Modal.opened.at(-1)!.close();
    await expect(promise).resolves.toBe("local");
  });

  it("discloses the one-time model download honestly, not a bundled/always-offline model", async () => {
    const { Modal } = await import("obsidian");
    Modal.opened.length = 0;
    const promise = confirmEmbeddingMode({} as any);
    const modal = Modal.opened.at(-1)!;
    const text = modal.contentEl.textContent ?? "";
    expect(text).toContain("Local (default, one-time model download)");
    expect(text).toContain("Downloads its model once");
    expect(text).toContain("130 MB");
    expect(text).not.toContain("bundled");
    Modal.opened.at(-1)!.close();
    await promise;
  });

  it("keeps embedding configurable without an extra first-selection popup", () => {
    expect(settingsTabSource).not.toContain("offerEmbeddingModeChoice()");
    expect(settingsTabSource).not.toContain("confirmEmbeddingMode(this.app)");
    expect(settingsTabSource).toContain("initialChoiceDone");
    expect(settingsTabSource).toContain("about 130 MB");
  });
});

describe("personal library settings layout", () => {
  // Whether the buttons actually land on one line is a question for a layout
  // engine, and happy-dom has none: it is settled by the desktop acceptance
  // harness (`library-row-geometry`), which measures the real renderer. What
  // can be settled here is the declaration that made the row overflow — a
  // control box allowed to shrink under buttons that refuse to shrink with it.
  const libraryControlsBlock = (): string => {
    const css = readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");
    const match = css.match(
      /\.arxiv-daily-settings \.setting-item-control\.arxiv-daily-settings__library-controls \{([^}]*)\}/,
    );
    expect(match).not.toBeNull();
    return match![1];
  };

  it("keeps the library action buttons hugging the right edge", () => {
    expect(libraryControlsBlock()).toMatch(/justify-content:\s*flex-end;/);
  });

  it("caps the button strip so the description column always keeps a reading width", () => {
    const block = libraryControlsBlock();
    // The cap, not a panel-width threshold, is what decides whether the buttons
    // wrap — so the strip must be capped and must be the one thing that cannot
    // shrink out from under it. The cap is stated as the reading floor rather
    // than as a number of its own, so the two cannot drift apart.
    expect(block).toMatch(
      /max-width:\s*calc\(100% - var\(--arxiv-daily-library-description-floor\) - \d+px\);/,
    );
    expect(block).toMatch(/flex-shrink:\s*0;/);
    expect(block).toMatch(/flex-wrap:\s*wrap;/);
  });

  it("derives the whole bargain from one declared reading floor", () => {
    const css = readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");
    // The floor is declared exactly once, and it is a real width — the desktop
    // harness refuses anything under 150px as unreadable.
    const declarations = css.match(/--arxiv-daily-library-description-floor:\s*(\d+)px;/g) ?? [];
    expect(declarations).toHaveLength(1);
    const floor = Number(/(\d+)px/.exec(declarations[0])![1]);
    expect(floor).toBeGreaterThanOrEqual(150);
  });

  it("never keys the layout off how many buttons the row happens to carry", () => {
    // The two states used to get different caps, which made the three-button
    // row read better than the two-button one. Readability is one invariant,
    // so there is one rule.
    const css = readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");
    expect(css).not.toMatch(/library-controls[^{]*nth-of-type/);
  });

  it("gives the description column a floor rather than letting it collapse", () => {
    const css = readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");
    const match = css.match(
      /\.arxiv-daily-settings \.setting-item:has\(> \.arxiv-daily-settings__library-controls\) \.setting-item-info \{([^}]*)\}/,
    );
    expect(match).not.toBeNull();
    expect(match![1]).toMatch(/min-width:\s*var\(--arxiv-daily-library-description-floor\);/);
  });

  it("drops the cap where Obsidian stacks the row and there is no second column", () => {
    const css = readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");
    expect(css).toMatch(/@container \(max-width: 340px\) \{[\s\S]*?max-width:\s*100%;/);
  });
});

describe("legacy topic card header", () => {
  function renderLegacyTopicCard(
    tab: ArxivDailySettingTab,
    topics: typeof DEFAULT_SETTINGS.arxiv.topics,
  ): HTMLElement {
    const container = document.createElement("div");
    const render = Reflect.get(tab, "renderTopicCard") as (
      containerEl: HTMLElement,
      topics: unknown[],
      index: number,
    ) => void;
    render.call(tab, container, topics, 0);
    return container;
  }

  it("hides the tag chip while collapsed so a long name keeps the header width", () => {
    const { tab } = makeLegacyApiKeyTab(async () => {});
    const container = renderLegacyTopicCard(tab, [{ directions: [{ id: "fixture-direction", text: "Something", origin: "manual" as const }],
      id: "topic-1",
      name: "A very long research topic name that would otherwise get truncated",
      tag: "long-topic-tag",
      description: "Something",
      detail: false,
    }]);

    expect(container.querySelector(".arxiv-daily-settings__topic-tag")).toBeNull();

    const header = container.querySelector(
      ".arxiv-daily-settings__topic-header",
    ) as HTMLButtonElement;
    header.click();

    expect(container.querySelector(".arxiv-daily-settings__topic-tag")).toBeNull();
  });
});
