import { beforeAll, describe, expect, it, vi } from "vitest";
import {
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
import {
  API_KEY_CONFIGURED_SENTINEL,
  ArxivDailySettingTab,
  isValidLocalTime,
  llmHttpWarning,
  modelFetchNoticeMessage,
  runWindowTimeOptions,
  validateOutputDirectoryDraft,
} from "../src/settings/tab";
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
  } as unknown as ArxivDailyPlugin;
  const tab = new ArxivDailySettingTab({} as App, plugin);
  vi.spyOn(tab, "refreshSetupGuide").mockImplementation(() => undefined);
  return { tab, settings, refreshSensitiveValues, installOutputStores };
}

function renderLegacyApiKey(tab: ArxivDailySettingTab) {
  const container = document.createElement("div");
  const render = Reflect.get(tab, "renderApiKeySetting") as (
    containerEl: HTMLElement,
  ) => void;
  render.call(tab, container);
  const input = container.querySelector("input");
  const buttons = [...container.querySelectorAll("button")];
  if (!input || buttons.length !== 3) throw new Error("API key row did not render");
  return {
    input,
    replace: buttons[0] as HTMLButtonElement,
    cancel: buttons[1] as HTMLButtonElement,
    clear: buttons[2] as HTMLButtonElement,
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

describe("legacy transactional renderers", () => {
  it.each([
    ["API base URL", TextComponent, "https://candidate.example/v1", "llm", "baseUrl"],
    ["Model", DropdownComponent, "candidate-model", "llm", "model"],
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
    const { input, replace } = renderLegacyApiKey(tab);

    expect(input.value).toBe(API_KEY_CONFIGURED_SENTINEL);
    expect(input.value).not.toContain(settings.llm.apiKey);
    replace.click();
    input.value = "new-secret";
    input.dispatchEvent(new Event("input"));
    replace.click();

    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    expect(settings.llm.apiKey).toBe("old-secret");
    expect(refreshSensitiveValues).not.toHaveBeenCalled();
    finishSave?.();

    await vi.waitFor(() => expect(settings.llm.apiKey).toBe("new-secret"));
    expect(refreshSensitiveValues).toHaveBeenCalledTimes(1);
    expect(input.value).toBe(API_KEY_CONFIGURED_SENTINEL);
    expect(input.readOnly).toBe(true);
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
    const { input, replace } = renderLegacyApiKey(tab);

    replace.click();
    input.value = "rejected-secret";
    input.dispatchEvent(new Event("input"));
    replace.click();
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(1));
    input.value = "accepted-secret";
    input.dispatchEvent(new Event("input"));
    replace.click();
    rejectFirst(new Error("first save failed"));
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledTimes(2));
    expect(input.value).toBe("accepted-secret");
    expect(input.readOnly).toBe(false);

    resolveSecond();
    await vi.waitFor(() => expect(settings.llm.apiKey).toBe("accepted-secret"));
    expect(input.value).toBe(API_KEY_CONFIGURED_SENTINEL);
    expect(input.readOnly).toBe(true);
  });

  it("restores the hidden configured state when candidate persistence fails", async () => {
    const persistSettings = vi.fn().mockRejectedValue(new Error("disk full"));
    const { tab, settings, refreshSensitiveValues } = makeLegacyApiKeyTab(
      persistSettings,
    );
    const { input, replace } = renderLegacyApiKey(tab);

    replace.click();
    input.value = "new-secret";
    input.dispatchEvent(new Event("input"));
    replace.click();

    await vi.waitFor(() => expect(tab.plugin.logger.error).toHaveBeenCalled());
    expect(settings.llm.apiKey).toBe("old-secret");
    expect(refreshSensitiveValues).not.toHaveBeenCalled();
    expect(input.value).toBe(API_KEY_CONFIGURED_SENTINEL);
    expect(input.readOnly).toBe(true);
    expect(replace.textContent).toBe("Replace");
  });

  it.each([
    ["renderEmailApiKeySetting", "apiKey"],
    ["renderHostedTokenSetting", "hostedToken"],
  ] as const)(
    "restores the hidden %s secret when candidate persistence fails",
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
      const replace = container.querySelector("button") as HTMLButtonElement;

      replace.click();
      input.value = "new-email-secret";
      input.dispatchEvent(new Event("input"));
      replace.click();

      await vi.waitFor(() => expect(tab.plugin.logger.error).toHaveBeenCalled());
      expect(settings.email[settingKey]).toBe("old-email-secret");
      expect(refreshSensitiveValues).not.toHaveBeenCalled();
      expect(input.value).toBe(API_KEY_CONFIGURED_SENTINEL);
      expect(input.readOnly).toBe(true);
      expect(replace.textContent).toBe("Replace");
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
    expect(settingsTabSource).toContain('this.runAction("save selected model"');
    expect(settingsTabSource).toContain('this.reportActionError("save run window"');
  });

  it("renders an accessible four-step first-report guide without duplicate inputs", () => {
    const guideBody = settingsTabSource.match(
      /public createSetupGuide\(\)[\s\S]*?\n  private renderSetupItem/,
    )?.[0];
    expect(guideBody).toBeDefined();
    expect(guideBody).toContain('createEl("ol"');
    expect(settingsTabSource).toContain('parent.createEl("li"');
    expect(guideBody).toContain('text: `${completedCount} of 4 complete`');
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
    expect(guideBody).toContain("status.readyToRun ? \"Generate first report\" : undefined");
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
    expect(scrollBody).toContain("targetEl.ownerDocument.defaultView");
    expect(scrollBody).toContain('matchMedia?.("(prefers-reduced-motion: reduce)")');
    expect(scrollBody).toContain('targetEl.setAttribute("tabindex", "-1")');
    expect(scrollBody).toContain('behavior: reduceMotion ? "auto" : "smooth"');
    expect(scrollBody).toContain("targetEl.focus({ preventScroll: true })");
  });

  it("uses clear sentence-case labels", () => {
    expect(settingsTabSource).toContain('"arXiv categories"');
    expect(settingsTabSource).toContain('"Research topics"');
    expect(settingsTabSource).toContain('"Output & schedule"');
    expect(settingsTabSource).toContain('"API key"');
    expect(settingsTabSource).not.toContain('"+ Add Category"');
  });

  it("never renders the saved API key and requires explicit replace/save/cancel/clear actions", () => {
    const apiKeyBody = settingsTabSource.match(
      /private renderApiKeySetting\([\s\S]*?\n  private renderSetupGuide/,
    )?.[0];
    expect(apiKeyBody).toBeDefined();
    expect(apiKeyBody).not.toContain("input.value = this.plugin.settings.llm.apiKey");
    expect(apiKeyBody).not.toContain("setValue(s.llm.apiKey)");
    expect(apiKeyBody).toContain("API_KEY_CONFIGURED_SENTINEL");
    expect(apiKeyBody).toContain('text: configured ? "Replace" : "Save"');
    expect(apiKeyBody).toContain('text: "Cancel"');
    expect(apiKeyBody).toContain('text: "Clear"');
  });

  it("does not register a second change listener when models are fetched", () => {
    const showModelDropdownBody = settingsTabSource.match(
      /public showModelDropdown\([\s\S]*?\n  private textareaSetting/,
    )?.[0];

    expect(showModelDropdownBody).toBeDefined();
    expect(showModelDropdownBody).not.toContain('select.addEventListener("change"');
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
    expect(settingsTabSource).toContain('attr: { for: tagId }');
    expect(settingsTabSource).toContain('attr: { for: descId }');
    expect(settingsTabSource).toContain('"aria-describedby": nameHintId');
  });

  it("confirms topic deletion by name before persistence", () => {
    expect(settingsTabSource).toContain('Delete the research topic "${topicName}"?');
    expect(settingsTabSource).toContain("if (!confirmed) return");
    expect(settingsTabSource.indexOf("if (!confirmed) return")).toBeLessThan(
      settingsTabSource.indexOf("topics.splice(index, 1)"),
    );
  });

  it("renders one understandable automatic detail-note setting near topics", () => {
    const headingIndex = settingsTabSource.indexOf('"Research topics"');
    const policyIndex = settingsTabSource.indexOf('"Automatic detail notes"');
    const timezoneIndex = settingsTabSource.indexOf('.setName("Timezone")');
    expect(policyIndex).toBeGreaterThan(headingIndex);
    expect(policyIndex).toBeLessThan(timezoneIndex);
    expect(settingsTabSource).toContain(
      "Only topics with Detail report turned on are considered",
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
    expect(settingsTabSource).toContain(
      "this.plugin.settings.arxiv.categories = normalized;",
    );
    expect(settingsTabSource).toMatch(
      /const apply = async \(\) => \{[\s\S]*?s\.arxiv\.categories = \[tpl\.category\];/,
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
