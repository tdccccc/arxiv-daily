import { beforeAll, describe, expect, it, vi } from "vitest";
import type { App } from "obsidian";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import type ArxivDailyPlugin from "../main";
import { ArxivDailySettingTab } from "../src/settings/tab";
import {
  allSettingKeys,
  buildSettingDefinitions,
  readSettingValue,
  SETTING_KEYS,
} from "../src/settings/definitions";
import {
  renderApiKeyRow,
  renderCategoryRow,
  renderEmailApiKeyRow,
  renderEmailToRow,
  renderHostedTokenRow,
  renderReasoningEffortRow,
} from "../src/settings/declarative-rows";

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
    createEl?: (tag: string, options?: CreateOptions) => HTMLElement;
  };
  proto.empty ??= function () { this.replaceChildren(); };
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
});

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

/** Tab with a mocked plugin whose settings persist through saveSettings. */
function makeTab() {
  const settings = structuredClone(DEFAULT_SETTINGS);
  const saveSettings = vi.fn().mockResolvedValue(undefined);
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
    logger: { setSensitiveValues: vi.fn(), error: vi.fn() },
    refreshSensitiveValues: vi.fn(),
    sendHostedVerificationEmail: vi.fn().mockResolvedValue("Verification sent"),
    sendTestEmail: vi.fn().mockResolvedValue("Test sent"),
    getLibraryConnectionStatus: vi.fn().mockReturnValue({ kind: "disconnected" }),
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
  } as unknown as ArxivDailyPlugin;
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

describe("wired getSettingDefinitions", () => {
  it("returns non-empty definitions with section groups", () => {
    const { tab } = makeTab();
    const items = tab.getSettingDefinitions();
    expect(items.length).toBeGreaterThanOrEqual(4);
    const groups = items.filter((item) => item.type === "group");
    expect(groups.map((g) => g.heading)).toEqual(
      expect.arrayContaining([
        "LLM",
        "Output & schedule",
        "Email delivery",
        "Advanced",
        "Help & feedback",
      ]),
    );
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
        "Getting started",
        "Enable · Paused",
        "Timezone",
        "Run window",
        "Check every (minutes)",
        "Resend API key",
        "From email",
        "From name",
        "Daily auto-send",
      ]),
    );
  });

  it("routes list mutations and actions to the tab's public methods", async () => {
    const { tab, settings, saveSettings } = makeTab();
    vi.spyOn(tab, "refreshSettings").mockImplementation(() => {});

    const topicsList = tab
      .getSettingDefinitions()
      .find(
        (item) => item.type === "list" && item.heading === "Research topics",
      );
    await topicsList?.addItem?.action();
    expect(settings.arxiv.topics).toHaveLength(1);
    expect(saveSettings).toHaveBeenCalledTimes(1);
    expect(tab.refreshSettings).toHaveBeenCalledTimes(1);

    expect(topicsList?.onReorder).toBeUndefined();
  });
});

describe("personal library settings row", () => {
  function renderLibraryButtons(tab: ArxivDailySettingTab) {
    const buttons: Array<{
      text: string;
      cta: boolean;
      warning: boolean;
      click?: () => void;
    }> = [];
    const setting = {
      setDesc: vi.fn().mockReturnThis(),
      addButton(callback: (button: any) => void) {
        const state = { text: "", cta: false, warning: false } as {
          text: string;
          cta: boolean;
          warning: boolean;
          click?: () => void;
        };
        const button = {
          setButtonText(text: string) { state.text = text; return button; },
          setCta() { state.cta = true; return button; },
          setWarning() { state.warning = true; return button; },
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

  it("shows only folder selection while disconnected", () => {
    const { tab } = makeTab();
    const { buttons } = renderLibraryButtons(tab);
    expect(buttons.map((button) => button.text)).toEqual(["Choose folder"]);
  });

  it("shows authorization after selection and preview/revoke after approval", () => {
    const { tab, plugin } = makeTab();
    const runAction = vi.spyOn(tab, "runAction").mockImplementation(() => {});
    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorization-required",
      rootLabel: "papers",
    });
    const pending = renderLibraryButtons(tab).buttons;
    expect(pending.map((button) => button.text)).toEqual([
      "Change folder",
      "Preview",
      "Scan library",
      "Reload catalog",
      "Review & authorize",
    ]);
    expect(pending[4]?.cta).toBe(true);
    pending[4]?.click?.();
    expect(runAction).toHaveBeenCalledWith(
      "authorize personal library",
      expect.any(Function),
    );

    vi.mocked(plugin.getLibraryConnectionStatus).mockReturnValue({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: "2026-08-02T12:00:00.000Z",
    });
    const authorized = renderLibraryButtons(tab).buttons;
    expect(authorized.map((button) => button.text)).toEqual([
      "Change folder",
      "Preview",
      "Scan library",
      "Reload catalog",
      "Revoke",
    ]);
    expect(authorized[4]?.warning).toBe(true);
  });
});

describe("declarative LLM and category rows", () => {
  it("saves and clears the API key on blur and toggles visibility", async () => {
    const { tab, settings, saveSettings } = makeTab();
    vi.spyOn(tab, "refreshDeclarativeSetupGuide").mockImplementation(() => {});
    const setting = renderSetting();
    renderApiKeyRow(tab, setting as never);

    const input = setting.controlEl.querySelector("input") as HTMLInputElement;
    const reveal = setting.controlEl.querySelector("button") as HTMLButtonElement;
    expect(input.type).toBe("password");
    expect(setting.controlEl.textContent).not.toContain("Replace");
    expect(setting.controlEl.textContent).not.toContain("Clear");

    input.value = "sk-secret";
    input.dispatchEvent(new Event("blur"));
    await vi.waitFor(() => {
      expect(settings.llm.apiKey).toBe("sk-secret");
      expect(saveSettings).toHaveBeenCalledTimes(1);
    });
    await vi.waitFor(() => expect(input.disabled).toBe(false));

    reveal.click();
    expect(input.type).toBe("text");
    expect(reveal.getAttribute("aria-label")).toBe("Hide LLM API key");

    input.value = "";
    input.dispatchEvent(new Event("blur"));
    await vi.waitFor(() => expect(settings.llm.apiKey).toBe(""));
    expect(saveSettings).toHaveBeenCalledTimes(2);
  });

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
    await Promise.resolve();
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
    expect(setting.controlEl.textContent).not.toContain("Replace");
    expect(setting.controlEl.textContent).not.toContain("Clear");

    reveal.click();
    expect(input.type).toBe("text");
    input.value = " new token \n value ";
    sendTest.click();

    await vi.waitFor(() => {
      expect(settings.email.hostedToken).toBe("newtokenvalue");
      expect(saveSettings).toHaveBeenCalledTimes(1);
      expect(plugin.refreshSensitiveValues).toHaveBeenCalled();
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
    input.dispatchEvent(new Event("blur"));
    await vi.waitFor(() => {
      expect(settings.email.hostedToken).toBe("old-token");
      expect(input.value).toBe("old-token");
      expect(plugin.refreshSensitiveValues).toHaveBeenCalledTimes(2);
    });
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
    });
    const tabItems = tab.getSettingDefinitions();
    expect(tabItems.length).toBe(hostItems.length);
    const keyOf = (item: unknown): string =>
      (item as { name?: string }).name ?? "";
    expect(tabItems.map(keyOf)).toEqual(hostItems.map(keyOf));
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
});
