import { beforeAll, describe, expect, it, vi } from "vitest";
import { Setting, ToggleComponent, type App } from "obsidian";
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
  renderEmailModeRow,
  renderEmailToRow,
  renderHostedTokenRow,
  renderLlmBaseUrlRow,
  renderModelRow,
  renderReasoningEffortRow,
  renderRunWindowRow,
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

/** Tab with a mocked plugin whose settings persist through saveSettings. */
function makeTab() {
  const settings = structuredClone(DEFAULT_SETTINGS);
  const saveSettings = vi.fn((_candidate?: unknown) => Promise.resolve());
  const plugin = {
    settings,
    saveSettings,
    manifest: { version: "0.0.0-test" },
    app: {},
    stateStore: { snapshot: () => ({}) },
    logger: {
      setSensitiveValues: vi.fn(),
      setLevel: vi.fn(),
      setTimezone: vi.fn(),
      error: vi.fn(),
    },
    refreshSensitiveValues: vi.fn(),
    restartScheduler: vi.fn(),
    sendHostedVerificationEmail: vi.fn().mockResolvedValue("Verification sent"),
    sendTestEmail: vi.fn().mockResolvedValue("Test sent"),
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
    settings.arxiv.topics.push({
      id: "topic-1",
      name: "Language models",
      tag: "language-models",
      description: "Research about language models",
      detail: false,
    });
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
  it("keeps topic fields focused while updating the setup guide", async () => {
    const { tab, settings, saveSettings } = makeTab();
    settings.arxiv.topics.push({
      id: "topic-1",
      name: "",
      tag: "topic-1",
      description: "",
      detail: false,
    });
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
        ".arxiv-daily-settings__topic-tag-input",
      ),
      topicSetting.settingEl.querySelector(
        ".arxiv-daily-settings__topic-description",
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
