import {
  Notice,
  setIcon,
  ToggleComponent,
  type Setting,
} from "obsidian";
import type { ArxivDailySettingTab } from "./tab";
import {
  API_KEY_CONFIGURED_SENTINEL,
  addCategoryOptions,
  llmHttpWarning,
  modelFetchNoticeMessage,
  persistApiKeyChange,
  renderRunWindowTimeSelect,
  TIMEZONE_OPTIONS,
} from "./tab";
import { arxivCategories, LlmClient } from "@arxiv-daily/core";

/**
 * Prepare a declarative row for (re)rendering. Obsidian reuses the same
 * Setting row and calls the render callback again on update(), so the
 * previous render's content must be cleared to keep re-renders idempotent.
 */
function prepareRow(setting: Setting): void {
  setting.controlEl.empty();
}

/** Remove previously rendered siblings from a row's main element. */
function clearSettingEl(setting: Setting, ...classes: string[]): void {
  for (const cls of classes) {
    for (const el of Array.from(setting.settingEl.querySelectorAll(`.${cls}`))) {
      el.remove();
    }
  }
}

/**
 * Imperative row renderers for the Obsidian 1.13+ declarative settings API.
 * Lives in its own module (not definitions.ts) so the shared sentinel
 * state machines can reach tab internals via the tab instance.
 */
export function renderLlmBaseUrlRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  setting.controlEl.addClass("arxiv-daily-settings__llm-url-control");
  const input = setting.controlEl.createEl("input", {
    cls: "arxiv-daily-settings__llm-url-input",
    type: "url",
    attr: { placeholder: "https://api.deepseek.com/v1" },
  });
  input.value = tab.plugin.settings.llm.baseUrl;
  const warningEl = setting.controlEl.createDiv({
    cls: "arxiv-daily-settings__llm-inline-warning",
  });
  const refreshWarning = () => {
    const warning = llmHttpWarning(input.value);
    warningEl.empty();
    warningEl.toggleClass("is-visible", Boolean(warning));
    if (warning) warningEl.setText(warning.message);
  };
  refreshWarning();
  input.addEventListener("input", refreshWarning);
  input.addEventListener("change", () => {
    tab.runAction("save API base URL", async () => {
      tab.plugin.settings.llm.baseUrl = input.value.trim();
      await tab.plugin.saveSettings();
      tab.refreshDeclarativeSetupGuide();
    });
  });
}

export function renderApiKeyRow(tab: ArxivDailySettingTab, setting: Setting): void {
  prepareRow(setting);
  const input = setting.controlEl.createEl("input", {
    cls: "arxiv-daily-settings__llm-input",
    type: "password",
    attr: {
      placeholder: "Enter API key",
      autocomplete: "off",
      "aria-label": "LLM API key",
    },
  });
  input.value = tab.plugin.settings.llm.apiKey;

  const reveal = setting.controlEl.createEl("button", {
    cls: "arxiv-daily-settings__reveal-key",
    attr: {
      type: "button",
      "aria-label": "Show API key",
      title: "Show API key",
    },
  });
  setIcon(reveal, "eye");

  const setRevealed = (revealed: boolean) => {
    input.type = revealed ? "text" : "password";
    reveal.setAttribute("aria-label", revealed ? "Hide API key" : "Show API key");
    reveal.title = revealed ? "Hide API key" : "Show API key";
    reveal.empty();
    setIcon(reveal, revealed ? "eye-off" : "eye");
  };
  reveal.addEventListener("pointerdown", (event) => event.preventDefault());
  reveal.addEventListener("click", () => setRevealed(input.type === "password"));

  let savedValue = input.value;
  let saving = false;
  const save = async () => {
    const next = input.value.trim();
    if (saving || next === savedValue) return;
    saving = true;
    input.disabled = true;
    reveal.disabled = true;
    try {
      await persistApiKeyChange(
        tab.plugin.settings,
        tab.plugin.logger,
        () => tab.plugin.saveSettings(),
        next,
      );
      savedValue = next;
      input.value = next;
      tab.refreshDeclarativeSetupGuide();
    } catch (error) {
      input.value = savedValue;
      throw error;
    } finally {
      saving = false;
      input.disabled = false;
      reveal.disabled = false;
    }
  };
  input.addEventListener("blur", () => tab.runAction("save API key", save));
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    input.blur();
  });
}

export function renderReasoningEffortRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const select = setting.controlEl.createEl("select");
  const options = [
    ["none", "None"],
    ["low", "Low"],
    ["medium", "Medium"],
    ["high", "High"],
  ] as const;
  for (const [value, label] of options) {
    select.createEl("option", { value, text: label });
  }
  select.value = tab.plugin.settings.llm.thinkingMode
    ? tab.plugin.settings.llm.reasoningEffort
    : "none";
  if (!options.some(([value]) => value === select.value)) select.value = "medium";
  select.addEventListener("change", () => {
    tab.runAction("save reasoning effort", async () => {
      if (select.value === "none") {
        tab.plugin.settings.llm.thinkingMode = false;
      } else {
        tab.plugin.settings.llm.thinkingMode = true;
        tab.plugin.settings.llm.reasoningEffort = select.value;
      }
      await tab.plugin.saveSettings();
    });
  });
}

export function renderModelRow(tab: ArxivDailySettingTab, setting: Setting): void {
  prepareRow(setting);
  const select = setting.controlEl.createEl("select", {
    cls: "arxiv-daily-settings__model-select",
  });
  const current = tab.plugin.settings.llm.model;
  if (current) select.createEl("option", { value: current, text: current });
  select.value = current;
  select.addEventListener("change", () => {
    void (async () => {
      tab.plugin.settings.llm.model = select.value;
      await tab.plugin.saveSettings();
      tab.refreshDeclarativeSetupGuide();
    })();
  });

  const button = setting.controlEl.createEl("button", {
    text: "Get models",
    attr: { type: "button" },
  });
  button.addEventListener("click", () => {
    void (async () => {
    button.textContent = "Fetching…";
    button.disabled = true;
    try {
      const client = new LlmClient(
        tab.plugin.settings.llm,
        tab.plugin.logger,
        tab.plugin.getHttpClient(),
      );
      const models = await client.fetchModels();
      if (models.length > 0) {
        tab.showModelDropdown(models, setting.settingEl);
        new Notice(modelFetchNoticeMessage({ kind: "success", count: models.length }));
      } else {
        new Notice(modelFetchNoticeMessage({ kind: "empty" }));
      }
    } catch (e) {
      new Notice(modelFetchNoticeMessage(
        { kind: "error", message: e instanceof Error ? e.message : String(e) },
        [tab.plugin.settings.llm.apiKey],
      ));
    } finally {
      button.textContent = "Get models";
      button.disabled = false;
    }
    })();
  });
}

export function renderSetupGuideRow(tab: ArxivDailySettingTab, setting: Setting): void {
  clearSettingEl(setting, "arxiv-daily-setup");
  const guide = tab.createSetupGuide();
  if (guide) setting.settingEl.appendChild(guide);
}

/** One arXiv category: known-category dropdown + free-text override. */
export function renderCategoryRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
  index: number,
): void {
  prepareRow(setting);
  const categories = arxivCategories(tab.plugin.settings.arxiv);
  const current = categories[index] ?? "";
  const select = setting.controlEl.createEl("select", {
    cls: "arxiv-daily-settings__category-select",
  });
  addCategoryOptions(select, current);
  select.value = current;
  select.addEventListener("change", () => {
    const next = [...categories];
    next[index] = select.value;
    void tab.runAction("save category", async () => {
      await tab.setArxivCategories(next);
      tab.refreshSettings();
    });
  });
}

/** One research topic: the shared expandable topic card. */
export function renderTopicRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
  index: number,
): void {
  tab.renderTopicRow(setting, index);
}

/** Timezone picker: preset dropdown + free-text override. */
export function renderTimezoneRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const select = setting.controlEl.createEl("select");
  for (const zone of TIMEZONE_OPTIONS) {
    const option = select.createEl("option");
    option.value = zone.value;
    option.textContent = zone.label;
  }
  select.value = tab.plugin.settings.arxiv.timezone;
  select.addEventListener("change", () => {
    tab.plugin.settings.arxiv.timezone = select.value;
    void tab.plugin.saveSettings();
  });
  const input = setting.controlEl.createEl("input", {
    type: "text",
    placeholder: "Or enter custom timezone",
  });
  input.addEventListener("input", () => {
    if (input.value.trim()) {
      tab.plugin.settings.arxiv.timezone = input.value.trim();
      void tab.plugin.saveSettings();
    }
  });
}

/** Scheduler enable toggle; routes through setScheduleEnabled (validation + modal). */
export function renderScheduleEnabledRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  new ToggleComponent(setting.controlEl)
    .setValue(tab.plugin.settings.schedule.enabled)
    .onChange(async (value) => {
      await tab.plugin.setScheduleEnabled(value);
      tab.refreshSettings();
    });
}

/** Run window: Start/End local-time selects (24-hour clock). */
export function renderRunWindowRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const schedule = tab.plugin.settings.schedule;
  renderRunWindowTimeSelect(
    setting.controlEl,
    "Start",
    "arxiv-daily-run-window-start",
    schedule.runAtLocal,
    (value) => tab.saveRunWindowTime("runAtLocal", value),
  );
  renderRunWindowTimeSelect(
    setting.controlEl,
    "End",
    "arxiv-daily-run-window-end",
    schedule.runUntilLocal,
    (value) => tab.saveRunWindowTime("runUntilLocal", value),
  );
}

/** Check-every (minutes) input; mirrors display()'s sanitize + scheduler restart. */
export function renderTickIntervalRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const input = setting.controlEl.createEl("input", {
    type: "text",
    cls: "arxiv-daily-settings__tick-input",
  });
  input.value = String(tab.plugin.settings.schedule.tickIntervalMin);
  input.addEventListener("input", () => {
    tab.plugin.settings.schedule.tickIntervalMin =
      Math.max(1, Number(input.value) || 20);
    void tab.plugin.saveSettings().then(() => tab.plugin.restartScheduler());
  });
}

/** Email delivery guide strip for the current mode. */
export function renderEmailGuideRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  clearSettingEl(setting, "arxiv-daily-settings__email-guide");
  const { title, lines } = tab.emailGuideContent();
  const wrap = setting.settingEl.createDiv({
    cls: "arxiv-daily-settings__email-guide",
  });
  wrap.createDiv({
    cls: "arxiv-daily-settings__email-guide-title",
    text: title,
  });
  for (const line of lines) {
    wrap.createDiv({
      cls: "arxiv-daily-settings__email-guide-line",
      text: line,
    });
  }
}

/** Email mode dropdown (Send yourself / Official delivery). */
export function renderEmailModeRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const select = setting.controlEl.createEl("select");
  const selfOption = select.createEl("option");
  selfOption.value = "self";
  selfOption.textContent = "Send yourself";
  const hostedOption = select.createEl("option");
  hostedOption.value = "hosted";
  hostedOption.textContent = "Official delivery (Beta)";
  select.value = tab.plugin.settings.email.mode === "hosted" ? "hosted" : "self";
  select.addEventListener("change", () => {
    tab.plugin.settings.email.mode =
      select.value === "hosted" ? "hosted" : "self";
    void tab.runAction("save email mode", async () => {
      await tab.plugin.saveSettings();
      tab.refreshSettings();
    });
  });
}

/** Resend API key sentinel row (self mode). */
export function renderEmailApiKeyRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const configured = Boolean(tab.plugin.settings.email.apiKey?.trim());
  let editing = !configured;
  let draft = "";
  const input = setting.controlEl.createEl("input", {
    cls: "arxiv-daily-settings__llm-input",
    type: editing ? "password" : "text",
    attr: { placeholder: "Paste your Resend API key" },
  });
  input.value = configured ? API_KEY_CONFIGURED_SENTINEL : "";
  input.readOnly = !editing;

  const replace = setting.controlEl.createEl("button", {
    text: configured ? "Replace" : "Save",
    attr: { type: "button" },
  });
  const cancel = setting.controlEl.createEl("button", {
    text: "Cancel",
    attr: { type: "button" },
  });
  cancel.hidden = !configured;
  const clear = setting.controlEl.createEl("button", {
    text: "Clear",
    attr: { type: "button" },
  });
  clear.hidden = !configured;

  const enterEdit = () => {
    editing = true;
    draft = "";
    input.type = "password";
    input.readOnly = false;
    input.value = "";
    replace.textContent = "Save";
    cancel.hidden = false;
    input.focus();
  };
  const reset = () => {
    editing = false;
    draft = "";
    input.type = "text";
    input.readOnly = true;
    input.value = API_KEY_CONFIGURED_SENTINEL;
    replace.textContent = "Replace";
    cancel.hidden = true;
  };
  input.addEventListener("input", () => {
    if (editing) draft = input.value;
  });
  replace.addEventListener("click", () => {
    if (!editing) {
      enterEdit();
      return;
    }
    const next = draft.trim();
    if (!next) return;
    tab.runAction("save Resend API key", async () => {
      tab.plugin.settings.email.apiKey = next;
      await tab.plugin.saveSettings();
      reset();
      clear.hidden = false;
    });
  });
  cancel.addEventListener("click", () => {
    if (configured || tab.plugin.settings.email.apiKey?.trim()) reset();
    else {
      draft = "";
      input.value = "";
    }
  });
  clear.addEventListener("click", () => {
    tab.runAction("clear Resend API key", async () => {
      const confirmed = await tab.confirmReplace(
        "Clear the saved Resend API key? Email delivery will stop until a replacement is saved.",
        "Clear",
      );
      if (!confirmed) return;
      tab.plugin.settings.email.apiKey = "";
      await tab.plugin.saveSettings();
      tab.refreshSettings();
    });
  });
}

/** Verification-code sentinel row (hosted mode). */
export function renderHostedTokenRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const configured = Boolean(tab.plugin.settings.email.hostedToken?.trim());
  let editing = !configured;
  let draft = "";
  const input = setting.controlEl.createEl("input", {
    cls: "arxiv-daily-settings__llm-input",
    type: editing ? "password" : "text",
    attr: { placeholder: "Paste the code from the verification page" },
  });
  input.value = configured ? API_KEY_CONFIGURED_SENTINEL : "";
  input.readOnly = !editing;

  const replace = setting.controlEl.createEl("button", {
    text: configured ? "Replace" : "Save",
    attr: { type: "button" },
  });
  const cancel = setting.controlEl.createEl("button", {
    text: "Cancel",
    attr: { type: "button" },
  });
  cancel.hidden = !configured;
  const clear = setting.controlEl.createEl("button", {
    text: "Clear",
    attr: { type: "button" },
  });
  clear.hidden = !configured;

  const enterEdit = () => {
    editing = true;
    draft = "";
    input.type = "password";
    input.readOnly = false;
    input.value = "";
    replace.textContent = "Save";
    cancel.hidden = false;
    input.focus();
  };
  const reset = () => {
    editing = false;
    draft = "";
    input.type = "text";
    input.readOnly = true;
    input.value = API_KEY_CONFIGURED_SENTINEL;
    replace.textContent = "Replace";
    cancel.hidden = true;
  };
  input.addEventListener("input", () => {
    if (editing) draft = input.value;
  });
  replace.addEventListener("click", () => {
    if (!editing) {
      enterEdit();
      return;
    }
    const next = draft.replace(/\s+/g, "").trim();
    if (!next) {
      new Notice("Paste the verification code before saving.");
      return;
    }
    tab.runAction("save verification code", async () => {
      tab.plugin.settings.email.hostedToken = next;
      await tab.plugin.saveSettings();
      tab.plugin.refreshSensitiveValues();
      tab.refreshSettings();
    });
  });
  cancel.addEventListener("click", () => {
    if (configured || tab.plugin.settings.email.hostedToken?.trim()) reset();
    else {
      draft = "";
      input.value = "";
    }
  });
  clear.addEventListener("click", () => {
    tab.runAction("clear verification code", async () => {
      const confirmed = await tab.confirmReplace(
        "Clear the verification code? You will need to verify your email again for Official delivery.",
        "Clear",
      );
      if (!confirmed) return;
      tab.plugin.settings.email.hostedToken = "";
      await tab.plugin.saveSettings();
      tab.plugin.refreshSensitiveValues();
      tab.refreshSettings();
    });
  });
}

/** Send the hosted-mode verification email (action row). */
export function sendVerificationEmail(tab: ArxivDailySettingTab): void {
  tab.runAction("send verification email", async () => {
    const message = await tab.plugin.sendHostedVerificationEmail();
    new Notice(message, 10_000);
  });
}

/** Send a sample digest now (action row). */
export function sendTestEmail(tab: ArxivDailySettingTab): void {
  tab.runAction("send test email", async () => {
    const message = await tab.plugin.sendTestEmail();
    new Notice(message, 10_000);
  });
}
