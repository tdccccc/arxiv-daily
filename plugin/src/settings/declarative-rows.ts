import {
  Notice,
  setIcon,
  ToggleComponent,
  type Setting,
} from "obsidian";
import type { ArxivDailySettingTab } from "./tab";
import { renderSensitiveInput } from "./sensitive-input";
import {
  addCategoryOptions,
  llmHttpWarning,
  modelFetchNoticeMessage,
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
export function renderLibraryConnectionRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  tab.renderLibraryConnectionControls(setting);
}

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
    const next = input.value.trim();
    const revision = tab.beginControlChange(input);
    tab.runAction("save API base URL", async () => {
      try {
        await tab.changeSettingValue("llm.baseUrl", next);
        if (tab.isCurrentControlChange(input, revision)) input.value = next;
        tab.refreshDeclarativeSetupGuide();
      } catch (error) {
        if (tab.isCurrentControlChange(input, revision)) {
          input.value = tab.restoreCurrentStringControlValue(error, "llm.baseUrl");
        }
        throw error;
      }
    });
  });
}

export function renderApiKeyRow(tab: ArxivDailySettingTab, setting: Setting): void {
  prepareRow(setting);
  renderSensitiveInput(tab, setting, {
    value: tab.plugin.settings.llm.apiKey,
    placeholder: "Enter API key",
    ariaLabel: "LLM API key",
    save: (next) => tab.changeSettingValue("llm.apiKey", next),
    onCommitted: () => tab.refreshDeclarativeSetupGuide(),
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
    const next = select.value;
    const revision = tab.beginControlChange(select);
    tab.runAction("save reasoning effort", async () => {
      try {
        await tab.changeSettingValues(
          next === "none"
            ? [{ key: "llm.thinkingMode", value: false }]
            : [
                { key: "llm.thinkingMode", value: true },
                { key: "llm.reasoningEffort", value: next },
              ],
        );
      } catch (error) {
        if (tab.isCurrentControlChange(select, revision)) {
          select.value = tab.plugin.settings.llm.thinkingMode
            ? tab.plugin.settings.llm.reasoningEffort
            : "none";
        }
        throw error;
      }
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
    const next = select.value;
    const revision = tab.beginControlChange(select);
    tab.runAction("save model", async () => {
      try {
        await tab.changeSettingValue("llm.model", next);
        tab.refreshDeclarativeSetupGuide();
      } catch (error) {
        if (tab.isCurrentControlChange(select, revision)) {
          select.value = tab.restoreCurrentStringControlValue(error, "llm.model");
        }
        throw error;
      }
    });
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

/** Timezone picker: preset dropdown + committed free-text draft. */
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
    const next = select.value;
    const revision = tab.beginControlChange(select);
    tab.runAction("save timezone", async () => {
      try {
        await tab.saveTimezone(next);
      } catch (error) {
        if (tab.isCurrentControlChange(select, revision)) {
          select.value = tab.restoreCurrentStringControlValue(error, "arxiv.timezone");
        }
        throw error;
      }
    });
  });
  const input = setting.controlEl.createEl("input", {
    type: "text",
    placeholder: "Or enter custom timezone",
  });
  tab.bindTimezoneDraftInput(input, select);
}

/** Scheduler enable toggle; routes through setScheduleEnabled (validation + modal). */
export function renderScheduleEnabledRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const toggle = new ToggleComponent(setting.controlEl)
    .setValue(tab.plugin.settings.schedule.enabled)
    .onChange(async (value) => {
      const revision = tab.beginControlChange(toggle);
      try {
        const changed = await tab.plugin.setScheduleEnabled(value);
        if (tab.isCurrentControlChange(toggle, revision) && !changed) {
          toggle.setValue(tab.plugin.settings.schedule.enabled);
        }
      } catch (error) {
        if (tab.isCurrentControlChange(toggle, revision)) {
          toggle.setValue(tab.plugin.settings.schedule.enabled);
          tab.reportSettingsActionError("save schedule enabled", error);
        }
      } finally {
        if (tab.isCurrentControlChange(toggle, revision)) tab.refreshSettings();
      }
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
  tab.bindTickIntervalInput(input);
}

/** Email delivery guide strip for the current mode. */
export function renderEmailGuideRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  clearSettingEl(setting, "arxiv-daily-settings__email-guide");
  setting.settingEl.addClass("arxiv-daily-settings__email-guide-host");
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
    const next = select.value === "hosted" ? "hosted" : "self";
    const revision = tab.beginControlChange(select);
    tab.runAction("save email mode", async () => {
      try {
        await tab.changeSettingValue("email.mode", next);
        tab.refreshSettings();
      } catch (error) {
        if (tab.isCurrentControlChange(select, revision)) {
          select.value = tab.restoreCurrentStringControlValue(error, "email.mode", "self");
        }
        throw error;
      }
    });
  });
}

interface EmailSaveState {
  tail: Promise<void>;
  latest?: { value: string; promise: Promise<void> };
}

const emailSaveStates = new WeakMap<ArxivDailySettingTab, EmailSaveState>();

function emailSaveState(tab: ArxivDailySettingTab): EmailSaveState {
  const existing = emailSaveStates.get(tab);
  if (existing) return existing;
  const state = { tail: Promise.resolve() };
  emailSaveStates.set(tab, state);
  return state;
}

async function saveEmailToDraft(
  tab: ArxivDailySettingTab,
  input: HTMLInputElement,
): Promise<void> {
  const next = input.value.trim();
  const state = emailSaveState(tab);
  if (state.latest?.value === next) return state.latest.promise;
  if (next === tab.plugin.settings.email.to) return;
  input.value = next;
  const revision = tab.beginControlChange(input);
  const operation = state.tail.then(async () => {
    if (next === tab.plugin.settings.email.to) return;
    try {
      await tab.changeSettingValue("email.to", next);
      if (tab.isCurrentControlChange(input, revision)) input.value = next;
    } catch (error) {
      if (tab.isCurrentControlChange(input, revision)) {
        input.value = tab.restoreCurrentStringControlValue(error, "email.to");
      }
      throw error;
    }
  });
  let save: Promise<void>;
  save = operation.finally(() => {
    if (state.latest?.promise === save) state.latest = undefined;
  });
  state.latest = { value: next, promise: save };
  state.tail = save.catch(() => undefined);
  await save;
}

async function waitForEmailToSave(tab: ArxivDailySettingTab): Promise<void> {
  await emailSaveStates.get(tab)?.latest?.promise;
}

function renderEmailActionButton(
  tab: ArxivDailySettingTab,
  setting: Setting,
  options: {
    label: string;
    pendingLabel: string;
    action: string;
    preserveFocus?: boolean;
    beforeRun?: () => Promise<void>;
    run: () => Promise<string>;
  },
): void {
  const button = setting.controlEl.createEl("button", {
    text: options.label,
    attr: { type: "button" },
  });
  if (options.preserveFocus) {
    button.addEventListener("pointerdown", (event) => event.preventDefault());
  }
  button.addEventListener("click", () => {
    void (async () => {
      button.disabled = true;
      button.textContent = options.pendingLabel;
      await tab.runActionAndWait(options.action, async () => {
        await options.beforeRun?.();
        const message = await options.run();
        new Notice(message, 10_000);
      });
      button.disabled = false;
      button.textContent = options.label;
    })();
  });
}

export function renderEmailToRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const input = setting.controlEl.createEl("input", {
    type: "email",
    attr: { placeholder: "you@example.com" },
  });
  input.value = tab.plugin.settings.email.to;
  const saveEmail = () => saveEmailToDraft(tab, input);
  input.addEventListener("change", () => {
    tab.runAction("save email address", saveEmail);
  });
  if (tab.plugin.settings.email.mode === "hosted") {
    renderEmailActionButton(tab, setting, {
      label: "Send verification",
      pendingLabel: "Sending…",
      action: "send verification email",
      preserveFocus: true,
      beforeRun: saveEmail,
      run: () => tab.plugin.sendHostedVerificationEmail(),
    });
  }
}

/** Resend API key masked input row (self mode). */
export function renderEmailApiKeyRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  renderSensitiveInput(tab, setting, {
    value: tab.plugin.settings.email.apiKey ?? "",
    placeholder: "Paste your Resend API key",
    ariaLabel: "Resend API key",
    save: (next) => tab.changeSettingValue("email.apiKey", next),
    onCommitted: () => tab.refreshDeclarativeSetupGuide(),
  });
  renderEmailActionButton(tab, setting, {
    label: "Send test",
    pendingLabel: "Sending…",
    action: "send test email",
    beforeRun: () => waitForEmailToSave(tab),
    run: () => tab.plugin.sendTestEmail(),
  });
}

/** Verification-code masked input row (hosted mode). */
export function renderHostedTokenRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const saveToken = renderSensitiveInput(tab, setting, {
    value: tab.plugin.settings.email.hostedToken ?? "",
    placeholder: "Paste the code from the verification page",
    ariaLabel: "verification code",
    normalize: (value) => value.replace(/\s+/g, "").trim(),
    save: (next) => tab.changeSettingValue("email.hostedToken", next),
    onCommitted: () => tab.refreshDeclarativeSetupGuide(),
  });
  renderEmailActionButton(tab, setting, {
    label: "Send test",
    pendingLabel: "Sending…",
    action: "send test email",
    preserveFocus: true,
    beforeRun: async () => {
      await waitForEmailToSave(tab);
      await saveToken();
    },
    run: () => tab.plugin.sendTestEmail(),
  });
}

export function renderEmbeddingModeRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const select = setting.controlEl.createEl("select");
  const local = select.createEl("option", { text: "Local (offline, default)" });
  local.value = "local";
  const remote = select.createEl("option", { text: "Remote (fast, full text leaves this device)" });
  remote.value = "remote";
  select.value = tab.plugin.settings.embedding.mode;
  select.addEventListener("change", () => {
    tab.runAction("save embedding mode", async () => {
      tab.plugin.settings.embedding.mode = select.value === "remote" ? "remote" : "local";
      await tab.plugin.saveSettings();
    });
  });
}

export function renderEmbeddingBaseUrlRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const input = setting.controlEl.createEl("input", {
    type: "url",
    attr: { placeholder: "https://api.openai.com/v1" },
  });
  input.value = tab.plugin.settings.embedding.baseUrl;
  input.addEventListener("change", () => {
    tab.runAction("save embedding base url", async () => {
      tab.plugin.settings.embedding.baseUrl = input.value.trim();
      await tab.plugin.saveSettings();
    });
  });
}

export function renderEmbeddingApiKeyRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  renderSensitiveInput(tab, setting, {
    value: tab.plugin.settings.embedding.apiKey,
    placeholder: "Enter API key",
    ariaLabel: "Embedding API key",
    save: (next) => tab.changeSettingValue("embedding.apiKey", next),
  });
}

export function renderEmbeddingModelRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const input = setting.controlEl.createEl("input", {
    attr: { placeholder: "text-embedding-3-small" },
  });
  input.value = tab.plugin.settings.embedding.model;
  input.addEventListener("change", () => {
    tab.runAction("save embedding model", async () => {
      tab.plugin.settings.embedding.model = input.value.trim();
      await tab.plugin.saveSettings();
    });
  });
}

export function renderEmbeddingDimensionRow(
  tab: ArxivDailySettingTab,
  setting: Setting,
): void {
  prepareRow(setting);
  const input = setting.controlEl.createEl("input", {
    type: "number",
    attr: { placeholder: "1536", min: "1", step: "1" },
  });
  input.value = String(tab.plugin.settings.embedding.dimension);
  input.addEventListener("change", () => {
    tab.runAction("save embedding dimension", async () => {
      const parsed = Number(input.value.trim());
      if (Number.isInteger(parsed) && parsed > 0) {
        tab.plugin.settings.embedding.dimension = parsed;
        await tab.plugin.saveSettings();
      }
    });
  });
}
