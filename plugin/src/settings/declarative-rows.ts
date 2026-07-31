import { Notice, type Setting } from "obsidian";
import type { ArxivDailySettingTab } from "./tab";
import {
  API_KEY_CONFIGURED_SENTINEL,
  modelFetchNoticeMessage,
  persistApiKeyChange,
} from "./tab";
import { LlmClient } from "@arxiv-daily/core";

/**
 * Imperative row renderers for the Obsidian 1.13+ declarative settings API.
 * Lives in its own module (not definitions.ts) so the shared sentinel
 * state machines can reach tab internals via the tab instance.
 */
export function renderApiKeyRow(tab: ArxivDailySettingTab, setting: Setting): void {
  const configured = Boolean(tab.plugin.settings.llm.apiKey.trim());
  let editing = !configured;
  let draft = "";
  const input = setting.controlEl.createEl("input", {
    cls: "arxiv-daily-settings__llm-input",
    type: editing ? "password" : "text",
    attr: { placeholder: "Enter API key" },
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
    tab.runAction("save API key", async () => {
      await persistApiKeyChange(
        tab.plugin.settings,
        tab.plugin.logger,
        () => tab.plugin.saveSettings(),
        next,
      );
      tab.refreshSetupGuide();
      reset();
      clear.hidden = false;
    });
  });
  cancel.addEventListener("click", () => {
    if (configured || tab.plugin.settings.llm.apiKey.trim()) reset();
    else {
      draft = "";
      input.value = "";
    }
  });
  clear.addEventListener("click", () => {
    tab.runAction("clear API key", async () => {
      const confirmed = await tab.confirmReplace(
        "Clear the saved API key? AI features will stop until you save a new key.",
        "Clear",
      );
      if (!confirmed) return;
      await persistApiKeyChange(
        tab.plugin.settings,
        tab.plugin.logger,
        () => tab.plugin.saveSettings(),
        "",
      );
      tab.refreshSetupGuide();
    });
  });
}

export function renderModelRow(tab: ArxivDailySettingTab, setting: Setting): void {
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
      tab.refreshSetupGuide();
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
  const guide = tab.createSetupGuide();
  if (guide) setting.settingEl.appendChild(guide);
}
