import { App, Modal, Notice, PluginSettingTab, Setting, setTooltip } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { PROVIDER_PRESETS, type ProviderPreset } from "@arxiv-daily/core";
import { ARXIV_CATEGORIES } from "@arxiv-daily/core";
import { TOPIC_TEMPLATES } from "@arxiv-daily/core";
import type { Topic } from "@arxiv-daily/core";
import { slugify } from "@arxiv-daily/core";
import {
  validateSchedulerConfig,
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "@arxiv-daily/core";
import { arxivCategories } from "@arxiv-daily/core";
import { getSetupStatus, shouldRenderSetupGuide } from "../onboarding";
import { executeObsidianCommand, openDashboardView } from "../dashboard/view";
import { LlmClient, redactText } from "@arxiv-daily/core";

export const API_KEY_CONFIGURED_SENTINEL = "Configured";

export function validateOutputDirectoryDraft(
  draft: string,
  siblingDirectory?: string,
) {
  const validation = validateVaultRelativeDirectory(draft);
  if (
    validation.ok &&
    validation.value &&
    siblingDirectory &&
    vaultRelativeDirectoriesCollide(validation.value, siblingDirectory)
  ) {
    return {
      ok: false as const,
      reason: "Daily and papers directories must be different",
    };
  }
  return validation;
}

export type ModelFetchNotice =
  | { kind: "success"; count: number }
  | { kind: "empty" }
  | { kind: "error"; message: string };

export function modelFetchNoticeMessage(
  result: ModelFetchNotice,
  secrets: readonly string[] = [],
): string {
  switch (result.kind) {
    case "success":
      return `API connection successful. Found ${result.count} models.`;
    case "empty":
      return "API connection successful, but no available models were found.";
    case "error":
      return `API connection failed: ${redactText(result.message, { secrets })}`;
  }
}

export type LlmHttpWarning =
  | { kind: "plaintext"; message: string }
  | { kind: "local"; message: string };

export function llmHttpWarning(baseUrl: string): LlmHttpWarning | null {
  let url: URL;
  try {
    url = new URL(baseUrl.trim());
  } catch {
    return null;
  }
  if (url.protocol !== "http:") return null;
  if (isLoopbackHost(url.hostname)) {
    return {
      kind: "local",
      message: "Using a local HTTP LLM endpoint; ensure this is intentional.",
    };
  }
  return {
    kind: "plaintext",
    message: "Your LLM endpoint uses HTTP; API keys will be sent in plaintext.",
  };
}

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return host === "localhost" || host === "::1" || /^127(?:\.\d{1,3}){3}$/.test(host);
}

export class ArxivDailySettingTab extends PluginSettingTab {
  private expandedTopics = new Set<string>();

  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app, plugin);
  }

  /** Append a circled "?" to a Setting's name with an Obsidian-styled tooltip. */
  private attachHelp(setting: Setting, text: string): Setting {
    const q = setting.nameEl.createEl("span", {
      cls: "arxiv-daily-settings__help",
      text: "?",
    });
    setTooltip(q, text, { placement: "top" });
    return setting;
  }

  /** Inline muted hint, used inside topic cards under a label. */
  private hint(parent: HTMLElement, text: string, id?: string): HTMLElement {
    const hint = parent.createEl("div", {
      cls: "arxiv-daily-settings__hint",
      text,
    });
    if (id) hint.id = id;
    return hint;
  }

  private sectionHeading(
    containerEl: HTMLElement,
    name: string,
    section: "llm" | "arxiv" | "topics" | "schedule" | "advanced",
    desc?: string,
  ): Setting {
    const heading = new Setting(containerEl).setName(name).setHeading();
    if (desc) heading.setDesc(desc);
    heading.settingEl.addClass("arxiv-daily-settings__section");
    heading.settingEl.setAttribute("data-arxiv-daily-section", section);
    return heading;
  }

  private async setArxivCategories(categories: string[]): Promise<void> {
    const normalized = normalizeUniqueCategories(categories);
    this.plugin.settings.arxiv.categories = normalized;
    if (normalized[0]) this.plugin.settings.arxiv.category = normalized[0];
    await this.plugin.saveSettings();
  }

  private async applyOutputDirectoryDraft(
    key: "dailyDir" | "papersDir",
    draft: string,
    input: HTMLInputElement,
  ): Promise<void> {
    const siblingKey = key === "dailyDir" ? "papersDir" : "dailyDir";
    const validation = validateOutputDirectoryDraft(
      draft,
      this.plugin.settings.output[siblingKey],
    );
    input.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
    input.toggleClass("is-invalid", !validation.ok);
    if (!validation.ok || !validation.value) return;

    const previous = this.plugin.settings.output[key];
    if (validation.value === previous) return;
    this.plugin.settings.output[key] = validation.value;
    try {
      await this.plugin.reloadStateStoreForOutputPaths();
      await this.plugin.saveSettings();
      input.value = validation.value;
    } catch (error) {
      this.plugin.settings.output[key] = previous;
      input.value = previous;
      input.setCustomValidity("");
      input.removeClass("is-invalid");
      try {
        await this.plugin.reloadStateStoreForOutputPaths();
      } catch (rollbackError) {
        this.plugin.logger.error("settings: failed to restore output stores after rollback", rollbackError);
      }
      this.plugin.logger.error(`settings: rejected ${key} after store reload failed`, error);
      new Notice(`arXiv Daily: output path was not changed: ${(error as Error).message}`, 10_000);
    }
  }

  display(): void {
    const { containerEl } = this;
    const s = this.plugin.settings;
    containerEl.empty();
    containerEl.addClass("arxiv-daily-settings");

    this.renderSetupGuide(containerEl);

    // ─── Config-invalid banner (top) ─────────────────
    const v = validateSchedulerConfig(s);
    if (!v.ok) {
      const banner = containerEl.createDiv({
        cls: "arxiv-daily-settings__invalid-banner",
      });
      banner.createEl("strong", { text: "Configuration incomplete" });
      const ul = banner.createEl("ul", {
        cls: "arxiv-daily-settings__invalid-list",
      });
      for (const r of v.reasons) ul.createEl("li", { text: r });
    }

    // ─── Enable toggle (top) ─────────────────────────
    new Setting(containerEl)
      .setName(`Enable · ${s.schedule.enabled ? "Running" : "Paused"}`)
      .setDesc("Auto-summarize arXiv papers on schedule (skips weekends)")
      .addToggle((t) =>
        t.setValue(s.schedule.enabled).onChange(async (v) => {
          await this.plugin.setScheduleEnabled(v);
          this.display();
        }),
      );

    // ─── LLM ──────────────────────────────────────────
    this.sectionHeading(containerEl, "LLM", "llm");

    // Base URL — always editable, default to DeepSeek
    new Setting(containerEl)
      .setName("Base URL")
      .setDesc("LLM endpoint base. Default is DeepSeek. Override for other providers.")
      .addText((t) => {
        t.inputEl.addClass("arxiv-daily-settings__llm-input");
        t.setPlaceholder("https://api.deepseek.com/v1")
          .setValue(s.llm.baseUrl || "https://api.deepseek.com/v1")
          .onChange(async (v) => {
            s.llm.baseUrl = v;
            await this.plugin.saveSettings();
            renderLlmHttpWarning(v);
            this.refreshSetupGuide();
          });
      });
    const llmWarningEl = containerEl.createDiv({
      cls: "arxiv-daily-settings__llm-http-warning",
    });
    const renderLlmHttpWarning = (baseUrl: string) => {
      const warning = llmHttpWarning(baseUrl);
      llmWarningEl.empty();
      llmWarningEl.toggleClass("is-visible", Boolean(warning));
      if (warning) llmWarningEl.setText(warning.message);
    };
    renderLlmHttpWarning(s.llm.baseUrl || "https://api.deepseek.com/v1");

    this.renderApiKeySetting(containerEl);

    // Model — Get Models button + dropdown
    const modelSetting = new Setting(containerEl)
      .setName("Model")
      .setDesc("Click Get models to fetch available models from API.");

    // Get models button
    const fetchModelsButton = modelSetting.addButton((b) => {
      b.setButtonText("Get models");
      b.onClick(async () => {
        b.setButtonText("Fetching…");
        b.setDisabled(true);
        try {
          const client = new LlmClient(this.plugin.settings.llm, this.plugin.logger, this.plugin.getHttpClient());
          const models = await client.fetchModels();
          if (models.length > 0) {
            this.showModelDropdown(models, modelSetting.settingEl);
            new Notice(modelFetchNoticeMessage({ kind: "success", count: models.length }));
          } else {
            new Notice(modelFetchNoticeMessage({ kind: "empty" }));
          }
        } catch (e) {
          new Notice(modelFetchNoticeMessage(
            { kind: "error", message: e instanceof Error ? e.message : String(e) },
            [this.plugin.settings.llm.apiKey],
          ));
        } finally {
          b.setButtonText("Get models");
          b.setDisabled(false);
        }
      });
    });

    // Model dropdown (empty by default, populated by Get models)
    modelSetting.addDropdown((d) => {
      d.selectEl.addClass("arxiv-daily-settings__model-select");
      if (s.llm.model) {
        d.addOption(s.llm.model, s.llm.model);
      }
      d.setValue(s.llm.model).onChange(async (v) => {
        s.llm.model = v;
        await this.plugin.saveSettings();
        this.refreshSetupGuide();
      });
    });

    // Thinking mode — desc varies by provider
    const thinkingDesc = s.llm.provider === "anthropic"
      ? "Enable Anthropic Extended Thinking"
      : s.llm.provider === "deepseek"
        ? "Enable reasoning mode (DeepSeek V4)"
        : "Enable reasoning/thinking mode";

    new Setting(containerEl)
      .setName("Thinking mode")
      .setDesc(thinkingDesc)
      .addToggle((t) =>
        t.setValue(s.llm.thinkingMode).onChange(async (v) => {
          s.llm.thinkingMode = v;
          await this.plugin.saveSettings();
        }),
      );

    // Reasoning effort — provider-specific options + custom input
    const efforts = ["low", "medium", "high"];
    new Setting(containerEl)
      .setName("Reasoning effort")
      .setDesc(s.llm.provider === "anthropic" ? "Maps to thinking budget tier" : "Reasoning strength")
      .addDropdown((d) => {
        for (const e of efforts) {
          d.addOption(e, e);
        }
        d.setValue(efforts.includes(s.llm.reasoningEffort) ? s.llm.reasoningEffort : efforts[0]!)
          .onChange(async (v) => {
            s.llm.reasoningEffort = v;
            await this.plugin.saveSettings();
          });
      })
      .addText((t) => {
        t.setPlaceholder("or enter custom value")
          .setValue("")
          .onChange(async (v) => {
            if (v.trim()) {
              s.llm.reasoningEffort = v.trim();
              await this.plugin.saveSettings();
            }
          });
      });

    // ─── arXiv ────────────────────────────────────────
    this.sectionHeading(containerEl, "arXiv", "arxiv");

    const categories = arxivCategories(s.arxiv);
    new Setting(containerEl)
      .setName("arXiv Categories")
      .setDesc("Fetch one or more arXiv categories; duplicate papers are merged by arXiv ID.")
      .setHeading();

    for (let i = 0; i < categories.length; i++) {
      const category = categories[i];
      if (!category) continue;
      new Setting(containerEl)
        .setName(`Category ${i + 1}`)
        .addDropdown((d) => {
          addCategoryOptions(d.selectEl, category);
          d.setValue(category).onChange(async (v) => {
            const next = [...categories];
            next[i] = v;
            await this.setArxivCategories(next);
            this.display();
          });
        })
        .addText((t) => {
          t.setPlaceholder("or enter custom category")
            .setValue("")
            .onChange(async (v) => {
              if (v.trim()) {
                const next = [...categories];
                next[i] = v.trim();
                await this.setArxivCategories(next);
                this.display();
              }
            });
        })
        .addButton((b) =>
          b
            .setButtonText("Remove")
            .setDisabled(categories.length === 1)
            .onClick(async () => {
              if (categories.length === 1) return;
              await this.setArxivCategories(categories.filter((_, j) => j !== i));
              this.display();
            }),
        );
    }

    new Setting(containerEl).addButton((b) =>
      b.setButtonText("+ Add Category").onClick(async () => {
        await this.setArxivCategories([
          ...categories,
          nextCategoryCandidate(categories),
        ]);
        this.display();
      }),
    );

    // ─── Research Topics ─────────────────────────────
    this.sectionHeading(
      containerEl,
      "Research Topics",
      "topics",
      "Each topic becomes one section in the daily report.",
    );

    new Setting(containerEl)
      .setName("Quick start")
      .setDesc("Load a preset bundle of topics or add one manually.")
      .addDropdown((d) => {
        d.addOption("", "Load template…");
        for (const tpl of TOPIC_TEMPLATES) {
          d.addOption(tpl.id, tpl.name);
        }
        d.onChange(async (id) => {
          if (!id) return;
          d.setValue("");
          const tpl = TOPIC_TEMPLATES.find((t) => t.id === id);
          if (!tpl) return;
          const apply = async () => {
            s.arxiv.category = tpl.category;
            s.arxiv.categories = [tpl.category];
            s.arxiv.topics = tpl.topics.map((t) => ({ ...t, id: crypto.randomUUID() }));
            await this.plugin.saveSettings();
            this.display();
          };
          const replacesCategories = categoriesWillChange(categories, [tpl.category]);
          if (s.arxiv.topics.length === 0 && !replacesCategories) {
            await apply();
            return;
          }
          const confirmed = await this.confirmReplace(
            quickStartTemplateConfirmMessage(
              s.arxiv.topics.length,
              tpl.name,
              replacesCategories,
            ),
          );
          if (confirmed) await apply();
        });
      })
      .addButton((b) => {
        b.setButtonText("+ Add topic").onClick(async () => {
          const newId = crypto.randomUUID();
          s.arxiv.topics.push({
            id: newId,
            name: "",
            tag: `topic-${s.arxiv.topics.length + 1}`,
            description: "",
            detail: false,
          });
          this.expandedTopics.add(newId);
          await this.plugin.saveSettings();
          this.display();
        });
      });

    const topicsContainer = containerEl.createDiv();
    if (s.arxiv.topics.length === 0) {
      const empty = topicsContainer.createDiv({
        cls: "arxiv-daily-settings__empty-topics",
      });
      empty.createEl("strong", { text: "No topics yet." });
      empty.createEl("div", {
        text: "Pick a template above or click + Add topic to define what to track. The plugin will not call the LLM until at least one topic exists.",
      });
    }
    for (let i = 0; i < s.arxiv.topics.length; i++) {
      this.renderTopicCard(topicsContainer, s.arxiv.topics, i);
    }

    new Setting(containerEl)
      .setName("Timezone")
      .addDropdown((d) => {
        const zones = [
          { v: "Asia/Shanghai", l: "Shanghai (UTC+8)" },
          { v: "Asia/Tokyo", l: "Tokyo (UTC+9)" },
          { v: "US/Eastern", l: "US East (UTC-5)" },
          { v: "US/Pacific", l: "US West (UTC-8)" },
          { v: "Europe/London", l: "London (UTC+0)" },
          { v: "Europe/Berlin", l: "Berlin (UTC+1)" },
          { v: "Europe/Moscow", l: "Moscow (UTC+3)" },
          { v: "Australia/Sydney", l: "Sydney (UTC+10)" },
          { v: "UTC", l: "UTC" },
        ];
        for (const z of zones) {
          d.addOption(z.v, z.l);
        }
        d.setValue(s.arxiv.timezone).onChange(async (v) => {
          s.arxiv.timezone = v;
          await this.plugin.saveSettings();
        });
      })
      .addText((t) => {
        t.setPlaceholder("or enter custom timezone")
          .setValue("")
          .onChange(async (v) => {
            if (v.trim()) {
              s.arxiv.timezone = v.trim();
              await this.plugin.saveSettings();
            }
          });
      });

    // ─── Output & Schedule ────────────────────────────
    this.sectionHeading(containerEl, "Output & Schedule", "schedule");

    new Setting(containerEl)
      .setName("Daily path")
      .setDesc("Relative path in vault")
      .addText((t) => {
        t.setValue(s.output.dailyDir);
        t.inputEl.addEventListener("input", () => {
          const validation = validateOutputDirectoryDraft(t.inputEl.value);
          t.inputEl.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
          t.inputEl.toggleClass("is-invalid", !validation.ok);
        });
        t.inputEl.addEventListener("change", () => {
          void this.applyOutputDirectoryDraft("dailyDir", t.inputEl.value, t.inputEl);
        });
      });

    new Setting(containerEl)
      .setName("Papers path")
      .setDesc("Relative path in vault")
      .addText((t) => {
        t.setValue(s.output.papersDir);
        t.inputEl.addEventListener("input", () => {
          const validation = validateOutputDirectoryDraft(t.inputEl.value);
          t.inputEl.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
          t.inputEl.toggleClass("is-invalid", !validation.ok);
        });
        t.inputEl.addEventListener("change", () => {
          void this.applyOutputDirectoryDraft("papersDir", t.inputEl.value, t.inputEl);
        });
      });

    new Setting(containerEl)
      .setName("Link style")
      .setDesc("Markdown links generated in daily reports")
      .addDropdown((d) =>
        d
          .addOption("wikilink", "Obsidian wikilink")
          .addOption("relative", "Standard relative link")
          .setValue(s.output.linkStyle ?? "wikilink")
          .onChange(async (v) => {
            s.output.linkStyle = v === "relative" ? "relative" : "wikilink";
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Summary language")
      .setDesc("Language used for daily and detail summaries")
      .addDropdown((d) =>
        d
          .addOption("zh", "Chinese")
          .addOption("en", "English")
          .setValue(s.output.summaryLanguage ?? "zh")
          .onChange(async (v) => {
            s.output.summaryLanguage = v === "en" ? "en" : "zh";
            await this.plugin.saveSettings();
          }),
      );

    const runWindow = new Setting(containerEl)
      .setName("Run window")
      .setDesc("Local 24-hour time window for scheduled polling.");
    renderRunWindowTimeSelect(
      runWindow.controlEl,
      "Start",
      "arxiv-daily-run-window-start",
      s.schedule.runAtLocal,
      async (value) => {
        s.schedule.runAtLocal = value;
        await this.plugin.saveSettings();
      },
    );
    renderRunWindowTimeSelect(
      runWindow.controlEl,
      "End",
      "arxiv-daily-run-window-end",
      s.schedule.runUntilLocal,
      async (value) => {
        s.schedule.runUntilLocal = value;
        await this.plugin.saveSettings();
      },
    );

    this.attachHelp(
      new Setting(containerEl).setName("Tick interval (min)").addText((t) =>
        t.setValue(String(s.schedule.tickIntervalMin)).onChange(async (v) => {
          s.schedule.tickIntervalMin = Math.max(1, Number(v) || 20);
          await this.plugin.saveSettings();
          this.plugin.restartScheduler();
        }),
      ),
      "How often the scheduler interval wakes up to check pending dates. Default 20 minutes.",
    );

    // ─── Advanced ─────────────────────────────────────
    this.sectionHeading(containerEl, "Advanced", "advanced");

    this.attachHelp(
      new Setting(containerEl).setName("Log level").addDropdown((d) =>
        d
          .addOption("debug", "debug")
          .addOption("info", "info")
          .addOption("warn", "warn")
          .addOption("error", "error")
          .setValue(s.advanced.logLevel)
          .onChange(async (v) => {
            s.advanced.logLevel = v as any;
            await this.plugin.saveSettings();
            this.plugin.logger.setLevel(v as any);
          }),
      ),
      "Console log verbosity. 'debug' is noisy; 'info' is the default.",
    );
  }

  private renderApiKeySetting(containerEl: HTMLElement): void {
    const configured = Boolean(this.plugin.settings.llm.apiKey.trim());
    const setting = new Setting(containerEl)
      .setName("API Key")
      .setDesc("Stored locally in data.json. Saved keys are never rendered into this page.");
    let editing = !configured;
    let draft = "";
    const input = document.createElement("input");
    input.type = editing ? "password" : "text";
    input.classList.add("arxiv-daily-settings__llm-input");
    input.value = configured ? API_KEY_CONFIGURED_SENTINEL : "";
    input.placeholder = "Enter API key";
    input.readOnly = !editing;
    setting.controlEl.appendChild(input);

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
    replace.addEventListener("click", async () => {
      if (!editing) {
        enterEdit();
        return;
      }
      const next = draft.trim();
      if (!next) return;
      this.plugin.settings.llm.apiKey = next;
      this.plugin.logger.setSensitiveValues([next]);
      await this.plugin.saveSettings();
      this.refreshSetupGuide();
      reset();
      clear.hidden = false;
    });
    cancel.addEventListener("click", () => {
      if (configured || this.plugin.settings.llm.apiKey.trim()) reset();
      else {
        draft = "";
        input.value = "";
      }
    });
    clear.addEventListener("click", async () => {
      const confirmed = await this.confirmReplace(
        "Clear the saved API key? LLM operations will stop until a replacement is saved.",
        "Clear",
      );
      if (!confirmed) return;
      this.plugin.settings.llm.apiKey = "";
      this.plugin.logger.setSensitiveValues([]);
      await this.plugin.saveSettings();
      this.refreshSetupGuide();
      this.display();
    });
  }

  private renderSetupGuide(containerEl: HTMLElement): void {
    const guide = this.createSetupGuide();
    if (guide) containerEl.appendChild(guide);
  }

  private refreshSetupGuide(): void {
    const current = this.containerEl.querySelector(".arxiv-daily-setup");
    const next = this.createSetupGuide();
    if (current instanceof HTMLElement) {
      if (next) {
        current.replaceWith(next);
      } else {
        current.remove();
      }
    } else if (next) {
      this.containerEl.prepend(next);
    }
  }

  private createSetupGuide(): HTMLElement | null {
    const status = getSetupStatus(this.plugin.settings);
    if (!shouldRenderSetupGuide(status)) return null;

    const guide = document.createElement("section");
    guide.addClass("arxiv-daily-setup");
    const header = guide.createEl("div", {
      cls: "arxiv-daily-setup__header",
    });
    header.createEl("div", {
      cls: "arxiv-daily-setup__title",
      text: "Getting started",
    });
    header.createEl("div", {
      cls: "arxiv-daily-setup__subtitle",
      text: "Complete these items before the first run.",
    });

    const list = guide.createEl("div", {
      cls: "arxiv-daily-setup__list",
    });
    this.renderSetupItem(
      list,
      status.llmReady,
      "LLM API key, base URL, and model",
      "Configure LLM",
      () => this.scrollToSection("llm"),
    );
    this.renderSetupItem(
      list,
      status.categoriesReady,
      "At least one arXiv category",
      "Choose categories",
      () => this.scrollToSection("arxiv"),
    );
    this.renderSetupItem(
      list,
      status.topicsReady,
      "At least one complete research topic",
      "Set topics",
      () => this.scrollToSection("topics"),
    );
    this.renderSetupItem(
      list,
      status.readyToRun,
      "Ready to run",
      "Review missing items",
      () => this.scrollToFirstMissingSection(status),
    );

    if (!status.readyToRun && status.reasons.length > 0) {
      const details = guide.createEl("details", {
        cls: "arxiv-daily-setup__details",
      });
      details.createEl("summary", { text: "Show missing configuration" });
      const ul = details.createEl("ul");
      for (const reason of status.reasons) ul.createEl("li", { text: reason });
    }

    const actions = guide.createEl("div", {
      cls: "arxiv-daily-setup__actions",
    });
    const run = actions.createEl("button", {
      text: "Run today",
      attr: { type: "button" },
    }) as HTMLButtonElement;
    run.disabled = !status.readyToRun;
    run.addEventListener("click", () => {
      void this.executeCommand("run-now");
    });

    const dashboard = actions.createEl("button", {
      text: "Open dashboard",
      attr: { type: "button" },
    });
    dashboard.addEventListener("click", () => {
      void openDashboardView(this.plugin);
    });

    return guide;
  }

  private renderSetupItem(
    parent: HTMLElement,
    done: boolean,
    label: string,
    actionLabel: string,
    onAction: () => void,
  ): void {
    const item = parent.createEl("div", {
      cls: `arxiv-daily-setup__item ${done ? "is-done" : "is-pending"}`,
    });
    item.createEl("span", {
      cls: "arxiv-daily-setup__check",
      text: done ? "✓" : "•",
    });
    item.createEl("span", {
      cls: "arxiv-daily-setup__label",
      text: label,
    });
    const action = item.createEl("button", {
      cls: "arxiv-daily-setup__link",
      text: done ? "Done" : actionLabel,
      attr: { type: "button" },
    }) as HTMLButtonElement;
    action.disabled = done;
    action.addEventListener("click", onAction);
  }

  private scrollToFirstMissingSection(status: ReturnType<typeof getSetupStatus>): void {
    if (!status.llmReady) {
      this.scrollToSection("llm");
      return;
    }
    if (!status.categoriesReady) {
      this.scrollToSection("arxiv");
      return;
    }
    if (!status.topicsReady) {
      this.scrollToSection("topics");
      return;
    }
  }

  private scrollToSection(section: "llm" | "arxiv" | "topics" | "schedule" | "advanced"): void {
    const target = this.containerEl.querySelector(
      `[data-arxiv-daily-section="${section}"]`,
    );
    if (target instanceof HTMLElement) {
      target.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  }

  private async executeCommand(commandId: string): Promise<void> {
    await executeObsidianCommand(
      this.plugin.app,
      commandId,
      this.plugin.manifest.id,
    );
  }

  private renderTopicCard(container: HTMLElement, topics: Topic[], index: number): void {
    const topic = topics[index];
    if (!topic) return;
    const isExpanded = this.expandedTopics.has(topic.id);
    const idPrefix = `arxiv-daily-topic-${stableDomId(topic.id)}`;
    const formId = `${idPrefix}-form`;

    const card = container.createDiv({
      cls: "arxiv-daily-settings__topic-card",
    });

    // ─── Header row (always visible, clickable) ────────────
    const header = card.createEl("button", {
      cls: "arxiv-daily-settings__topic-header",
      attr: {
        type: "button",
        "aria-expanded": String(isExpanded),
        "aria-controls": formId,
      },
    });

    const caret = header.createEl("span", {
      cls: "arxiv-daily-settings__topic-caret",
      text: isExpanded ? "▾" : "▸",
    });

    const titleSpan = header.createEl("span", {
      cls: "arxiv-daily-settings__topic-title",
      text: topic.name.trim() || "(unnamed)",
    });
    titleSpan.toggleClass("is-muted", !topic.name.trim());

    let star: HTMLElement | null = null;
    if (topic.detail) {
      star = header.createEl("span", {
        cls: "arxiv-daily-settings__topic-star",
        text: "★",
        attr: { title: "Detail report enabled" },
      });
    }

    let tagChip: HTMLElement | null = null;
    if (topic.tag) {
      tagChip = header.createEl("span", {
        cls: "arxiv-daily-settings__topic-tag",
        text: "#" + topic.tag,
      });
    }

    // ─── Expanded form (toggled via display) ────────────────
    const form = card.createDiv({
      cls: "arxiv-daily-settings__topic-form",
    });
    form.id = formId;
    form.hidden = !isExpanded;
    form.toggleClass("is-collapsed", !isExpanded);

    // Name row
    const nameRow = form.createDiv({
      cls: "arxiv-daily-settings__topic-row",
    });
    const nameId = `${idPrefix}-name`;
    const nameHintId = `${nameId}-hint`;
    nameRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: "Name",
      attr: { for: nameId },
    });
    this.hint(nameRow, "Heading text used as the section title in the daily report.", nameHintId);
    const nameInput = nameRow.createEl("input", {
      cls: "arxiv-daily-settings__topic-name-input",
      type: "text",
      attr: { id: nameId, "aria-describedby": nameHintId },
    });
    nameInput.value = topic.name;
    nameInput.placeholder = "e.g. Photometric Redshift";

    // Tag row
    const tagRow = form.createDiv({
      cls: "arxiv-daily-settings__topic-row",
    });
    const tagId = `${idPrefix}-tag`;
    const tagHintId = `${tagId}-hint`;
    tagRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: "Tag",
      attr: { for: tagId },
    });
    this.hint(tagRow, "Kebab-case ASCII slug. Written into each paper's YAML frontmatter as an Obsidian #tag.", tagHintId);
    const tagInput = tagRow.createEl("input", {
      cls: "arxiv-daily-settings__topic-tag-input",
      type: "text",
      attr: { id: tagId, "aria-describedby": tagHintId },
    });
    tagInput.value = topic.tag;
    tagInput.placeholder = "kebab-case-slug";
    const autoBadge = tagRow.createEl("span", {
      cls: "arxiv-daily-settings__topic-auto",
      text: "Auto",
    });
    const refreshAutoBadge = () => {
      autoBadge.toggleClass("is-hidden", topic.tag !== slugify(topic.name));
    };
    refreshAutoBadge();

    const refreshHeader = () => {
      titleSpan.textContent = topic.name.trim() || "(unnamed)";
      titleSpan.toggleClass("is-muted", !topic.name.trim());
    };

    nameInput.oninput = async () => {
      const wasAuto = topic.tag === slugify(topic.name);
      topic.name = nameInput.value;
      if (wasAuto) {
        const derived = slugify(topic.name);
        topic.tag = derived || `topic-${index + 1}`;
        tagInput.value = topic.tag;
      }
      refreshAutoBadge();
      refreshHeader();
      await this.plugin.saveSettings();
      this.refreshSetupGuide();
    };

    tagInput.oninput = async () => {
      topic.tag = tagInput.value;
      refreshAutoBadge();
      await this.plugin.saveSettings();
      this.refreshSetupGuide();
    };

    // Description
    const descRow = form.createDiv({
      cls: "arxiv-daily-settings__topic-row",
    });
    const descId = `${idPrefix}-description`;
    const descHintId = `${descId}-hint`;
    descRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: "Description",
      attr: { for: descId },
    });
    this.hint(descRow, "Plain-language description of what belongs here. The LLM reads this to decide which papers go into this topic.", descHintId);
    const descArea = descRow.createEl("textarea", {
      cls: "arxiv-daily-settings__topic-description",
      attr: { id: descId, "aria-describedby": descHintId },
    });
    descArea.value = topic.description;
    descArea.rows = 3;
    descArea.placeholder =
      "What kinds of papers should be grouped under this topic? (natural language)";
    descArea.oninput = async () => {
      topic.description = descArea.value;
      await this.plugin.saveSettings();
      this.refreshSetupGuide();
    };

    // Detail toggle + delete (right-aligned, only visible when expanded)
    this.hint(form, "Detail report = generate a full, deep-dive markdown file for primary contributions to this topic. Delete = remove this topic.");
    const footer = form.createDiv({
      cls: "arxiv-daily-settings__topic-footer",
    });

    const detailLabel = footer.createEl("label", {
      cls: "arxiv-daily-settings__topic-detail-label",
    });
    const detailCheckbox = detailLabel.createEl("input", { type: "checkbox" });
    detailCheckbox.checked = topic.detail;
    detailCheckbox.addClass("arxiv-daily-settings__topic-detail-checkbox");
    detailLabel.appendText("Detail report");
    detailCheckbox.onchange = async () => {
      topic.detail = detailCheckbox.checked;
      await this.plugin.saveSettings();
      // Refresh the header star indicator without a full re-render.
      star?.remove();
      star = null;
      if (topic.detail) {
        star = document.createElement("span");
        star.addClass("arxiv-daily-settings__topic-star");
        star.textContent = "★";
        star.setAttribute("title", "Detail report enabled");
        // Insert after the title (second child).
        if (tagChip) header.insertBefore(star, tagChip);
        else header.appendChild(star);
      }
    };

    const delBtn = footer.createEl("button", {
      text: "Delete",
      attr: { type: "button" },
    });
    delBtn.classList.add("mod-warning");
    delBtn.onclick = async (e) => {
      e.stopPropagation();
      const topicName = topic.name.trim() || "(unnamed)";
      const confirmed = await this.confirmReplace(
        `Delete the research topic "${topicName}"? This cannot be undone.`,
        "Delete",
      );
      if (!confirmed) return;
      topics.splice(index, 1);
      this.expandedTopics.delete(topic.id);
      await this.plugin.saveSettings();
      this.display();
    };

    // Toggle expand/collapse on header click
    header.onclick = () => {
      const expanded = !this.expandedTopics.has(topic.id);
      if (expanded) this.expandedTopics.add(topic.id);
      else this.expandedTopics.delete(topic.id);
      form.hidden = !expanded;
      form.toggleClass("is-collapsed", !expanded);
      header.setAttribute("aria-expanded", String(expanded));
      caret.textContent = expanded ? "▾" : "▸";
    };
  }

  private confirmReplace(message: string, confirmLabel = "Replace"): Promise<boolean> {
    return new Promise((resolve) => {
      const modal = new Modal(this.app);
      modal.titleEl.setText("Confirm");
      modal.contentEl.createEl("p", { text: message });
      const btns = modal.contentEl.createDiv({
        cls: "arxiv-daily-modal-button-row",
      });
      const cancel = btns.createEl("button", { text: "Cancel" });
      const ok = btns.createEl("button", { text: confirmLabel });
      ok.classList.add("mod-warning");
      let settled = false;
      const finish = (value: boolean) => {
        if (settled) return;
        settled = true;
        resolve(value);
        modal.close();
      };
      cancel.onclick = () => finish(false);
      ok.onclick = () => finish(true);
      modal.onClose = () => finish(false);
      modal.open();
    });
  }

  private showModelDropdown(models: string[], container: HTMLElement): void {
    // Find the existing dropdown in the model setting
    const modelSetting = container.closest(".setting-item");
    if (!modelSetting) return;

    const select = modelSetting.querySelector("select") as HTMLSelectElement;
    if (!select) return;

    // Clear existing options without parsing HTML.
    select.replaceChildren();

    // Add new options
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      select.appendChild(option);
    }

    // Pre-select current model if in list
    const currentModel = this.plugin.settings.llm.model;
    if (models.includes(currentModel)) {
      select.value = currentModel;
    } else if (models.length > 0) {
      // Select first model if current not in list
      select.value = models[0]!;
      this.plugin.settings.llm.model = models[0]!;
      void this.plugin.saveSettings().catch(() => {});
    }
  }

  private textareaSetting(
    container: HTMLElement,
    name: string,
    desc: string,
    value: string,
    onChange: (v: string) => Promise<void>,
  ): Setting {
    return new Setting(container)
      .setName(name)
      .setDesc(desc)
      .addTextArea((t) => {
        t.setValue(value).onChange((v) => onChange(v));
        t.inputEl.rows = 6;
        t.inputEl.addClass("arxiv-daily-settings__textarea");
      });
  }
}

function stableDomId(value: string): string {
  const normalized = value.replace(/[^A-Za-z0-9_-]/g, "-");
  return normalized || "unnamed";
}

export function isValidLocalTime(value: string): boolean {
  return /^(?:[01]\d|2[0-3]):[0-5]\d$/.test(value);
}

export interface RunWindowTimeOption {
  value: string;
  label: string;
  valid: boolean;
}

export function runWindowTimeOptions(current: string): RunWindowTimeOption[] {
  const values: RunWindowTimeOption[] = [];
  for (let hour = 0; hour < 24; hour += 1) {
    for (let minute = 0; minute < 60; minute += 15) {
      const value = `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
      values.push({ value, label: value, valid: true });
    }
  }

  if (!values.some((option) => option.value === current)) {
    const valid = isValidLocalTime(current);
    values.push({
      value: current,
      label: valid ? current : `${current || "(empty)"} — invalid`,
      valid,
    });
  }
  return values.sort((a, b) => a.value.localeCompare(b.value));
}

function renderRunWindowTimeSelect(
  parent: HTMLElement,
  labelText: string,
  id: string,
  current: string,
  onChange: (value: string) => Promise<void>,
): void {
  const field = parent.createDiv({
    cls: "arxiv-daily-settings__time-field",
  });
  field.createEl("label", {
    cls: "arxiv-daily-settings__time-label",
    text: labelText,
    attr: { for: id },
  });
  const select = field.createEl("select", {
    cls: "dropdown arxiv-daily-settings__time-select",
    attr: { id },
  });
  for (const option of runWindowTimeOptions(current)) {
    const optionEl = select.createEl("option", {
      value: option.value,
      text: option.label,
    });
    optionEl.disabled = !option.valid;
  }
  select.value = current;
  select.addEventListener("change", () => {
    if (!isValidLocalTime(select.value)) return;
    void onChange(select.value);
  });
}

function addCategoryOptions(
  selectEl: HTMLSelectElement,
  current?: string,
): void {
  let hasCurrent = false;
  for (const group of ARXIV_CATEGORIES) {
    const optgroup = selectEl.createEl("optgroup");
    optgroup.label = group.label;
    for (const cat of group.categories) {
      if (cat.id === current) hasCurrent = true;
      const opt = optgroup.createEl("option");
      opt.value = cat.id;
      opt.textContent = `${cat.id} — ${cat.name}`;
    }
  }
  if (current && !hasCurrent) {
    const opt = selectEl.createEl("option");
    opt.value = current;
    opt.textContent = `${current} — custom`;
  }
}

function normalizeUniqueCategories(categories: string[]): string[] {
  const out: string[] = [];
  for (const value of categories) {
    const category = value.trim();
    if (!category || out.includes(category)) continue;
    out.push(category);
  }
  return out;
}

function nextCategoryCandidate(existing: string[]): string {
  const seen = new Set(existing);
  for (const group of ARXIV_CATEGORIES) {
    for (const category of group.categories) {
      if (!seen.has(category.id)) return category.id;
    }
  }
  return "cs.LG";
}

function categoriesWillChange(current: string[], next: string[]): boolean {
  if (current.length !== next.length) return true;
  return current.some((category, index) => category !== next[index]);
}

function quickStartTemplateConfirmMessage(
  topicCount: number,
  templateName: string,
  replacesCategories: boolean,
): string {
  if (topicCount > 0 && replacesCategories) {
    return `Replace your ${topicCount} topic(s) and arXiv categories with the "${templateName}" template?`;
  }
  if (topicCount > 0) {
    return `Replace your ${topicCount} topic(s) with the "${templateName}" template?`;
  }
  return `Replace your arXiv categories with the "${templateName}" template?`;
}
