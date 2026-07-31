import {
  App,
  Modal,
  Notice,
  PluginSettingTab,
  requireApiVersion,
  Setting,
  type SettingDefinitionItem,
} from "obsidian";
import type ArxivDailyPlugin from "../../main";
import {
  buildSettingDefinitions,
  readSettingValue,
  SETTING_KEYS,
  writeSettingValue,
} from "./definitions";
import * as declarativeRows from "./declarative-rows";
import {
  describeResult,
  detailSelectionPreset,
  formatDate,
  todayInTz,
  type LogLevel,
} from "@arxiv-daily/core";
import { ARXIV_CATEGORIES } from "@arxiv-daily/core";
import { TOPIC_TEMPLATES } from "@arxiv-daily/core";
import type { Topic } from "@arxiv-daily/core";
import { slugify } from "@arxiv-daily/core";
import {
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "@arxiv-daily/core";
import { arxivCategories } from "@arxiv-daily/core";
import { getSetupStatus, shouldRenderSetupGuide } from "../onboarding";
import { openDashboardView } from "../dashboard/view";
import { LlmClient, redactText } from "@arxiv-daily/core";
import {
  ARXIV_DAILY_DOCS_URL,
  ARXIV_DAILY_REPO_URL,
  buildBugReportUrl,
  buildFeatureRequestUrl,
} from "../feedback";
import { ObsidianResourceOpener } from "../hosts/obsidian/resource-opener";

export const API_KEY_CONFIGURED_SENTINEL = "Configured";

export async function persistApiKeyChange(
  settings: { llm: { apiKey: string } },
  logger: { setSensitiveValues(values: readonly string[]): void },
  saveSettings: () => Promise<void>,
  nextApiKey: string,
): Promise<void> {
  const previousApiKey = settings.llm.apiKey;
  settings.llm.apiKey = nextApiKey;
  logger.setSensitiveValues(nextApiKey ? [nextApiKey] : []);
  try {
    await saveSettings();
  } catch (error) {
    settings.llm.apiKey = previousApiKey;
    logger.setSensitiveValues(previousApiKey ? [previousApiKey] : []);
    throw error;
  }
}

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
      message: "This address uses plain HTTP on this computer. Only continue if you meant to use a local AI service.",
    };
  }
  return {
    kind: "plaintext",
    message: "This address uses plain HTTP. Your API key would be sent without encryption—prefer HTTPS.",
  };
}

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return host === "localhost" || host === "::1" || /^127(?:\.\d{1,3}){3}$/.test(host);
}

function isLogLevel(value: string): value is LogLevel {
  return value === "debug" || value === "info" || value === "warn" || value === "error";
}

export class ArxivDailySettingTab extends PluginSettingTab {
  private expandedTopics = new Set<string>();

  constructor(app: App, public plugin: ArxivDailyPlugin) {
    super(app, plugin);
  }

  /**
   * Declarative settings for Obsidian 1.13+ (searchable in Settings
   * search). Host callbacks bind the shared row renderers and the tab's
   * mutation methods; display() stays as the <1.13 fallback. Only called
   * by the framework on 1.13+, so no version guard is needed here.
   */
  override getSettingDefinitions(): SettingDefinitionItem[] {
    return buildSettingDefinitions({
      plugin: this.plugin,
      renderSetupGuideRow: (setting) =>
        declarativeRows.renderSetupGuideRow(this, setting),
      renderScheduleEnabledRow: (setting) =>
        declarativeRows.renderScheduleEnabledRow(this, setting),
      renderApiKeyRow: (setting) =>
        declarativeRows.renderApiKeyRow(this, setting),
      renderModelRow: (setting) =>
        declarativeRows.renderModelRow(this, setting),
      renderCategoryRow: (setting, index) =>
        declarativeRows.renderCategoryRow(this, setting, index),
      renderQuickStartRow: (setting) =>
        declarativeRows.renderQuickStartRow(this, setting),
      renderTopicRow: (setting, index) =>
        declarativeRows.renderTopicRow(this, setting, index),
      renderTimezoneRow: (setting) =>
        declarativeRows.renderTimezoneRow(this, setting),
      renderRunWindowRow: (setting) =>
        declarativeRows.renderRunWindowRow(this, setting),
      renderTickIntervalRow: (setting) =>
        declarativeRows.renderTickIntervalRow(this, setting),
      renderEmailGuideRow: (setting) =>
        declarativeRows.renderEmailGuideRow(this, setting),
      renderEmailModeRow: (setting) =>
        declarativeRows.renderEmailModeRow(this, setting),
      renderEmailApiKeyRow: (setting) =>
        declarativeRows.renderEmailApiKeyRow(this, setting),
      renderHostedTokenRow: (setting) =>
        declarativeRows.renderHostedTokenRow(this, setting),
      addCategory: () => void this.addCategory(),
      deleteCategory: (index) => void this.deleteCategory(index),
      reorderCategories: (oldIndex, newIndex) =>
        void this.reorderCategories(oldIndex, newIndex),
      addTopic: () => void this.addTopic(),
      reorderTopics: (oldIndex, newIndex) =>
        void this.reorderTopics(oldIndex, newIndex),
      sendVerificationEmail: () => declarativeRows.sendVerificationEmail(this),
      sendTestEmail: () => declarativeRows.sendTestEmail(this),
    });
  }

  /** Resolve a flat declarative key against the nested settings object. */
  override getControlValue(key: string): unknown {
    return readSettingValue(this.plugin.settings, key);
  }

  /**
   * Persist a flat declarative key. Email To and From mirror display()'s
   * trimming on change; From name is written raw there, so it stays raw
   * here too.
   */
  override async setControlValue(key: string, value: unknown): Promise<void> {
    if (
      typeof value === "string" &&
      (key === SETTING_KEYS.email.to || key === SETTING_KEYS.email.fromEmail)
    ) {
      value = value.trim();
    }
    writeSettingValue(this.plugin.settings, key, value);
    await this.plugin.saveSettings();
  }

  /** Append an accessible circled "?" to a setting name. */
  private attachHelp(setting: Setting, text: string): Setting {
    setting.nameEl.createSpan({
      cls: "arxiv-daily-settings__help",
      text: "?",
      attr: { title: text, "aria-label": text },
    });
    return setting;
  }

  private reportActionError(action: string, error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    this.plugin.logger.error(`settings: ${action} failed`, error);
    new Notice(`arXiv Daily: ${action} failed: ${message}`, 10_000);
  }

  public runAction(action: string, operation: () => Promise<unknown>): void {
    void operation().catch((error) => this.reportActionError(action, error));
  }

  /** Inline muted hint, used inside topic cards under a label. */
  private hint(parent: HTMLElement, text: string, id?: string): HTMLElement {
    const hint = parent.createDiv({
      cls: "arxiv-daily-settings__hint",
      text,
    });
    if (id) hint.id = id;
    return hint;
  }

  private sectionHeading(
    containerEl: HTMLElement,
    name: string,
    section: "llm" | "arxiv" | "topics" | "schedule" | "email" | "advanced",
    desc?: string,
  ): Setting {
    const heading = new Setting(containerEl).setName(name).setHeading();
    if (desc) heading.setDesc(desc);
    heading.settingEl.addClass("arxiv-daily-settings__section");
    heading.settingEl.setAttribute("data-arxiv-daily-section", section);
    // Ensure heading name is easy to scan in long settings pages.
    heading.nameEl?.addClass("arxiv-daily-settings__section-title");
    return heading;
  }

  /** Full-width guide block aligned with Setting name column (not indented desc). */
  /** Guide strip copy for the current email mode; shared with the 1.13+ rows. */
  public emailGuideContent(): { title: string; lines: string[] } {
    const hostedMode = this.plugin.settings.email.mode === "hosted";
    return {
      title: hostedMode ? "Official delivery (Beta)" : "Send yourself",
      lines: hostedMode
        ? [
            "1. Enter your email, then send a verification message.",
            "2. Open the link in that email and copy the code shown on the page.",
            "3. Paste the code below, send a test email, then turn on daily auto-send.",
            "Capacity is limited: only a few messages per inbox per day (tests count). For heavier use, switch to Send yourself.",
          ]
        : [
            "1. Create a free Resend account and an API key at resend.com.",
            "2. Paste the key below. For a quick start, put your Resend account email in Your email and leave From email empty.",
            "3. Send a test email, then turn on daily auto-send when it works.",
          ],
    };
  }

  private emailGuide(
    containerEl: HTMLElement,
    opts: { title: string; lines: string[] },
  ): void {
    const wrap = containerEl.createDiv({
      cls: "arxiv-daily-settings__email-guide",
    });
    wrap.createDiv({
      cls: "arxiv-daily-settings__email-guide-title",
      text: opts.title,
    });
    for (const line of opts.lines) {
      wrap.createDiv({
        cls: "arxiv-daily-settings__email-guide-line",
        text: line,
      });
    }
  }

  public async setArxivCategories(categories: string[]): Promise<void> {
    const normalized = normalizeUniqueCategories(categories);
    this.plugin.settings.arxiv.categories = normalized;
    if (normalized[0]) this.plugin.settings.arxiv.category = normalized[0];
    await this.plugin.saveSettings();
  }

  /** Re-render the tab: declarative update() on Obsidian 1.13+, display() otherwise. */
  public refreshSettings(): void {
    if (
      requireApiVersion("1.13.0") &&
      this.getSettingDefinitions().length > 0
    ) {
      this.update();
    } else {
      this.display();
    }
  }

  /** Append a category (the first arXiv option not already in the list). */
  public async addCategory(): Promise<void> {
    const categories = arxivCategories(this.plugin.settings.arxiv);
    await this.setArxivCategories([
      ...categories,
      nextCategoryCandidate(categories),
    ]);
    this.refreshSettings();
  }

  /** Remove a category by index; keeps the last remaining category. */
  public async deleteCategory(index: number): Promise<void> {
    const categories = arxivCategories(this.plugin.settings.arxiv);
    if (categories.length <= 1) return;
    await this.setArxivCategories(categories.filter((_, j) => j !== index));
    this.refreshSettings();
  }

  /** Move a category to a new position (list drag-to-reorder). */
  public async reorderCategories(
    oldIndex: number,
    newIndex: number,
  ): Promise<void> {
    const categories = [...arxivCategories(this.plugin.settings.arxiv)];
    const [moved] = categories.splice(oldIndex, 1);
    if (moved === undefined) return;
    categories.splice(newIndex, 0, moved);
    await this.setArxivCategories(categories);
    this.refreshSettings();
  }

  /** Append a blank, expanded topic card. */
  public async addTopic(): Promise<void> {
    const newId = crypto.randomUUID();
    const topics = this.plugin.settings.arxiv.topics;
    topics.push({
      id: newId,
      name: "",
      tag: `topic-${topics.length + 1}`,
      description: "",
      detail: false,
    });
    this.expandedTopics.add(newId);
    await this.plugin.saveSettings();
    this.refreshSettings();
  }

  /** Delete a topic after confirmation; returns whether it was deleted. */
  public async deleteTopic(index: number): Promise<boolean> {
    const topics = this.plugin.settings.arxiv.topics;
    const topic = topics[index];
    if (!topic) return false;
    const topicName = topic.name.trim() || "(unnamed)";
    const confirmed = await this.confirmReplace(
      `Delete the research topic "${topicName}"? This cannot be undone.`,
      "Delete",
    );
    if (!confirmed) return false;
    topics.splice(index, 1);
    this.expandedTopics.delete(topic.id);
    await this.plugin.saveSettings();
    this.refreshSettings();
    return true;
  }

  /** Move a topic to a new position (list drag-to-reorder). */
  public async reorderTopics(oldIndex: number, newIndex: number): Promise<void> {
    const topics = this.plugin.settings.arxiv.topics;
    const [moved] = topics.splice(oldIndex, 1);
    if (moved === undefined) return;
    topics.splice(newIndex, 0, moved);
    await this.plugin.saveSettings();
    this.refreshSettings();
  }

  /**
   * Apply a quick-start template, replacing topics (and categories) after
   * confirmation when the current setup would be overwritten.
   */
  public async applyTopicTemplate(templateId: string): Promise<void> {
    const tpl = TOPIC_TEMPLATES.find((t) => t.id === templateId);
    if (!tpl) return;
    const settings = this.plugin.settings;
    const categories = arxivCategories(settings.arxiv);
    const apply = async () => {
      settings.arxiv.category = tpl.category;
      settings.arxiv.categories = [tpl.category];
      settings.arxiv.topics = tpl.topics.map((t) => ({
        ...t,
        id: crypto.randomUUID(),
      }));
      await this.plugin.saveSettings();
      this.refreshSettings();
    };
    const replacesCategories = categoriesWillChange(categories, [tpl.category]);
    if (settings.arxiv.topics.length === 0 && !replacesCategories) {
      await apply();
      return;
    }
    const confirmed = await this.confirmReplace(
      quickStartTemplateConfirmMessage(
        settings.arxiv.topics.length,
        tpl.name,
        replacesCategories,
      ),
    );
    if (confirmed) await apply();
  }

  public async saveRunWindowTime(
    key: "runAtLocal" | "runUntilLocal",
    value: string,
  ): Promise<void> {
    const previous = this.plugin.settings.schedule[key];
    this.plugin.settings.schedule[key] = value;
    try {
      await this.plugin.saveSettings();
    } catch (error) {
      this.plugin.settings.schedule[key] = previous;
      this.reportActionError("save run window", error);
    }
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

    // ─── Enable toggle (top) ─────────────────────────
    new Setting(containerEl)
      .setName(`Enable · ${s.schedule.enabled ? "Running" : "Paused"}`)
      .setDesc("When on, daily reports run automatically on weekdays (weekends are skipped).")
      .addToggle((t) =>
        t.setValue(s.schedule.enabled).onChange(async (v) => {
          await this.plugin.setScheduleEnabled(v);
          this.display();
        }),
      );

    // ─── LLM ──────────────────────────────────────────
    this.sectionHeading(containerEl, "AI model", "llm");

    // Base URL — always editable, default to DeepSeek
    new Setting(containerEl)
      .setName("API base URL")
      .setDesc("Where chat requests are sent. Default is DeepSeek; change only if you use another provider.")
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
      .setDesc("Choose a model, or click Get models to load the list from your provider.");

    // Get models button
    modelSetting.addButton((b) => {
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
      ? "Let the model spend extra effort on harder questions (Anthropic)."
      : s.llm.provider === "deepseek"
        ? "Let the model spend extra effort on harder questions (DeepSeek reasoning)."
        : "Let the model spend extra effort on harder questions when the provider supports it.";

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
      .setDesc("How hard the model tries when thinking mode is on. Higher may be slower and cost more.")
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
        t.setPlaceholder("Or enter custom value")
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
      .setName("arXiv categories")
      .setDesc("Which arXiv subject areas to watch. You can add several; the same paper is only kept once.")
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
          t.setPlaceholder("Or enter custom category")
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
            .onClick(() => void this.deleteCategory(i)),
        );
    }

    new Setting(containerEl).addButton((b) =>
      b.setButtonText("Add category").onClick(() => void this.addCategory()),
    );

    // ─── Research Topics ─────────────────────────────
    this.sectionHeading(
      containerEl,
      "Research topics",
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
          await this.applyTopicTemplate(id);
        });
      })
      .addButton((b) => {
        b.setButtonText("Add topic").onClick(() => void this.addTopic());
      });

    const topicsContainer = containerEl.createDiv();
    if (s.arxiv.topics.length === 0) {
      const empty = topicsContainer.createDiv({
        cls: "arxiv-daily-settings__empty-topics",
      });
      empty.createEl("strong", { text: "No topics yet." });
      empty.createDiv({
        text: "Pick a template above or click Add topic to define what to track. Daily reports need at least one topic before AI runs.",
      });
    }
    for (let i = 0; i < s.arxiv.topics.length; i++) {
      this.renderTopicCard(topicsContainer, s.arxiv.topics, i);
    }

    new Setting(containerEl)
      .setName("Automatic detail notes")
      .setDesc(
        "How often the plugin writes a longer note for a paper. Only topics with Detail report turned on are considered. Manual “summarize paper” is unchanged.",
      )
      .addDropdown((d) => {
        d.addOption("conservative", "Fewer")
          .addOption("balanced", "Recommended")
          .addOption("broad", "More");
        if (s.detailSelection.profile === "custom") {
          d.addOption("custom", "Custom (current values)");
        }
        d.setValue(s.detailSelection.profile).onChange(async (profile) => {
          if (
            profile !== "conservative" &&
            profile !== "balanced" &&
            profile !== "broad"
          ) return;
          s.detailSelection = detailSelectionPreset(profile);
          await this.plugin.saveSettings();
          this.display();
        });
      });

    new Setting(containerEl)
      .setName("Timezone")
      .addDropdown((d) => {
        for (const zone of TIMEZONE_OPTIONS) {
          d.addOption(zone.value, zone.label);
        }
        d.setValue(s.arxiv.timezone).onChange(async (v) => {
          s.arxiv.timezone = v;
          await this.plugin.saveSettings();
        });
      })
      .addText((t) => {
        t.setPlaceholder("Or enter custom timezone")
          .setValue("")
          .onChange(async (v) => {
            if (v.trim()) {
              s.arxiv.timezone = v.trim();
              await this.plugin.saveSettings();
            }
          });
      });

    // ─── Output & Schedule ────────────────────────────
    this.sectionHeading(containerEl, "Output & schedule", "schedule");

    new Setting(containerEl)
      .setName("Daily reports folder")
      .setDesc("Folder in this vault for daily report notes (relative path).")
      .addText((t) => {
        t.setValue(s.output.dailyDir);
        t.inputEl.addEventListener("input", () => {
          const validation = validateOutputDirectoryDraft(t.inputEl.value);
          t.inputEl.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
          t.inputEl.toggleClass("is-invalid", !validation.ok);
        });
        t.inputEl.addEventListener("change", () => {
          this.runAction("update daily path", () =>
            this.applyOutputDirectoryDraft("dailyDir", t.inputEl.value, t.inputEl));
        });
      });

    new Setting(containerEl)
      .setName("Paper notes folder")
      .setDesc("Folder in this vault for per-paper notes (relative path).")
      .addText((t) => {
        t.setValue(s.output.papersDir);
        t.inputEl.addEventListener("input", () => {
          const validation = validateOutputDirectoryDraft(t.inputEl.value);
          t.inputEl.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
          t.inputEl.toggleClass("is-invalid", !validation.ok);
        });
        t.inputEl.addEventListener("change", () => {
          this.runAction("update papers path", () =>
            this.applyOutputDirectoryDraft("papersDir", t.inputEl.value, t.inputEl));
        });
      });

    new Setting(containerEl)
      .setName("Link style")
      .setDesc("How links between notes are written in daily reports.")
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
      .setDesc("Language for daily reports and paper notes.")
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
      .setDesc("Local times when automatic runs may start (24-hour clock).");
    renderRunWindowTimeSelect(
      runWindow.controlEl,
      "Start",
      "arxiv-daily-run-window-start",
      s.schedule.runAtLocal,
      (value) => this.saveRunWindowTime("runAtLocal", value),
    );
    renderRunWindowTimeSelect(
      runWindow.controlEl,
      "End",
      "arxiv-daily-run-window-end",
      s.schedule.runUntilLocal,
      (value) => this.saveRunWindowTime("runUntilLocal", value),
    );

    this.attachHelp(
      new Setting(containerEl).setName("Check every (minutes)").addText((t) =>
        t.setValue(String(s.schedule.tickIntervalMin)).onChange(async (v) => {
          s.schedule.tickIntervalMin = Math.max(1, Number(v) || 20);
          await this.plugin.saveSettings();
          this.plugin.restartScheduler();
        }),
      ),
      "How often the plugin looks for a day that still needs a report. Default is 20 minutes.",
    );

    // ─── Email ───────────────────────────────────────────
    this.sectionHeading(containerEl, "Email delivery", "email");

    if (!s.email) {
      s.email = {
        enabled: false,
        mode: "self",
        to: "",
        fromEmail: "",
        fromName: "arXiv Daily",
        apiKey: "",
        hostedToken: "",
        hostedBaseUrl: "",
      };
    }
    if (s.email.mode !== "self" && s.email.mode !== "hosted") {
      s.email.mode = "self";
    }
    const hostedMode = s.email.mode === "hosted";

    this.emailGuide(containerEl, this.emailGuideContent());

    new Setting(containerEl)
      .setName("How to send")
      .setDesc(
        hostedMode
          ? "Official delivery (Beta) is a shared free service with a small daily limit. Prefer Send yourself if you need many messages or reliable high volume."
          : "Send yourself uses your own Resend account (no project quota). Official delivery (Beta) is a limited free option for light personal use.",
      )
      .addDropdown((d) => {
        d.addOption("self", "Send yourself");
        d.addOption("hosted", "Official delivery (Beta)");
        d.setValue(hostedMode ? "hosted" : "self");
        d.onChange(async (value) => {
          s.email.mode = value === "hosted" ? "hosted" : "self";
          await this.plugin.saveSettings();
          this.display();
        });
      });

    new Setting(containerEl)
      .setName("Your email")
      .setDesc(
        hostedMode
          ? "Where verification and daily digests are sent."
          : "Where digests are delivered. With From empty, use the email on your Resend account.",
      )
      .addText((t) => {
        t.setPlaceholder("you@example.com")
          .setValue(s.email.to)
          .onChange(async (v) => {
            s.email.to = v.trim();
            await this.plugin.saveSettings();
          });
      });

    if (hostedMode) {
      new Setting(containerEl)
        .setName("Send verification email")
        .setDesc("Sends a one-time link to confirm this address is yours.")
        .addButton((b) =>
          b.setButtonText("Send verification email").onClick(() => {
            this.runAction("send verification email", async () => {
              const message = await this.plugin.sendHostedVerificationEmail();
              new Notice(message, 10_000);
            });
          }),
        );

      this.renderHostedTokenSetting(containerEl);
    } else {
      this.renderEmailApiKeySetting(containerEl);

      new Setting(containerEl)
        .setName("From email")
        .setDesc(
          "Optional. Leave blank for the simplest setup (mail may only go to your Resend account email). Use an address on a domain you verified in Resend to send more freely.",
        )
        .addText((t) => {
          t.setPlaceholder("Leave blank for simplest setup")
            .setValue(s.email.fromEmail)
            .onChange(async (v) => {
              s.email.fromEmail = v.trim();
              await this.plugin.saveSettings();
            });
        });

      new Setting(containerEl)
        .setName("From name")
        .setDesc('Optional name shown as the sender. Default is "arXiv Daily".')
        .addText((t) => {
          t.setPlaceholder("arXiv Daily")
            .setValue(s.email.fromName ?? "")
            .onChange(async (v) => {
              s.email.fromName = v;
              await this.plugin.saveSettings();
            });
        });
    }

    new Setting(containerEl)
      .setName("Send test email")
      .setDesc(
        hostedMode
          ? "Sends a sample digest now. Needs your email and verification code. Tests count toward the daily limit."
          : "Sends a sample digest now. Needs your email and Resend API key.",
      )
      .addButton((b) =>
        b.setButtonText("Send test").setCta().onClick(() => {
          this.runAction("send test email", async () => {
            const message = await this.plugin.sendTestEmail();
            new Notice(message, 10_000);
          });
        }),
      );

    new Setting(containerEl)
      .setName("Daily auto-send")
      .setDesc(
        hostedMode
          ? "When on, a digest is emailed after each successful daily report. Official delivery may stop for the day if the shared limit is reached; report generation still continues."
          : "When on, a digest is emailed after each successful daily report. Email problems do not stop report generation.",
      )
      .addToggle((t) =>
        t.setValue(s.email.enabled).onChange(async (v) => {
          s.email.enabled = v;
          await this.plugin.saveSettings();
        }),
      );

    // ─── Advanced ─────────────────────────────────────
    this.sectionHeading(containerEl, "Advanced", "advanced");

    this.attachHelp(
      new Setting(containerEl).setName("Log level").addDropdown((d) =>
        d
          .addOption("debug", "Debug")
          .addOption("info", "Info")
          .addOption("warn", "Warn")
          .addOption("error", "Error")
          .setValue(s.advanced.logLevel)
          .onChange(async (value) => {
            if (!isLogLevel(value)) return;
            s.advanced.logLevel = value;
            await this.plugin.saveSettings();
            this.plugin.logger.setLevel(value);
          }),
      ),
      "How much detail appears in the developer console. Use debug only when troubleshooting; info is the default.",
    );

    // ─── Help & feedback ──────────────────────────────
    this.sectionHeading(
      containerEl,
      "Help & feedback",
      "advanced",
      "Documentation and GitHub issues. A short note is enough; do not paste API keys.",
    );

    new Setting(containerEl)
      .setName("Report a bug")
      .setDesc("Opens a blank GitHub issue with the plugin version. A short description is enough.")
      .addButton((b) =>
        b.setButtonText("Open bug report").onClick(() => {
          this.runAction("open bug report", async () => {
            await this.openExternalUrl(this.bugReportUrl());
          });
        }),
      );

    new Setting(containerEl)
      .setName("Request a feature")
      .setDesc("Opens a blank GitHub issue. Write freely.")
      .addButton((b) =>
        b.setButtonText("Open feature request").onClick(() => {
          this.runAction("open feature request", async () => {
            await this.openExternalUrl(buildFeatureRequestUrl());
          });
        }),
      );

    new Setting(containerEl)
      .setName("Documentation")
      .setDesc("Getting started guide on GitHub.")
      .addButton((b) =>
        b.setButtonText("Open docs").onClick(() => {
          this.runAction("open docs", async () => {
            await this.openExternalUrl(ARXIV_DAILY_DOCS_URL);
          });
        }),
      );

    new Setting(containerEl)
      .setName("Repository")
      .setDesc(ARXIV_DAILY_REPO_URL)
      .addButton((b) =>
        b.setButtonText("Open repository").onClick(() => {
          this.runAction("open repository", async () => {
            await this.openExternalUrl(ARXIV_DAILY_REPO_URL);
          });
        }),
      );
  }

  private bugReportUrl(): string {
    return buildBugReportUrl(this.plugin.manifest.version);
  }

  private async openExternalUrl(url: string): Promise<void> {
    await new ObsidianResourceOpener(this.app).openUrl(url);
  }

  private renderEmailApiKeySetting(containerEl: HTMLElement): void {
    const configured = Boolean(this.plugin.settings.email.apiKey?.trim());
    const setting = new Setting(containerEl)
      .setName("Resend API key")
      .setDesc(
        "From your Resend account. Saved only on this device; not shown again after you save.",
      );
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
      this.runAction("save Resend API key", async () => {
        this.plugin.settings.email.apiKey = next;
        await this.plugin.saveSettings();
        reset();
        clear.hidden = false;
      });
    });
    cancel.addEventListener("click", () => {
      if (configured || this.plugin.settings.email.apiKey?.trim()) reset();
      else {
        draft = "";
        input.value = "";
      }
    });
    clear.addEventListener("click", () => {
      this.runAction("clear Resend API key", async () => {
        const confirmed = await this.confirmReplace(
          "Clear the saved Resend API key? Email delivery will stop until a replacement is saved.",
          "Clear",
        );
        if (!confirmed) return;
        this.plugin.settings.email.apiKey = "";
        await this.plugin.saveSettings();
        this.display();
      });
    });
  }

  private renderHostedTokenSetting(containerEl: HTMLElement): void {
    const configured = Boolean(this.plugin.settings.email.hostedToken?.trim());
    const setting = new Setting(containerEl)
      .setName("Verification code")
      .setDesc(
        "After you open the verification link, copy the long code shown on the web page (not the short code in the email link). Use the same email address as above.",
      );
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
      this.runAction("save verification code", async () => {
        const next = draft.replace(/\s+/g, "").trim();
        if (!next) {
          new Notice("Paste the verification code before saving.");
          return;
        }
        this.plugin.settings.email.hostedToken = next;
        await this.plugin.saveSettings();
        this.plugin.refreshSensitiveValues();
        this.display();
      });
    });
    cancel.addEventListener("click", () => {
      if (configured || this.plugin.settings.email.hostedToken?.trim()) reset();
      else {
        draft = "";
        input.value = "";
      }
    });
    clear.addEventListener("click", () => {
      this.runAction("clear verification code", async () => {
        const confirmed = await this.confirmReplace(
          "Clear the verification code? You will need to verify your email again for Official delivery.",
          "Clear",
        );
        if (!confirmed) return;
        this.plugin.settings.email.hostedToken = "";
        await this.plugin.saveSettings();
        this.plugin.refreshSensitiveValues();
        this.display();
      });
    });
  }

  private renderApiKeySetting(containerEl: HTMLElement): void {
    const configured = Boolean(this.plugin.settings.llm.apiKey.trim());
    const setting = new Setting(containerEl)
      .setName("API key")
      .setDesc("Saved only on this device. After saving, the key is hidden.");
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
      this.runAction("save API key", async () => {
        await persistApiKeyChange(
          this.plugin.settings,
          this.plugin.logger,
          () => this.plugin.saveSettings(),
          next,
        );
        this.refreshSetupGuide();
        reset();
        clear.hidden = false;
      });
    });
    cancel.addEventListener("click", () => {
      if (configured || this.plugin.settings.llm.apiKey.trim()) reset();
      else {
        draft = "";
        input.value = "";
      }
    });
    clear.addEventListener("click", () => {
      this.runAction("clear API key", async () => {
        const confirmed = await this.confirmReplace(
          "Clear the saved API key? AI features will stop until you save a new key.",
          "Clear",
        );
        if (!confirmed) return;
        await persistApiKeyChange(
          this.plugin.settings,
          this.plugin.logger,
          () => this.plugin.saveSettings(),
          "",
        );
        this.refreshSetupGuide();
        this.display();
      });
    });
  }

  private renderSetupGuide(containerEl: HTMLElement): void {
    const guide = this.createSetupGuide();
    if (guide) containerEl.appendChild(guide);
  }

  public refreshSetupGuide(): void {
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

  public createSetupGuide(): HTMLElement {
    const status = getSetupStatus(
      this.plugin.settings,
      this.plugin.stateStore.snapshot(),
    );
    const guide = this.containerEl.createEl("section", {
      cls: "arxiv-daily-setup",
      attr: { "aria-labelledby": "arxiv-daily-setup-title" },
    });
    guide.detach();

    if (!shouldRenderSetupGuide(status)) {
      guide.addClass("arxiv-daily-setup--complete");
      const summary = guide.createDiv({
        cls: "arxiv-daily-setup__complete-summary",
      });
      summary.createDiv({
        cls: "arxiv-daily-setup__title",
        text: "Setup complete",
        attr: {
          id: "arxiv-daily-setup-title",
          role: "heading",
          "aria-level": "2",
        },
      });
      summary.createDiv({
        cls: "arxiv-daily-setup__complete-date",
        text: `Latest completed report: ${status.latestCompletedReportDate ?? "Unknown"}`,
      });
      this.renderConfigurationDetails(guide, status.schedulerReasons);
      this.renderDashboardAction(guide);
      return guide;
    }

    const header = guide.createDiv({
      cls: "arxiv-daily-setup__header",
    });
    header.createDiv({
      cls: "arxiv-daily-setup__title",
      text: "Getting started",
      attr: {
        id: "arxiv-daily-setup-title",
        role: "heading",
        "aria-level": "2",
      },
    });
    const completedCount = [
      status.llmReady,
      status.categoriesReady,
      status.topicsReady,
      status.firstReportComplete,
    ].filter(Boolean).length;
    header.createDiv({
      cls: "arxiv-daily-setup__progress-summary",
      text: `${completedCount} of 4 complete`,
      attr: { "aria-live": "polite" },
    });
    const progress = guide.createEl("progress", {
      cls: "arxiv-daily-setup__progress",
      attr: {
        max: "4",
        value: String(completedCount),
        "aria-label": "Setup progress",
      },
    });
    progress.setAttribute("value", String(completedCount));

    const list = guide.createEl("ol", {
      cls: "arxiv-daily-setup__list",
    });
    this.renderSetupItem(
      list,
      status.llmReady,
      "Connect AI",
      "Add an API key, API base URL, and model under AI model.",
      "Connect AI",
      () => this.scrollToSection("llm"),
    );
    this.renderSetupItem(
      list,
      status.categoriesReady,
      "Choose paper sources",
      "Select at least one arXiv category under arXiv.",
      "Choose sources",
      () => this.scrollToSection("arxiv"),
    );
    this.renderSetupItem(
      list,
      status.topicsReady,
      "Describe your research interests",
      "Add at least one complete research topic under Research topics.",
      "Describe interests",
      () => this.scrollToSection("topics"),
    );
    this.renderSetupItem(
      list,
      status.firstReportComplete,
      "Generate your first report",
      status.readyToRun
        ? "Your configuration is ready. Generate a report to finish setup."
        : "Complete the earlier configuration steps before generating a report.",
      status.readyToRun ? "Generate first report" : undefined,
      status.readyToRun
        ? () => {
            this.runAction("generate first report", () => this.generateFirstReport());
          }
        : undefined,
    );

    this.renderConfigurationDetails(guide, status.schedulerReasons);
    this.renderDashboardAction(guide);
    return guide;
  }

  private renderSetupItem(
    parent: HTMLElement,
    done: boolean,
    title: string,
    description: string,
    actionLabel?: string,
    onAction?: () => void,
  ): void {
    const item = parent.createEl("li", {
      cls: `arxiv-daily-setup__item ${done ? "is-done" : "is-pending"}`,
    });
    const body = item.createDiv({ cls: "arxiv-daily-setup__item-body" });
    body.createDiv({
      cls: "arxiv-daily-setup__label",
      text: title,
    });
    body.createDiv({
      cls: "arxiv-daily-setup__description",
      text: description,
    });
    item.createSpan({
      cls: "arxiv-daily-setup__status",
      text: done ? "Complete" : "Next",
    });
    if (!done && actionLabel && onAction) {
      const action = item.createEl("button", {
        cls: "arxiv-daily-setup__link",
        text: actionLabel,
        attr: { type: "button" },
      });
      action.addEventListener("click", onAction);
    }
  }

  private renderConfigurationDetails(
    parent: HTMLElement,
    reasons: readonly string[],
  ): void {
    if (reasons.length === 0) return;
    const details = parent.createEl("details", {
      cls: "arxiv-daily-setup__details",
    });
    details.createEl("summary", { text: "Configuration details" });
    const list = details.createEl("ul");
    for (const reason of reasons) list.createEl("li", { text: reason });
  }

  private renderDashboardAction(parent: HTMLElement): void {
    const actions = parent.createDiv({
      cls: "arxiv-daily-setup__actions",
    });
    const dashboard = actions.createEl("button", {
      text: "Open dashboard",
      attr: { type: "button" },
    });
    dashboard.addEventListener("click", () => {
      this.runAction("open dashboard", () => openDashboardView(this.plugin));
    });
  }

  private scrollToSection(section: "llm" | "arxiv" | "topics" | "schedule" | "advanced"): void {
    const target = this.containerEl.querySelector(
      `[data-arxiv-daily-section="${section}"]`,
    );
    if (!target) return;
    const targetEl = target as HTMLElement;
    const view = targetEl.ownerDocument.defaultView;
    const reduceMotion = view?.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    if (!targetEl.hasAttribute("tabindex")) targetEl.setAttribute("tabindex", "-1");
    targetEl.scrollIntoView({
      block: "start",
      behavior: reduceMotion ? "auto" : "smooth",
    });
    targetEl.focus({ preventScroll: true });
  }

  public async generateFirstReport(): Promise<void> {
    const date = formatDate(
      todayInTz(new Date(), this.plugin.settings.arxiv.timezone),
    );
    this.plugin.logger.info(`settings: first report requested for ${date}`);
    new Notice(`arXiv Daily: running for ${date}…`);
    const result = await this.plugin.scheduler.runForDateNow(date);
    new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
    this.refreshSetupGuide();
  }

  /** Render the topic card for one index into a declarative list row. */
  public renderTopicRow(setting: Setting, index: number): void {
    // Re-renders reuse the same row; drop the previous card first.
    for (const el of Array.from(
      setting.settingEl.querySelectorAll(".arxiv-daily-settings__topic-card"),
    )) {
      el.remove();
    }
    // Scopes the 1.13+ card layout (full-width row, aligned header, grid
    // form) without touching the <1.13 display() styling.
    setting.settingEl.addClass("arxiv-daily-settings__topic-host");
    this.renderTopicCard(
      setting.settingEl,
      this.plugin.settings.arxiv.topics,
      index,
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

    const caret = header.createSpan({
      cls: "arxiv-daily-settings__topic-caret",
      text: isExpanded ? "▾" : "▸",
    });

    const titleSpan = header.createSpan({
      cls: "arxiv-daily-settings__topic-title",
      text: topic.name.trim() || "(unnamed)",
      attr: { title: topic.name },
    });
    titleSpan.toggleClass("is-muted", !topic.name.trim());

    let star: HTMLElement | null = null;
    if (topic.detail) {
      star = header.createSpan({
        cls: "arxiv-daily-settings__topic-star",
        text: "★",
        attr: { title: "Detail report enabled" },
      });
    }

    let tagChip: HTMLElement | null = null;
    if (topic.tag) {
      tagChip = header.createSpan({
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
    nameInput.placeholder = "E.g. Photometric redshift";

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
    tagInput.placeholder = "Kebab-case-slug";
    const autoBadge = tagRow.createSpan({
      cls: "arxiv-daily-settings__topic-auto",
      text: "Auto",
    });
    const refreshAutoBadge = () => {
      autoBadge.toggleClass("is-hidden", topic.tag !== slugify(topic.name));
    };
    refreshAutoBadge();

    const refreshHeader = () => {
      titleSpan.textContent = topic.name.trim() || "(unnamed)";
      titleSpan.title = topic.name;
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
    this.hint(descRow, "Plain-language description of what belongs here. The AI uses this to decide which papers go into this topic.", descHintId);
    const descArea = descRow.createEl("textarea", {
      cls: "arxiv-daily-settings__topic-description",
      attr: { id: descId, "aria-describedby": descHintId },
    });
    descArea.value = topic.description;
    descArea.rows = 3;
    descArea.placeholder =
      "What kinds of papers should be grouped under this topic? (Natural language)";
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
        star = header.createSpan({
          cls: "arxiv-daily-settings__topic-star",
          text: "★",
          attr: { title: "Detail report enabled" },
        });
        // Keep the detail indicator before the tag chip.
        if (tagChip) header.insertBefore(star, tagChip);
      }
    };

    const delBtn = footer.createEl("button", {
      text: "Delete",
      attr: { type: "button" },
    });
    delBtn.classList.add("mod-warning");
    delBtn.onclick = async (e) => {
      e.stopPropagation();
      await this.deleteTopic(index);
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

  public confirmReplace(message: string, confirmLabel = "Replace"): Promise<boolean> {
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

  public showModelDropdown(models: string[], container: HTMLElement): void {
    // Find the existing dropdown in the model setting
    const modelSetting = container.closest(".setting-item");
    if (!modelSetting) return;

    const select = modelSetting.querySelector("select") as HTMLSelectElement;
    if (!select) return;

    // Clear existing options without parsing HTML.
    select.replaceChildren();

    // Add new options
    for (const model of models) {
      select.createEl("option", { value: model, text: model });
    }

    // Pre-select current model if in list
    const currentModel = this.plugin.settings.llm.model;
    if (models.includes(currentModel)) {
      select.value = currentModel;
    } else if (models.length > 0) {
      // Select first model if current not in list
      select.value = models[0]!;
      this.plugin.settings.llm.model = models[0]!;
      this.runAction("save selected model", () => this.plugin.saveSettings());
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

export function renderRunWindowTimeSelect(
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

/** Timezone presets for the arXiv section; shared by display() and the 1.13+ rows. */
export const TIMEZONE_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "Asia/Shanghai", label: "Shanghai (UTC+8)" },
  { value: "Asia/Tokyo", label: "Tokyo (UTC+9)" },
  { value: "US/Eastern", label: "US East (UTC-5)" },
  { value: "US/Pacific", label: "US West (UTC-8)" },
  { value: "Europe/London", label: "London (UTC+0)" },
  { value: "Europe/Berlin", label: "Berlin (UTC+1)" },
  { value: "Europe/Moscow", label: "Moscow (UTC+3)" },
  { value: "Australia/Sydney", label: "Sydney (UTC+10)" },
  { value: "UTC", label: "UTC" },
];

export function addCategoryOptions(
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

function nextCategoryCandidate(existing: string[]): string {  const seen = new Set(existing);
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
