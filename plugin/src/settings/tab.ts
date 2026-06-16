import { App, Modal, PluginSettingTab, Setting, setTooltip } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { PROVIDER_PRESETS, type ProviderPreset } from "./providers";
import { ARXIV_CATEGORIES } from "./arxiv-categories";
import { TOPIC_TEMPLATES } from "./topic-templates";
import type { Topic } from "./types";
import { slugify } from "../utils/slugify";
import { validateFilterConfig } from "./validation";
import { arxivCategories } from "./categories";
import { getSetupStatus } from "../onboarding";
import { executeObsidianCommand, openDashboardView } from "../dashboard/view";

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
  private hint(parent: HTMLElement, text: string): void {
    parent.createEl("div", {
      cls: "arxiv-daily-settings__hint",
      text,
    });
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
    const next = normalized.length > 0 ? normalized : ["astro-ph"];
    this.plugin.settings.arxiv.categories = next;
    this.plugin.settings.arxiv.category = next[0];
    await this.plugin.saveSettings();
  }

  display(): void {
    const { containerEl } = this;
    const s = this.plugin.settings;
    containerEl.empty();
    containerEl.addClass("arxiv-daily-settings");

    this.renderSetupGuide(containerEl);

    // ─── Config-invalid banner (top) ─────────────────
    const v = validateFilterConfig(s);
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

    // Provider dropdown
    new Setting(containerEl)
      .setName("Provider")
      .setDesc("Auto-fills URL and model; all fields remain editable")
      .addDropdown((d) => {
        for (const [key, preset] of Object.entries(PROVIDER_PRESETS)) {
          d.addOption(key, preset.name);
        }
        d.setValue(s.llm.provider).onChange(async (v) => {
          s.llm.provider = v;
          const preset = PROVIDER_PRESETS[v];
          if (preset && v !== "custom") {
            s.llm.baseUrl = preset.baseUrl;
            if (preset.models.length > 0) {
              s.llm.model = preset.models[0].value;
            }
            s.llm.thinkingMode = preset.thinkingMode;
            if (!preset.reasoningEfforts.includes(s.llm.reasoningEffort)) {
              s.llm.reasoningEffort = preset.reasoningEfforts[0];
            }
          }
          await this.plugin.saveSettings();
          this.display(); // re-render to update model/effort UI
        });
      });

    // API Key
    new Setting(containerEl)
      .setName("API Key")
      .setDesc("Required. Your LLM provider's API key. Stored locally in data.json.")
      .addText((t) => {
        t.inputEl.type = "password";
        t.setPlaceholder("sk-...")
          .setValue(s.llm.apiKey)
          .onChange(async (v) => {
            s.llm.apiKey = v;
            await this.plugin.saveSettings();
            this.refreshSetupGuide();
          });
      });

    // Base URL — always editable, auto-filled by provider
    new Setting(containerEl)
      .setName("Base URL")
      .setDesc("LLM endpoint base. Auto-filled when you pick a provider; override for self-hosted or proxy.")
      .addText((t) =>
        t.setValue(s.llm.baseUrl).onChange(async (v) => {
          s.llm.baseUrl = v;
          await this.plugin.saveSettings();
          this.refreshSetupGuide();
        }),
      );

    // Model — dropdown preset + text input for custom override
    const preset = PROVIDER_PRESETS[s.llm.provider];
    if (preset && preset.models.length > 0) {
      new Setting(containerEl)
        .setName("Model")
        .setDesc("Pick a preset or type a custom model ID in the right-hand box.")
        .addDropdown((d) => {
          for (const m of preset.models) {
            d.addOption(m.value, m.label);
          }
            d.setValue(s.llm.model).onChange(async (v) => {
              s.llm.model = v;
              await this.plugin.saveSettings();
              this.refreshSetupGuide();
            });
        })
        .addText((t) => {
          t.setPlaceholder("or enter custom model ID")
            .setValue("")
            .onChange(async (v) => {
              if (v.trim()) {
                s.llm.model = v.trim();
                await this.plugin.saveSettings();
                this.refreshSetupGuide();
              }
            });
        });
    } else {
      new Setting(containerEl)
        .setName("Model")
        .setDesc("Model ID for the custom provider.")
        .addText((t) =>
        t.setValue(s.llm.model).onChange(async (v) => {
          s.llm.model = v;
          await this.plugin.saveSettings();
          this.refreshSetupGuide();
        }),
      );
    }

    this.attachHelp(
      new Setting(containerEl).setName("Temperature").addText((t) =>
        t.setValue(String(s.llm.temperature)).onChange(async (v) => {
          s.llm.temperature = Number(v) || 0;
          await this.plugin.saveSettings();
        }),
      ),
      "Sampling temperature. 0 = deterministic, 1+ = creative. Default 0.3.",
    );

    this.attachHelp(
      new Setting(containerEl).setName("Timeout (sec)").addText((t) =>
        t.setValue(String(s.llm.timeoutMs / 1000)).onChange(async (v) => {
          s.llm.timeoutMs = (Number(v) || 300) * 1000;
          await this.plugin.saveSettings();
        }),
      ),
      "Per-LLM-call timeout in seconds. Raise this if your model is slow (e.g. reasoning models).",
    );

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
    const efforts = preset?.reasoningEfforts ?? ["low", "medium", "high"];
    new Setting(containerEl)
      .setName("Reasoning effort")
      .setDesc(s.llm.provider === "anthropic" ? "Maps to thinking budget tier" : "Reasoning strength")
      .addDropdown((d) => {
        for (const e of efforts) {
          d.addOption(e, e);
        }
        d.setValue(efforts.includes(s.llm.reasoningEffort) ? s.llm.reasoningEffort : efforts[0])
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
      new Setting(containerEl)
        .setName(`Category ${i + 1}`)
        .addDropdown((d) => {
          addCategoryOptions(d.selectEl, categories[i]);
          d.setValue(categories[i]).onChange(async (v) => {
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
        d.addOption("", "Load Template…");
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
          if (s.arxiv.topics.length === 0) {
            await apply();
            return;
          }
          const confirmed = await this.confirmReplace(
            `Replace your ${s.arxiv.topics.length} topic(s) with the "${tpl.name}" template?`,
          );
          if (confirmed) await apply();
        });
      })
      .addButton((b) => {
        b.setButtonText("+ Add Topic").onClick(async () => {
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
        text: "Pick a template above or click + Add Topic to define what to track. The plugin will not call the LLM until at least one topic exists.",
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
      .addText((t) =>
        t.setValue(s.output.dailyDir).onChange(async (v) => {
          s.output.dailyDir = v.trim();
          await this.plugin.saveSettings();
          await this.plugin.reloadStateStoreForOutputPaths();
        }),
      );

    new Setting(containerEl)
      .setName("Papers path")
      .setDesc("Relative path in vault")
      .addText((t) =>
        t.setValue(s.output.papersDir).onChange(async (v) => {
          s.output.papersDir = v.trim();
          await this.plugin.saveSettings();
          await this.plugin.reloadStateStoreForOutputPaths();
        }),
      );

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
      .setName("Run time (HH:MM)")
      .setDesc("Local time the scheduler aims to fire today's batch. Earlier ticks for today are skipped.")
      .addText((t) =>
        t.setValue(s.schedule.runAtLocal).onChange(async (v) => {
          s.schedule.runAtLocal = v.trim();
          await this.plugin.saveSettings();
        }),
      );

    this.attachHelp(
      new Setting(containerEl).setName("Tick interval (min)").addText((t) =>
        t.setValue(String(s.schedule.tickIntervalMin)).onChange(async (v) => {
          s.schedule.tickIntervalMin = Number(v) || 20;
          await this.plugin.saveSettings();
          this.plugin.restartScheduler();
        }),
      ),
      "How often the scheduler interval wakes up to check pending dates. Default 20 minutes.",
    );

    new Setting(containerEl)
      .setName("Lookback days")
      .setDesc("Max 5 (limited by arXiv /recent)")
      .addText((t) =>
        t.setValue(String(s.schedule.lookbackDays)).onChange(async (v) => {
          s.schedule.lookbackDays = Math.min(5, Math.max(1, Number(v) || 5));
          await this.plugin.saveSettings();
        }),
      );

    // ─── Advanced ─────────────────────────────────────
    this.sectionHeading(containerEl, "Advanced", "advanced");

    this.attachHelp(
      new Setting(containerEl).setName("Request delay (ms)").addText((t) =>
        t.setValue(String(s.advanced.requestDelayMs)).onChange(async (v) => {
          s.advanced.requestDelayMs = Number(v) || 3000;
          await this.plugin.saveSettings();
        }),
      ),
      "Pause between HTTP requests to arXiv. Lower = faster fetch but rougher on the server.",
    );

    this.attachHelp(
      new Setting(containerEl).setName("Cache expiry (days)").addText((t) =>
        t.setValue(String(s.advanced.cacheExpiryDays)).onChange(async (v) => {
          s.advanced.cacheExpiryDays = Number(v) || 7;
          await this.plugin.saveSettings();
        }),
      ),
      "How long to keep cached paper HTML on disk before re-fetching.",
    );

    this.attachHelp(
      new Setting(containerEl).setName("Section char limit").addText((t) =>
        t.setValue(String(s.advanced.sectionCharLimit)).onChange(async (v) => {
          s.advanced.sectionCharLimit = Number(v) || 8000;
          await this.plugin.saveSettings();
        }),
      ),
      "Max characters per paper section sent to the LLM. Lower for small-context models.",
    );

    this.attachHelp(
      new Setting(containerEl).setName("Paper char limit").addText((t) =>
        t.setValue(String(s.advanced.paperCharLimit)).onChange(async (v) => {
          s.advanced.paperCharLimit = Number(v) || 50000;
          await this.plugin.saveSettings();
        }),
      ),
      "Max characters of full-text body fed to daily summaries and per-paper detail prompts.",
    );

    this.attachHelp(
      new Setting(containerEl).setName("Daily char limit").addText((t) =>
        t.setValue(String(s.advanced.dailyCharLimit)).onChange(async (v) => {
          s.advanced.dailyCharLimit = Number(v) || 400000;
          await this.plugin.saveSettings();
        }),
      ),
      "When total filtered papers exceed this many chars, split the daily summary into batched LLM calls.",
    );

    this.attachHelp(
      this.textareaSetting(
        containerEl,
        "Skip sections (one per line)",
        "",
        s.advanced.skipSections.join("\n"),
        async (v) => {
          s.advanced.skipSections = v
            .split("\n")
            .map((x) => x.trim())
            .filter(Boolean);
          await this.plugin.saveSettings();
        },
      ),
      "Section headings to drop before sending the paper to the LLM (e.g. References, Acknowledgments). One per line, case-insensitive.",
    );

    this.attachHelp(
      this.textareaSetting(
        containerEl,
        "Priority sections (one per line)",
        "",
        s.advanced.prioritySections.join("\n"),
        async (v) => {
          s.advanced.prioritySections = v
            .split("\n")
            .map((x) => x.trim())
            .filter(Boolean);
          await this.plugin.saveSettings();
        },
      ),
      "Section headings to keep first when trimming to fit the char limit (e.g. Abstract, Conclusion).",
    );

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

  private renderSetupGuide(containerEl: HTMLElement): void {
    containerEl.appendChild(this.createSetupGuide());
  }

  private refreshSetupGuide(): void {
    const current = this.containerEl.querySelector(".arxiv-daily-setup");
    if (current instanceof HTMLElement) {
      current.replaceWith(this.createSetupGuide());
    }
  }

  private createSetupGuide(): HTMLElement {
    const status = getSetupStatus(this.plugin.settings);
    const guide = document.createElement("section");
    guide.addClass("arxiv-daily-setup");
    const header = guide.createEl("div", {
      cls: "arxiv-daily-setup__header",
    });
    header.createEl("div", {
      cls: "arxiv-daily-setup__title",
      text: "Getting Started",
    });
    header.createEl("div", {
      cls: "arxiv-daily-setup__subtitle",
      text: status.readyToRun
        ? "Configuration is ready. Run today or open the Dashboard."
        : "Complete these items before the first run.",
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
      text: "Run Today",
      attr: { type: "button" },
    }) as HTMLButtonElement;
    run.disabled = !status.readyToRun;
    run.addEventListener("click", () => {
      void this.executeCommand("arxiv-daily-run-now");
    });

    const dashboard = actions.createEl("button", {
      text: "Open Dashboard",
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
    const isExpanded = this.expandedTopics.has(topic.id);

    const card = container.createDiv({
      cls: "arxiv-daily-settings__topic-card",
    });

    // ─── Header row (always visible, clickable) ────────────
    const header = card.createDiv({
      cls: "arxiv-daily-settings__topic-header",
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
    form.toggleClass("is-collapsed", !isExpanded);

    // Name row
    const nameRow = form.createDiv({
      cls: "arxiv-daily-settings__topic-row",
    });
    nameRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: "Name",
    });
    this.hint(nameRow, "Heading text used as the section title in the daily report.");
    const nameInput = nameRow.createEl("input", {
      cls: "arxiv-daily-settings__topic-name-input",
      type: "text",
    });
    nameInput.value = topic.name;
    nameInput.placeholder = "e.g. Photometric Redshift";

    // Tag row
    const tagRow = form.createDiv({
      cls: "arxiv-daily-settings__topic-row",
    });
    tagRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: "Tag",
    });
    this.hint(tagRow, "Kebab-case ASCII slug. Written into each paper's YAML frontmatter as an Obsidian #tag.");
    const tagInput = tagRow.createEl("input", {
      cls: "arxiv-daily-settings__topic-tag-input",
      type: "text",
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
    descRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: "Description",
    });
    this.hint(descRow, "Plain-language description of what belongs here. The LLM reads this to decide which papers go into this topic.");
    const descArea = descRow.createEl("textarea", {
      cls: "arxiv-daily-settings__topic-description",
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

    const delBtn = footer.createEl("button", { text: "Delete" });
    delBtn.classList.add("mod-warning");
    delBtn.onclick = async (e) => {
      e.stopPropagation();
      topics.splice(index, 1);
      this.expandedTopics.delete(topic.id);
      await this.plugin.saveSettings();
      this.display();
    };

    // Toggle expand/collapse on header click
    header.onclick = () => {
      if (this.expandedTopics.has(topic.id)) {
        this.expandedTopics.delete(topic.id);
        form.addClass("is-collapsed");
        caret.textContent = "▸";
      } else {
        this.expandedTopics.add(topic.id);
        form.removeClass("is-collapsed");
        caret.textContent = "▾";
      }
    };
  }

  private confirmReplace(message: string): Promise<boolean> {
    return new Promise((resolve) => {
      const modal = new Modal(this.app);
      modal.titleEl.setText("Confirm");
      modal.contentEl.createEl("p", { text: message });
      const btns = modal.contentEl.createDiv({
        cls: "arxiv-daily-modal-button-row",
      });
      const cancel = btns.createEl("button", { text: "Cancel" });
      const ok = btns.createEl("button", { text: "Replace" });
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
