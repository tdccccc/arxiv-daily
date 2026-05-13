import { App, Modal, PluginSettingTab, Setting } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { PROVIDER_PRESETS, type ProviderPreset } from "./providers";
import { ARXIV_CATEGORIES } from "./arxiv-categories";
import { TOPIC_TEMPLATES } from "./topic-templates";
import type { Topic } from "./types";
import { slugify } from "../utils/slugify";

export class ArxivDailySettingTab extends PluginSettingTab {
  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    const s = this.plugin.settings;
    containerEl.empty();

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
    containerEl.createEl("h2", { text: "LLM" });

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
      .addText((t) => {
        t.inputEl.type = "password";
        t.setPlaceholder("sk-...")
          .setValue(s.llm.apiKey)
          .onChange(async (v) => {
            s.llm.apiKey = v;
            await this.plugin.saveSettings();
          });
      });

    // Base URL — always editable, auto-filled by provider
    new Setting(containerEl)
      .setName("Base URL")
      .addText((t) =>
        t.setValue(s.llm.baseUrl).onChange(async (v) => {
          s.llm.baseUrl = v;
          await this.plugin.saveSettings();
        }),
      );

    // Model — dropdown preset + text input for custom override
    const preset = PROVIDER_PRESETS[s.llm.provider];
    if (preset && preset.models.length > 0) {
      new Setting(containerEl)
        .setName("Model")
        .addDropdown((d) => {
          for (const m of preset.models) {
            d.addOption(m.value, m.label);
          }
          d.setValue(s.llm.model).onChange(async (v) => {
            s.llm.model = v;
            await this.plugin.saveSettings();
          });
        })
        .addText((t) => {
          t.setPlaceholder("or enter custom model ID")
            .setValue("")
            .onChange(async (v) => {
              if (v.trim()) {
                s.llm.model = v.trim();
                await this.plugin.saveSettings();
              }
            });
        });
    } else {
      new Setting(containerEl)
        .setName("Model")
        .addText((t) =>
          t.setValue(s.llm.model).onChange(async (v) => {
            s.llm.model = v;
            await this.plugin.saveSettings();
          }),
        );
    }

    new Setting(containerEl).setName("Temperature").addText((t) =>
      t.setValue(String(s.llm.temperature)).onChange(async (v) => {
        s.llm.temperature = Number(v) || 0;
        await this.plugin.saveSettings();
      }),
    );

    new Setting(containerEl).setName("Timeout (sec)").addText((t) =>
      t.setValue(String(s.llm.timeoutMs / 1000)).onChange(async (v) => {
        s.llm.timeoutMs = (Number(v) || 300) * 1000;
        await this.plugin.saveSettings();
      }),
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
    containerEl.createEl("h2", { text: "arXiv" });

    // Category — grouped dropdown + custom text
    new Setting(containerEl)
      .setName("arXiv Category")
      .addDropdown((d) => {
        for (const group of ARXIV_CATEGORIES) {
          const optgroup = d.selectEl.createEl("optgroup");
          optgroup.label = group.label;
          for (const cat of group.categories) {
            const opt = optgroup.createEl("option");
            opt.value = cat.id;
            opt.textContent = `${cat.id} — ${cat.name}`;
          }
        }
        d.setValue(s.arxiv.category).onChange(async (v) => {
          s.arxiv.category = v;
          await this.plugin.saveSettings();
        });
      })
      .addText((t) => {
        t.setPlaceholder("or enter custom category")
          .setValue("")
          .onChange(async (v) => {
            if (v.trim()) {
              s.arxiv.category = v.trim();
              await this.plugin.saveSettings();
            }
          });
      });

    // ─── Research Topics ─────────────────────────────
    containerEl.createEl("h3", { text: "Research Topics" });
    const topicsDesc = containerEl.createEl("div", {
      text: "Each topic becomes one section in the daily report.",
    });
    topicsDesc.style.opacity = "0.7";
    topicsDesc.style.marginBottom = "0.5em";

    const controlsRow = containerEl.createDiv();
    controlsRow.style.display = "flex";
    controlsRow.style.gap = "0.5em";
    controlsRow.style.marginBottom = "0.75em";

    const templateSelect = controlsRow.createEl("select");
    const placeholderOpt = templateSelect.createEl("option");
    placeholderOpt.value = "";
    placeholderOpt.textContent = "Load Template…";
    for (const tpl of TOPIC_TEMPLATES) {
      const opt = templateSelect.createEl("option");
      opt.value = tpl.id;
      opt.textContent = tpl.name;
    }
    templateSelect.onchange = async () => {
      const id = templateSelect.value;
      if (!id) return;
      templateSelect.value = "";
      const tpl = TOPIC_TEMPLATES.find((t) => t.id === id);
      if (!tpl) return;
      const apply = async () => {
        s.arxiv.category = tpl.category;
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
    };

    const addBtn = controlsRow.createEl("button", { text: "+ Add Topic" });
    addBtn.onclick = async () => {
      s.arxiv.topics.push({
        id: crypto.randomUUID(),
        name: "",
        tag: `topic-${s.arxiv.topics.length + 1}`,
        description: "",
        detail: false,
      });
      await this.plugin.saveSettings();
      this.display();
    };

    const topicsContainer = containerEl.createDiv();
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
    containerEl.createEl("h2", { text: "Output & Schedule" });

    new Setting(containerEl)
      .setName("Daily path")
      .setDesc("Relative path in vault")
      .addText((t) =>
        t.setValue(s.output.dailyDir).onChange(async (v) => {
          s.output.dailyDir = v.trim();
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Papers path")
      .setDesc("Relative path in vault")
      .addText((t) =>
        t.setValue(s.output.papersDir).onChange(async (v) => {
          s.output.papersDir = v.trim();
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl).setName("Run time (HH:MM)").addText((t) =>
      t.setValue(s.schedule.runAtLocal).onChange(async (v) => {
        s.schedule.runAtLocal = v.trim();
        await this.plugin.saveSettings();
      }),
    );

    new Setting(containerEl).setName("Tick interval (min)").addText((t) =>
      t.setValue(String(s.schedule.tickIntervalMin)).onChange(async (v) => {
        s.schedule.tickIntervalMin = Number(v) || 20;
        await this.plugin.saveSettings();
        this.plugin.restartScheduler();
      }),
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
    containerEl.createEl("h2", { text: "Advanced" });

    new Setting(containerEl).setName("Request delay (ms)").addText((t) =>
      t.setValue(String(s.advanced.requestDelayMs)).onChange(async (v) => {
        s.advanced.requestDelayMs = Number(v) || 3000;
        await this.plugin.saveSettings();
      }),
    );

    new Setting(containerEl).setName("Cache expiry (days)").addText((t) =>
      t.setValue(String(s.advanced.cacheExpiryDays)).onChange(async (v) => {
        s.advanced.cacheExpiryDays = Number(v) || 7;
        await this.plugin.saveSettings();
      }),
    );

    new Setting(containerEl).setName("Section char limit").addText((t) =>
      t.setValue(String(s.advanced.sectionCharLimit)).onChange(async (v) => {
        s.advanced.sectionCharLimit = Number(v) || 8000;
        await this.plugin.saveSettings();
      }),
    );

    new Setting(containerEl).setName("Paper char limit").addText((t) =>
      t.setValue(String(s.advanced.paperCharLimit)).onChange(async (v) => {
        s.advanced.paperCharLimit = Number(v) || 50000;
        await this.plugin.saveSettings();
      }),
    );

    new Setting(containerEl).setName("Daily char limit").addText((t) =>
      t.setValue(String(s.advanced.dailyCharLimit)).onChange(async (v) => {
        s.advanced.dailyCharLimit = Number(v) || 400000;
        await this.plugin.saveSettings();
      }),
    );

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
    );

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
    );

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
    );
  }

  private renderTopicCard(container: HTMLElement, topics: Topic[], index: number): void {
    const topic = topics[index];
    const card = container.createDiv();
    card.style.border = "1px solid var(--background-modifier-border)";
    card.style.borderRadius = "6px";
    card.style.padding = "0.75em";
    card.style.marginBottom = "0.75em";

    // Name row
    const nameRow = card.createDiv();
    nameRow.style.marginBottom = "0.5em";
    const nameLabel = nameRow.createEl("label", { text: "Name" });
    nameLabel.style.cssText = "display:block;font-weight:600;margin-bottom:0.25em;";
    const nameInput = nameRow.createEl("input", { type: "text" });
    nameInput.value = topic.name;
    nameInput.style.width = "100%";
    nameInput.placeholder = "e.g. Photometric Redshift";

    // Tag row
    const tagRow = card.createDiv();
    tagRow.style.marginBottom = "0.5em";
    const tagLabel = tagRow.createEl("label", { text: "Tag" });
    tagLabel.style.cssText = "display:block;font-weight:600;margin-bottom:0.25em;";
    const tagInput = tagRow.createEl("input", { type: "text" });
    tagInput.value = topic.tag;
    tagInput.style.width = "60%";
    tagInput.placeholder = "kebab-case-slug";
    const autoBadge = tagRow.createEl("span", { text: "  Auto" });
    autoBadge.style.cssText = "opacity:0.5;font-size:0.85em;margin-left:0.5em;";
    const refreshAutoBadge = () => {
      autoBadge.style.display = topic.tag === slugify(topic.name) ? "" : "none";
    };
    refreshAutoBadge();

    nameInput.oninput = async () => {
      const wasAuto = topic.tag === slugify(topic.name);
      topic.name = nameInput.value;
      if (wasAuto) {
        const derived = slugify(topic.name);
        topic.tag = derived || `topic-${index + 1}`;
        tagInput.value = topic.tag;
      }
      refreshAutoBadge();
      await this.plugin.saveSettings();
    };

    tagInput.oninput = async () => {
      topic.tag = tagInput.value;
      refreshAutoBadge();
      await this.plugin.saveSettings();
    };

    // Description row
    const descRow = card.createDiv();
    descRow.style.marginBottom = "0.5em";
    const descLabel = descRow.createEl("label", { text: "Description" });
    descLabel.style.cssText = "display:block;font-weight:600;margin-bottom:0.25em;";
    const descArea = descRow.createEl("textarea");
    descArea.value = topic.description;
    descArea.rows = 3;
    descArea.style.width = "100%";
    descArea.placeholder =
      "What kinds of papers should be grouped under this topic? (natural language)";
    descArea.oninput = async () => {
      topic.description = descArea.value;
      await this.plugin.saveSettings();
    };

    // Footer: detail toggle + delete
    const footer = card.createDiv();
    footer.style.display = "flex";
    footer.style.justifyContent = "space-between";
    footer.style.alignItems = "center";

    const detailLabel = footer.createEl("label");
    detailLabel.style.cursor = "pointer";
    const detailCheckbox = detailLabel.createEl("input", { type: "checkbox" });
    detailCheckbox.checked = topic.detail;
    detailCheckbox.style.marginRight = "0.4em";
    detailLabel.appendText("Detail report");
    detailCheckbox.onchange = async () => {
      topic.detail = detailCheckbox.checked;
      await this.plugin.saveSettings();
    };

    const delBtn = footer.createEl("button", { text: "Delete" });
    delBtn.onclick = async () => {
      topics.splice(index, 1);
      await this.plugin.saveSettings();
      this.display();
    };
  }

  private confirmReplace(message: string): Promise<boolean> {
    return new Promise((resolve) => {
      const modal = new Modal(this.app);
      modal.titleEl.setText("Confirm");
      modal.contentEl.createEl("p", { text: message });
      const btns = modal.contentEl.createDiv();
      btns.style.display = "flex";
      btns.style.justifyContent = "flex-end";
      btns.style.gap = "0.5em";
      btns.style.marginTop = "0.75em";
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
  ) {
    new Setting(container)
      .setName(name)
      .setDesc(desc)
      .addTextArea((t) => {
        t.setValue(value).onChange((v) => onChange(v));
        t.inputEl.rows = 6;
        t.inputEl.style.width = "100%";
      });
  }
}
