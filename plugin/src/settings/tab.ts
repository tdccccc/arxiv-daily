import { App, PluginSettingTab, Setting } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { PROVIDER_PRESETS, type ProviderPreset } from "./providers";

export class ArxivDailySettingTab extends PluginSettingTab {
  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    const s = this.plugin.settings;
    containerEl.empty();

    // ─── LLM ──────────────────────────────────────────
    containerEl.createEl("h2", { text: "LLM 配置" });

    // Provider dropdown
    new Setting(containerEl)
      .setName("厂商")
      .setDesc("选择厂商自动填充 URL 和模型，所有字段仍可手动修改")
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
          t.setPlaceholder("或输入其他模型 ID")
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

    new Setting(containerEl).setName("Timeout (秒)").addText((t) =>
      t.setValue(String(s.llm.timeoutMs / 1000)).onChange(async (v) => {
        s.llm.timeoutMs = (Number(v) || 300) * 1000;
        await this.plugin.saveSettings();
      }),
    );

    // Thinking mode — desc varies by provider
    const thinkingDesc = s.llm.provider === "anthropic"
      ? "启用 Anthropic Extended Thinking"
      : s.llm.provider === "deepseek"
        ? "启用推理模式 (DeepSeek V4 系列支持)"
        : "启用推理/思考模式";

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
      .setDesc(s.llm.provider === "anthropic" ? "映射到 thinking budget 档位" : "推理力度")
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
        t.setPlaceholder("或输入自定义值")
          .setValue("")
          .onChange(async (v) => {
            if (v.trim()) {
              s.llm.reasoningEffort = v.trim();
              await this.plugin.saveSettings();
            }
          });
      });

    // ─── arXiv ────────────────────────────────────────
    containerEl.createEl("h2", { text: "arXiv 配置" });

    new Setting(containerEl)
      .setName("分类")
      .setDesc("arXiv 分类，如 astro-ph、cs.LG、hep-ph")
      .addText((t) =>
        t.setValue(s.arxiv.category).onChange(async (v) => {
          s.arxiv.category = v.trim();
          await this.plugin.saveSettings();
        }),
      );

    this.textareaSetting(
      containerEl,
      "研究兴趣",
      "用自然语言描述",
      s.arxiv.researchInterests,
      async (v) => {
        s.arxiv.researchInterests = v;
        await this.plugin.saveSettings();
      },
    );

    this.textareaSetting(
      containerEl,
      "详细收录标准",
      "符合此标准的论文会生成详细报告",
      s.arxiv.detailCriteria,
      async (v) => {
        s.arxiv.detailCriteria = v;
        await this.plugin.saveSettings();
      },
    );

    this.textareaSetting(
      containerEl,
      "允许 detail 的语义分类",
      "一行一个，LLM 输出的语义分类（非 arXiv 官方分类）",
      s.arxiv.detailCategories.join("\n"),
      async (v) => {
        s.arxiv.detailCategories = v
          .split("\n")
          .map((x) => x.trim())
          .filter(Boolean);
        await this.plugin.saveSettings();
      },
    );

    this.textareaSetting(
      containerEl,
      "Category → Tag map (JSON)",
      "",
      JSON.stringify(s.arxiv.categoryTagMap, null, 2),
      async (v) => {
        try {
          s.arxiv.categoryTagMap = JSON.parse(v);
          await this.plugin.saveSettings();
        } catch {
          /* keep last valid */
        }
      },
    );

    this.textareaSetting(
      containerEl,
      "Category → Display name (JSON)",
      "",
      JSON.stringify(s.arxiv.categoryDisplayMap, null, 2),
      async (v) => {
        try {
          s.arxiv.categoryDisplayMap = JSON.parse(v);
          await this.plugin.saveSettings();
        } catch {
          /* keep last valid */
        }
      },
    );

    new Setting(containerEl).setName("时区").addText((t) =>
      t.setValue(s.arxiv.timezone).onChange(async (v) => {
        s.arxiv.timezone = v.trim();
        await this.plugin.saveSettings();
      }),
    );

    // ─── Output & Schedule ────────────────────────────
    containerEl.createEl("h2", { text: "输出 & 调度" });

    new Setting(containerEl)
      .setName("Daily 路径")
      .setDesc("vault 内相对路径")
      .addText((t) =>
        t.setValue(s.output.dailyDir).onChange(async (v) => {
          s.output.dailyDir = v.trim();
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Papers 路径")
      .setDesc("vault 内相对路径")
      .addText((t) =>
        t.setValue(s.output.papersDir).onChange(async (v) => {
          s.output.papersDir = v.trim();
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("启用自动调度")
      .setDesc("启用后立即总结今天（周末跳过，等下个工作日）")
      .addToggle((t) =>
        t.setValue(s.schedule.enabled).onChange(async (v) => {
          await this.plugin.setScheduleEnabled(v);
        }),
      );

    new Setting(containerEl).setName("调度时间 (HH:MM)").addText((t) =>
      t.setValue(s.schedule.runAtLocal).onChange(async (v) => {
        s.schedule.runAtLocal = v.trim();
        await this.plugin.saveSettings();
      }),
    );

    new Setting(containerEl).setName("Tick interval (分钟)").addText((t) =>
      t.setValue(String(s.schedule.tickIntervalMin)).onChange(async (v) => {
        s.schedule.tickIntervalMin = Number(v) || 20;
        await this.plugin.saveSettings();
        this.plugin.restartScheduler();
      }),
    );

    new Setting(containerEl)
      .setName("Lookback 天数")
      .setDesc("最大 5 (受 arXiv /recent 限制)")
      .addText((t) =>
        t.setValue(String(s.schedule.lookbackDays)).onChange(async (v) => {
          s.schedule.lookbackDays = Math.min(5, Math.max(1, Number(v) || 5));
          await this.plugin.saveSettings();
        }),
      );

    // ─── Advanced ─────────────────────────────────────
    containerEl.createEl("h2", { text: "高级" });

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
      "Skip sections (一行一个)",
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
      "Priority sections (一行一个)",
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
