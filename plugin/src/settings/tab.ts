import { App, PluginSettingTab, Setting } from "obsidian";
import type ArxivDailyPlugin from "../../main";

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

    new Setting(containerEl)
      .setName("API Key")
      .setDesc("OpenAI 兼容 API Key (DeepSeek / OpenAI / 其他)")
      .addText((t) => {
        t.inputEl.type = "password";
        t.setPlaceholder("sk-...")
          .setValue(s.llm.apiKey)
          .onChange(async (v) => {
            s.llm.apiKey = v;
            await this.plugin.saveSettings();
          });
      });

    new Setting(containerEl)
      .setName("Base URL")
      .setDesc("API 端点")
      .addText((t) =>
        t.setValue(s.llm.baseUrl).onChange(async (v) => {
          s.llm.baseUrl = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Model")
      .setDesc(
        "推荐 deepseek-v4-pro；可选 deepseek-v4-flash / deepseek-chat (将弃用) / deepseek-reasoner (将弃用)",
      )
      .addText((t) =>
        t.setValue(s.llm.model).onChange(async (v) => {
          s.llm.model = v;
          await this.plugin.saveSettings();
        }),
      );

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

    new Setting(containerEl)
      .setName("Thinking mode")
      .setDesc("启用推理模式 (DeepSeek V4 系列支持)")
      .addToggle((t) =>
        t.setValue(s.llm.thinkingMode).onChange(async (v) => {
          s.llm.thinkingMode = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl).setName("Reasoning effort").addDropdown((d) =>
      d
        .addOption("low", "low")
        .addOption("medium", "medium")
        .addOption("high", "high")
        .setValue(s.llm.reasoningEffort)
        .onChange(async (v) => {
          s.llm.reasoningEffort = v as any;
          await this.plugin.saveSettings();
        }),
    );

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

    new Setting(containerEl).setName("启用自动调度").addToggle((t) =>
      t.setValue(s.schedule.enabled).onChange(async (v) => {
        s.schedule.enabled = v;
        await this.plugin.saveSettings();
        this.plugin.restartScheduler();
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
