import type { PluginSettings } from "./types";

export const DEFAULT_SETTINGS: PluginSettings = {
  llm: {
    apiKey: "",
    provider: "deepseek",
    baseUrl: "https://api.deepseek.com/v1",
    model: "deepseek-v4-pro",
    temperature: 0.3,
    timeoutMs: 300_000,
    thinkingMode: true,
    reasoningEffort: "high",
  },
  arxiv: {
    category: "astro-ph",
    researchInterests:
      "1. 星系光度红移估计 (photometric redshift / photo-z)：方法、目录、比较\n" +
      "2. 星系团 (galaxy clusters)：搜寻、质量标定、目录、SZ/X-ray/光学巡天\n" +
      "3. 天文中的 ML/DL 应用：深度学习、模拟推断 (SBI) 等",
    detailCriteria:
      "- Photo-z 方法论文（提出或比较 photo-z 方法/目录）\n" +
      "- 星系团巡天/目录/质量标定论文",
    detailCategories: ["photo-z", "galaxy-cluster"],
    categoryTagMap: {
      "photo-z": "photo-z",
      "galaxy-cluster": "galaxy-cluster",
      "ml": "ml",
    },
    categoryDisplayMap: {
      "galaxy-cluster": "Galaxy Cluster 相关",
      "photo-z": "Photo-z 相关",
      "ml": "ML 相关",
      "other": "其他",
    },
    timezone: "Asia/Shanghai",
  },
  output: {
    dailyDir: "arxiv-daily/daily",
    papersDir: "arxiv-daily/papers",
  },
  schedule: {
    enabled: false,
    runAtLocal: "09:30",
    tickIntervalMin: 20,
    lookbackDays: 5,
  },
  advanced: {
    requestDelayMs: 3000,
    cacheExpiryDays: 7,
    sectionCharLimit: 8000,
    paperCharLimit: 50_000,
    dailyCharLimit: 400_000,
    skipSections: [
      "reference",
      "bibliography",
      "appendix",
      "acknowledgement",
      "acknowledgment",
      "author contribution",
      "data availability",
      "conflict of interest",
      "orcid",
    ],
    prioritySections: ["abstract", "conclusion", "summary"],
    logLevel: "info",
  },
};
