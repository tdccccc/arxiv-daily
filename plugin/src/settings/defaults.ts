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
    categories: ["astro-ph"],
    topics: [],
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
