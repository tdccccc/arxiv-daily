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
    linkStyle: "wikilink",
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
    sectionCharLimit: 16000,
    paperCharLimit: 100_000,
    dailyCharLimit: 400_000,
    logLevel: "info",
  },
};
