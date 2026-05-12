export interface ProviderPreset {
  name: string;
  baseUrl: string;
  models: Array<{ label: string; value: string }>;
  thinkingMode: boolean;
  reasoningEfforts: string[];
  /** If true, Anthropic-style extended thinking (budget_tokens) instead of reasoning_effort */
  anthropicThinking?: boolean;
}

export const PROVIDER_PRESETS: Record<string, ProviderPreset> = {
  deepseek: {
    name: "DeepSeek",
    baseUrl: "https://api.deepseek.com/v1",
    models: [
      { label: "DeepSeek-V4 Pro (推荐)", value: "deepseek-v4-pro" },
      { label: "DeepSeek-V4 Flash", value: "deepseek-v4-flash" },
    ],
    thinkingMode: true,
    reasoningEfforts: ["low", "medium", "high"],
  },
  openai: {
    name: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    models: [
      { label: "GPT-5.5 (推荐)", value: "gpt-5.5" },
      { label: "GPT-5.4", value: "gpt-5.4" },
      { label: "GPT-5.4-mini (便宜)", value: "gpt-5.4-mini" },
      { label: "GPT-5.3-codex (代码)", value: "gpt-5.3-codex" },
      { label: "GPT-5.2", value: "gpt-5.2" },
    ],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high", "xhigh"],
  },
  anthropic: {
    name: "Anthropic",
    baseUrl: "https://api.anthropic.com/v1",
    models: [
      { label: "Claude Opus 4.7 (推荐)", value: "claude-opus-4-7" },
      { label: "Claude Sonnet 4.6", value: "claude-sonnet-4-6" },
      { label: "Claude Haiku 4.5", value: "claude-haiku-4-5-20251001" },
    ],
    thinkingMode: true,
    reasoningEfforts: ["low", "medium", "high"],
    anthropicThinking: true,
  },
  zhipu: {
    name: "智谱 GLM",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    models: [
      { label: "GLM-5.1 (推荐)", value: "glm-5.1" },
      { label: "GLM-5", value: "glm-5" },
      { label: "GLM-5-Turbo", value: "glm-5-turbo" },
      { label: "GLM-4.7", value: "glm-4.7" },
      { label: "GLM-4.7-Flash (免费)", value: "glm-4.7-flash" },
    ],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high"],
  },
  custom: {
    name: "自定义",
    baseUrl: "",
    models: [],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high"],
  },
};
