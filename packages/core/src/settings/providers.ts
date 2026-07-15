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
      { label: "DeepSeek-V4 Pro (rec.)", value: "deepseek-v4-pro" },
      { label: "DeepSeek-V4 Flash", value: "deepseek-v4-flash" },
    ],
    thinkingMode: true,
    reasoningEfforts: ["low", "medium", "high"],
  },
  openai: {
    name: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    models: [
      { label: "GPT-5.5 (rec.)", value: "gpt-5.5" },
      { label: "GPT-5.4", value: "gpt-5.4" },
      { label: "GPT-5.4-mini (budget)", value: "gpt-5.4-mini" },
      { label: "GPT-5.3-codex (coding)", value: "gpt-5.3-codex" },
      { label: "GPT-5.2", value: "gpt-5.2" },
    ],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high", "xhigh"],
  },
  anthropic: {
    name: "Anthropic",
    baseUrl: "https://api.anthropic.com/v1",
    models: [
      { label: "Claude Opus 4.7 (rec.)", value: "claude-opus-4-7" },
      { label: "Claude Sonnet 4.6", value: "claude-sonnet-4-6" },
      { label: "Claude Haiku 4.5", value: "claude-haiku-4-5-20251001" },
    ],
    thinkingMode: true,
    reasoningEfforts: ["low", "medium", "high"],
    anthropicThinking: true,
  },
  zhipu: {
    name: "Zhipu GLM",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    models: [
      { label: "GLM-5.1 (rec.)", value: "glm-5.1" },
      { label: "GLM-5", value: "glm-5" },
      { label: "GLM-5-Turbo", value: "glm-5-turbo" },
      { label: "GLM-4.7", value: "glm-4.7" },
      { label: "GLM-4.7-Flash (free)", value: "glm-4.7-flash" },
    ],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high"],
  },
  custom: {
    name: "Custom",
    baseUrl: "",
    models: [],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high"],
  },
};
