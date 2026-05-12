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
      { label: "DeepSeek-Chat", value: "deepseek-chat" },
      { label: "DeepSeek-Reasoner", value: "deepseek-reasoner" },
    ],
    thinkingMode: true,
    reasoningEfforts: ["low", "medium", "high"],
  },
  openai: {
    name: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    models: [
      { label: "GPT-4o", value: "gpt-4o" },
      { label: "GPT-4o-mini", value: "gpt-4o-mini" },
      { label: "GPT-4.1", value: "gpt-4.1" },
      { label: "GPT-4.1-mini", value: "gpt-4.1-mini" },
      { label: "o3", value: "o3" },
      { label: "o3-mini", value: "o3-mini" },
      { label: "o4-mini", value: "o4-mini" },
    ],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high"],
  },
  anthropic: {
    name: "Anthropic",
    baseUrl: "https://api.anthropic.com/v1",
    models: [
      { label: "Claude Sonnet 4.6 (推荐)", value: "claude-sonnet-4-6" },
      { label: "Claude Opus 4.6", value: "claude-opus-4-6" },
      { label: "Claude Haiku 4.5", value: "claude-haiku-4-5-20251001" },
    ],
    thinkingMode: false,
    reasoningEfforts: ["low", "medium", "high"],
    anthropicThinking: true,
  },
  zhipu: {
    name: "智谱 GLM",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    models: [
      { label: "GLM-4-Plus (推荐)", value: "glm-4-plus" },
      { label: "GLM-4-Flash", value: "glm-4-flash" },
      { label: "GLM-Z1-Flash", value: "glm-z1-flash" },
      { label: "GLM-Z1-Air", value: "glm-z1-air" },
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
