import {
  DEFAULT_SETTINGS,
  migrateArxivSettings,
  type PluginSettings,
  type RunState,
} from "@arxiv-daily/core";

export function settingsAndStateFromPersistedData(raw: unknown): {
  settings: PluginSettings;
  runState: RunState;
} {
  const data = isRecord(raw) ? raw : {};
  const partial = isRecord(data.settings)
    ? data.settings as Partial<PluginSettings>
    : {};
  const merged = mergeSettings(DEFAULT_SETTINGS, partial);
  merged.arxiv = migrateArxivSettings(partial.arxiv);
  return {
    settings: merged,
    runState: isRecord(data.runState) ? data.runState as RunState : {},
  };
}

function mergeSettings(
  defaults: PluginSettings,
  partial: Partial<PluginSettings>,
): PluginSettings {
  return {
    llm: { ...defaults.llm, ...(partial.llm ?? {}) },
    arxiv: { ...defaults.arxiv, ...(partial.arxiv ?? {}) },
    output: { ...defaults.output, ...(partial.output ?? {}) },
    schedule: { ...defaults.schedule, ...(partial.schedule ?? {}) },
    advanced: { ...defaults.advanced, ...(partial.advanced ?? {}) },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
