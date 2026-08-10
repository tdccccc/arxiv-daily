import {
  DEFAULT_SETTINGS,
  migrateArxivSettings,
  migrateEmailSettings,
  sanitizeDetailSelection,
  type PluginSettings,
  type RunState,
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "@arxiv-daily/core";

export function settingsAndStateFromPersistedData(raw: unknown): {
  settings: PluginSettings;
  runState: RunState;
  warnings: string[];
} {
  const data = isRecord(raw) ? raw : {};
  const partial = isRecord(data.settings)
    ? data.settings as Partial<PluginSettings>
    : {};
  const merged = mergeSettings(DEFAULT_SETTINGS, partial);
  merged.arxiv = migrateArxivSettings(partial.arxiv);
  merged.email = migrateEmailSettings(partial.email);
  const warnings = sanitizePersistedOutputDirectories(merged);
  return {
    settings: merged,
    runState: isRecord(data.runState) ? data.runState as RunState : {},
    warnings,
  };
}

function mergeSettings(
  defaults: PluginSettings,
  partial: Partial<PluginSettings>,
): PluginSettings {
  return {
    llm: { ...defaults.llm, ...(partial.llm ?? {}) },
    arxiv: { ...defaults.arxiv, ...(partial.arxiv ?? {}) },
    detailSelection: sanitizeDetailSelection(partial.detailSelection),
    output: { ...defaults.output, ...(partial.output ?? {}) },
    schedule: { ...defaults.schedule, ...(partial.schedule ?? {}) },
    advanced: { ...defaults.advanced, ...(partial.advanced ?? {}) },
    email: migrateEmailSettings(partial.email ?? defaults.email),
    embedding: { ...defaults.embedding, ...(partial.embedding ?? {}) },
  };
}

function sanitizePersistedOutputDirectories(settings: PluginSettings): string[] {
  const warnings: string[] = [];
  for (const field of ["dailyDir", "papersDir"] as const) {
    const validation = validateVaultRelativeDirectory(settings.output[field]);
    if (validation.ok && validation.value) {
      settings.output[field] = validation.value;
      continue;
    }
    settings.output[field] = DEFAULT_SETTINGS.output[field];
    warnings.push(
      `Ignored unsafe persisted output.${field} (${validation.reason}); restored default`,
    );
  }
  if (
    vaultRelativeDirectoriesCollide(
      settings.output.dailyDir,
      settings.output.papersDir,
    )
  ) {
    settings.output.dailyDir = DEFAULT_SETTINGS.output.dailyDir;
    settings.output.papersDir = DEFAULT_SETTINGS.output.papersDir;
    warnings.push(
      "Persisted output directories collided; restored both defaults",
    );
  }
  return warnings;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
