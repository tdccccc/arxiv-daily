/** Interface preferences, independent of research output language and host storage. */
export interface UiAppearancePreferences {
  theme: "light" | "dark" | "system";
  language: "zh" | "en";
}
export const DEFAULT_UI_APPEARANCE: Readonly<UiAppearancePreferences> = Object.freeze({ theme: "light", language: "zh" });

export function validateUiAppearancePreferences(value: unknown): value is UiAppearancePreferences {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const entry = value as Record<string, unknown>;
  return Object.keys(entry).length === 2 && Object.hasOwn(entry, "theme") && Object.hasOwn(entry, "language")
    && (entry.theme === "light" || entry.theme === "dark" || entry.theme === "system")
    && (entry.language === "zh" || entry.language === "en");
}

/** Normalize persisted host data; API mutations must validate before calling this. */
export function normalizeUiAppearancePreferences(value: unknown): UiAppearancePreferences {
  return validateUiAppearancePreferences(value) ? { theme: value.theme, language: value.language } : { ...DEFAULT_UI_APPEARANCE };
}
