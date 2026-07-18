export const DETAIL_SELECTION_PROFILES = [
  "conservative",
  "balanced",
  "broad",
  "custom",
] as const;

export type DetailSelectionProfile = (typeof DETAIL_SELECTION_PROFILES)[number];
export type DetailSelectionPresetProfile = Exclude<DetailSelectionProfile, "custom">;

export interface DetailSelectionSettings {
  profile: DetailSelectionProfile;
  normalThreshold: number;
  exceptionalThreshold: number;
  softLimit: number;
}

export const DETAIL_SELECTION_PRESETS: Readonly<
  Record<DetailSelectionPresetProfile, Readonly<DetailSelectionSettings>>
> = Object.freeze({
  conservative: Object.freeze({
    profile: "conservative",
    normalThreshold: 85,
    exceptionalThreshold: 95,
    softLimit: 1,
  }),
  balanced: Object.freeze({
    profile: "balanced",
    normalThreshold: 75,
    exceptionalThreshold: 92,
    softLimit: 3,
  }),
  broad: Object.freeze({
    profile: "broad",
    normalThreshold: 65,
    exceptionalThreshold: 88,
    softLimit: 5,
  }),
});

export const DEFAULT_DETAIL_SELECTION: Readonly<DetailSelectionSettings> =
  DETAIL_SELECTION_PRESETS.balanced;

export function isDetailSelectionProfile(
  value: unknown,
): value is DetailSelectionProfile {
  return typeof value === "string"
    && (DETAIL_SELECTION_PROFILES as readonly string[]).includes(value);
}

/** Return a fresh settings object for one of the named policies. */
export function detailSelectionPreset(
  profile: DetailSelectionPresetProfile,
): DetailSelectionSettings {
  return { ...DETAIL_SELECTION_PRESETS[profile] };
}

/**
 * Sanitize persisted, UI, or CLI detail-selection values without throwing.
 * Missing fields use the selected preset (or balanced for custom/invalid data).
 */
export function sanitizeDetailSelection(
  raw: unknown,
): DetailSelectionSettings {
  if (!isRecord(raw) || !isDetailSelectionProfile(raw.profile)) {
    return detailSelectionPreset("balanced");
  }

  const profile = raw.profile;
  if (profile !== "custom") {
    // A named profile is a canonical policy, not a label for arbitrary values.
    return detailSelectionPreset(profile);
  }

  const fallback = DETAIL_SELECTION_PRESETS.balanced;
  const normalThreshold = finiteClamped(
    raw.normalThreshold,
    fallback.normalThreshold,
    0,
    100,
  );
  const exceptionalThreshold = Math.max(
    normalThreshold,
    finiteClamped(
      raw.exceptionalThreshold,
      fallback.exceptionalThreshold,
      0,
      100,
    ),
  );
  const softLimit = integerClamped(raw.softLimit, fallback.softLimit, 0, 20);

  return { profile, normalThreshold, exceptionalThreshold, softLimit };
}

function finiteClamped(
  value: unknown,
  fallback: number,
  min: number,
  max: number,
): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(max, Math.max(min, value))
    : fallback;
}

function integerClamped(
  value: unknown,
  fallback: number,
  min: number,
  max: number,
): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(max, Math.max(min, Math.round(value)))
    : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
