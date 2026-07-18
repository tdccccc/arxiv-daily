import * as fs from "node:fs/promises";
import * as path from "node:path";
import {
  DEFAULT_SETTINGS,
  detailSelectionPreset,
  isDetailSelectionProfile,
  sanitizeDetailSelection,
} from "@arxiv-daily/core";
import type {
  LinkStyle,
  PluginSettings,
  SummaryLanguage,
  Topic,
} from "@arxiv-daily/core";
import {
  arxivCategories,
  normalizeCategoryList,
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "@arxiv-daily/core";

export type { LinkStyle } from "@arxiv-daily/core";

export interface CliRuntimeConfig {
  settings: PluginSettings;
  vaultRoot: string;
  cacheDir: string;
  linkStyle: LinkStyle;
  configPath: string | null;
}

export interface LoadCliConfigOptions {
  cwd?: string;
  configPath?: string;
  env?: Record<string, string | undefined>;
  readText?: (path: string) => Promise<string>;
}

type PartialPluginSettings = {
  llm?: Partial<PluginSettings["llm"]>;
  arxiv?: Partial<PluginSettings["arxiv"]>;
  detailSelection?: Partial<PluginSettings["detailSelection"]>;
  output?: Partial<PluginSettings["output"]>;
  schedule?: Partial<PluginSettings["schedule"]>;
  advanced?: Partial<PluginSettings["advanced"]>;
};

export class CliConfigError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "CliConfigError";
  }
}

const DEFAULT_CONFIG_FILE = "arxiv-daily.config.json";

export async function loadCliConfig(
  opts: LoadCliConfigOptions = {},
): Promise<CliRuntimeConfig> {
  const cwd = path.resolve(opts.cwd ?? process.cwd());
  const env = opts.env ?? process.env;
  const file = await readConfigObject(opts, cwd);
  const envConfig = configFromEnv(env);
  const fileSettings = settingsObject(file);

  const settings = applyPartialSettings(
    applyPartialSettings(DEFAULT_SETTINGS, fileSettings),
    envConfig.settings,
  );
  settings.detailSelection = sanitizeDetailSelection(settings.detailSelection);
  const vaultRoot = path.resolve(
    cwd,
    envConfig.vaultRoot ?? stringOr(file.vaultRoot, "."),
  );
  const cacheDir = path.resolve(
    cwd,
    envConfig.cacheDir ?? stringOr(file.cacheDir, ".arxiv-daily/cache"),
  );
  const linkStyle = normalizeLinkStyle(
    envConfig.linkStyle ??
      stringOr(file.linkStyle, settings.output.linkStyle ?? "wikilink"),
  );
  settings.output.linkStyle = linkStyle;
  settings.output.summaryLanguage = normalizeSummaryLanguageSetting(
    settings.output.summaryLanguage ?? "zh",
  );
  settings.output.dailyDir = normalizeOutputDirectory(
    "dailyDir",
    settings.output.dailyDir,
  );
  settings.output.papersDir = normalizeOutputDirectory(
    "papersDir",
    settings.output.papersDir,
  );
  if (vaultRelativeDirectoriesCollide(
    settings.output.dailyDir,
    settings.output.papersDir,
  )) {
    throw new CliConfigError("dailyDir and papersDir must be different");
  }

  return {
    settings,
    vaultRoot,
    cacheDir,
    linkStyle,
    configPath: file.__configPath ?? null,
  };
}

interface RawCliConfig {
  settings?: unknown;
  llm?: unknown;
  arxiv?: unknown;
  detailSelection?: unknown;
  output?: unknown;
  schedule?: unknown;
  advanced?: unknown;
  vaultRoot?: unknown;
  cacheDir?: unknown;
  linkStyle?: unknown;
  __configPath?: string;
}

interface EnvCliConfig {
  settings: PartialPluginSettings;
  vaultRoot?: string;
  cacheDir?: string;
  linkStyle?: string;
}

async function readConfigObject(
  opts: LoadCliConfigOptions,
  cwd: string,
): Promise<RawCliConfig> {
  const configPath = opts.configPath
    ? path.resolve(cwd, opts.configPath)
    : path.join(cwd, DEFAULT_CONFIG_FILE);
  const explicit = Boolean(opts.configPath);
  const readText = opts.readText ?? ((p: string) => fs.readFile(p, "utf8"));

  let raw: string;
  try {
    raw = await readText(configPath);
  } catch (e) {
    if (!explicit && (e as NodeJS.ErrnoException).code === "ENOENT") {
      return {};
    }
    throw new CliConfigError(`failed to read CLI config: ${configPath}`, e);
  }

  try {
    const parsed = JSON.parse(raw);
    if (!isRecord(parsed)) {
      throw new Error("config root must be an object");
    }
    return { ...parsed, __configPath: configPath };
  } catch (e) {
    throw new CliConfigError(`failed to parse CLI config: ${configPath}`, e);
  }
}

function settingsObject(file: RawCliConfig): PartialPluginSettings {
  const nested = isRecord(file.settings) ? file.settings : {};
  return {
    ...(nested as Partial<PluginSettings>),
    ...(isRecord(file.llm) ? { llm: file.llm } : {}),
    ...(isRecord(file.arxiv) ? { arxiv: file.arxiv } : {}),
    ...(isRecord(file.detailSelection)
      ? { detailSelection: file.detailSelection }
      : {}),
    ...(isRecord(file.output) ? { output: file.output } : {}),
    ...(isRecord(file.schedule) ? { schedule: file.schedule } : {}),
    ...(isRecord(file.advanced) ? { advanced: file.advanced } : {}),
  } as PartialPluginSettings;
}

function configFromEnv(env: Record<string, string | undefined>): EnvCliConfig {
  const settings: PartialPluginSettings = {};
  const llm: Record<string, unknown> = {};
  const arxiv: Record<string, unknown> = {};
  const detailSelection: Record<string, unknown> = {};
  const output: Record<string, unknown> = {};
  const advanced: Record<string, unknown> = {};

  setString(llm, "apiKey", firstEnv(env, "ARXIV_DAILY_API_KEY", "ARXIV_DAILY_LLM_API_KEY"));
  setString(llm, "provider", env.ARXIV_DAILY_PROVIDER);
  setString(llm, "baseUrl", env.ARXIV_DAILY_BASE_URL);
  setString(llm, "model", env.ARXIV_DAILY_MODEL);
  setNumber(llm, "temperature", env.ARXIV_DAILY_TEMPERATURE);
  setNumber(llm, "timeoutMs", env.ARXIV_DAILY_TIMEOUT_MS);
  setBoolean(llm, "thinkingMode", env.ARXIV_DAILY_THINKING_MODE);
  setString(llm, "reasoningEffort", env.ARXIV_DAILY_REASONING_EFFORT);

  setString(arxiv, "category", env.ARXIV_DAILY_CATEGORY);
  if (env.ARXIV_DAILY_CATEGORIES) {
    arxiv.categories = env.ARXIV_DAILY_CATEGORIES.split(",");
  }
  setString(arxiv, "timezone", env.ARXIV_DAILY_TIMEZONE);
  if (env.ARXIV_DAILY_TOPICS_JSON) {
    arxiv.topics = parseTopicsJson(env.ARXIV_DAILY_TOPICS_JSON);
  }

  setString(detailSelection, "profile", env.ARXIV_DAILY_DETAIL_PROFILE);
  setSanitizedNumber(
    detailSelection,
    "normalThreshold",
    env.ARXIV_DAILY_DETAIL_NORMAL_THRESHOLD,
  );
  setSanitizedNumber(
    detailSelection,
    "exceptionalThreshold",
    env.ARXIV_DAILY_DETAIL_EXCEPTIONAL_THRESHOLD,
  );
  setSanitizedNumber(
    detailSelection,
    "softLimit",
    env.ARXIV_DAILY_DETAIL_SOFT_LIMIT,
  );

  setString(output, "dailyDir", env.ARXIV_DAILY_DAILY_DIR);
  setString(output, "papersDir", env.ARXIV_DAILY_PAPERS_DIR);
  setString(output, "summaryLanguage", env.ARXIV_DAILY_SUMMARY_LANGUAGE);
  setString(advanced, "logLevel", env.ARXIV_DAILY_LOG_LEVEL);

  if (Object.keys(llm).length > 0) settings.llm = llm;
  if (Object.keys(arxiv).length > 0) {
    settings.arxiv = arxiv;
  }
  if (Object.keys(detailSelection).length > 0) {
    settings.detailSelection = detailSelection;
  }
  if (Object.keys(output).length > 0) {
    settings.output = output;
  }
  if (Object.keys(advanced).length > 0) {
    settings.advanced = advanced;
  }

  return {
    settings,
    vaultRoot: env.ARXIV_DAILY_VAULT_ROOT,
    cacheDir: env.ARXIV_DAILY_CACHE_DIR,
    linkStyle: env.ARXIV_DAILY_LINK_STYLE,
  };
}

function applyPartialSettings(
  base: PluginSettings,
  partial: PartialPluginSettings,
): PluginSettings {
  const next: PluginSettings = {
    llm: { ...base.llm, ...(partial.llm ?? {}) },
    arxiv: { ...base.arxiv, ...(partial.arxiv ?? {}) },
    detailSelection: mergeDetailSelection(base.detailSelection, partial.detailSelection),
    output: { ...base.output, ...(partial.output ?? {}) },
    schedule: { ...base.schedule, ...(partial.schedule ?? {}) },
    advanced: { ...base.advanced, ...(partial.advanced ?? {}) },
  };
  const rawArxiv = partial.arxiv as Record<string, unknown> | undefined;
  if (rawArxiv && Object.prototype.hasOwnProperty.call(rawArxiv, "categories")) {
    next.arxiv.categories = normalizeCategoryList(
      rawArxiv.categories,
      arxivCategories(base.arxiv),
    );
  } else if (
    rawArxiv &&
    Object.prototype.hasOwnProperty.call(rawArxiv, "category")
  ) {
    next.arxiv.categories = normalizeCategoryList(
      rawArxiv.category,
      arxivCategories(base.arxiv),
    );
  }
  next.arxiv.category = arxivCategories(next.arxiv)[0] ?? base.arxiv.category;
  return next;
}

function mergeDetailSelection(
  base: PluginSettings["detailSelection"],
  partial: Partial<PluginSettings["detailSelection"]> | undefined,
): PluginSettings["detailSelection"] {
  if (!partial || !isRecord(partial)) return sanitizeDetailSelection(base);
  const hasProfile = Object.prototype.hasOwnProperty.call(partial, "profile");
  if (hasProfile && !isDetailSelectionProfile(partial.profile)) {
    return sanitizeDetailSelection(partial);
  }

  const hasNumericOverride = [
    "normalThreshold",
    "exceptionalThreshold",
    "softLimit",
  ].some((key) => Object.prototype.hasOwnProperty.call(partial, key));
  const requestedProfile = hasProfile ? partial.profile! : base.profile;

  // A profile-only layer explicitly selects its exact preset. Any layer with a
  // numeric field is an override and therefore becomes custom, even when its
  // values happen to equal a preset. This makes file/env precedence explicit.
  if (hasProfile && requestedProfile !== "custom" && !hasNumericOverride) {
    return detailSelectionPreset(requestedProfile);
  }

  const profileBase = hasProfile && requestedProfile !== "custom"
    ? detailSelectionPreset(requestedProfile)
    : base;
  return sanitizeDetailSelection({
    ...profileBase,
    ...partial,
    profile: hasNumericOverride ? "custom" : requestedProfile,
  });
}

function normalizeOutputDirectory(name: string, value: unknown): string {
  const result = validateVaultRelativeDirectory(value);
  if (!result.ok || !result.value) {
    throw new CliConfigError(`invalid ${name}: ${result.reason}`);
  }
  return result.value;
}

function normalizeLinkStyle(value: string): LinkStyle {
  if (value === "wikilink" || value === "relative") return value;
  throw new CliConfigError(`invalid linkStyle: ${value}`);
}

function normalizeSummaryLanguageSetting(value: unknown): SummaryLanguage {
  if (value === "zh" || value === undefined) return "zh";
  if (value === "en") return "en";
  throw new CliConfigError(`invalid summaryLanguage: ${String(value)}`);
}

function parseTopicsJson(value: string): Topic[] {
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) throw new Error("topics must be an array");
    return parsed as Topic[];
  } catch (e) {
    throw new CliConfigError("failed to parse ARXIV_DAILY_TOPICS_JSON", e);
  }
}

function setString(
  target: Record<string, unknown>,
  key: string,
  value: string | undefined,
): void {
  if (value !== undefined && value !== "") target[key] = value;
}

function setNumber(
  target: Record<string, unknown>,
  key: string,
  value: string | undefined,
): void {
  if (value === undefined || value === "") return;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new CliConfigError(`invalid numeric env value for ${key}: ${value}`);
  }
  target[key] = parsed;
}

function setSanitizedNumber(
  target: Record<string, unknown>,
  key: string,
  value: string | undefined,
): void {
  if (value === undefined || value === "") return;
  target[key] = Number(value);
}

function setBoolean(
  target: Record<string, unknown>,
  key: string,
  value: string | undefined,
): void {
  if (value === undefined || value === "") return;
  target[key] = /^(1|true|yes|on)$/i.test(value);
}

function firstEnv(
  env: Record<string, string | undefined>,
  ...keys: string[]
): string | undefined {
  return keys.map((key) => env[key]).find((value) => value);
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
