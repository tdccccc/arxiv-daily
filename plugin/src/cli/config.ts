import * as fs from "node:fs/promises";
import * as path from "node:path";
import { DEFAULT_SETTINGS } from "../settings/defaults";
import type {
  LinkStyle,
  PluginSettings,
  SummaryLanguage,
  Topic,
} from "../settings/types";
import {
  arxivCategories,
  normalizeCategoryList,
} from "../settings/categories";

export type { LinkStyle } from "../settings/types";

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
    ...(isRecord(file.output) ? { output: file.output } : {}),
    ...(isRecord(file.schedule) ? { schedule: file.schedule } : {}),
    ...(isRecord(file.advanced) ? { advanced: file.advanced } : {}),
  } as PartialPluginSettings;
}

function configFromEnv(env: Record<string, string | undefined>): EnvCliConfig {
  const settings: PartialPluginSettings = {};
  const llm: Record<string, unknown> = {};
  const arxiv: Record<string, unknown> = {};
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

  setString(output, "dailyDir", env.ARXIV_DAILY_DAILY_DIR);
  setString(output, "papersDir", env.ARXIV_DAILY_PAPERS_DIR);
  setString(output, "summaryLanguage", env.ARXIV_DAILY_SUMMARY_LANGUAGE);
  setString(advanced, "logLevel", env.ARXIV_DAILY_LOG_LEVEL);

  if (Object.keys(llm).length > 0) settings.llm = llm;
  if (Object.keys(arxiv).length > 0) {
    settings.arxiv = arxiv;
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
