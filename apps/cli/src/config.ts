import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { parse as parseToml } from "smol-toml";
import {
  DEFAULT_SETTINGS,
  arxivCategories,
  normalizeCategoryList,
  sanitizeDetailSelection,
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "@arxiv-daily/core";
import type {
  LinkStyle,
  PluginSettings,
  SummaryLanguage,
  Topic,
} from "@arxiv-daily/core";
import { resolveCliConfigPath } from "./config-path";

export type { LinkStyle } from "@arxiv-daily/core";

/** CLI-only schedule intent for OS timer install (not plugin in-process tick). */
export interface CliScheduleIntent {
  enabled: boolean;
  /** First daily fire HH:MM (machine local for cron). */
  on: string;
  /** 0 = once at `on`; >0 = every N hours from on while <= until. */
  intervalHours: number;
  until: string;
  weekdaysOnly: boolean;
}

export const DEFAULT_CLI_SCHEDULE: CliScheduleIntent = {
  enabled: false,
  on: "09:30",
  intervalHours: 0,
  until: "18:00",
  weekdaysOnly: true,
};

export interface CliRuntimeConfig {
  settings: PluginSettings;
  vaultRoot: string;
  cacheDir: string;
  linkStyle: LinkStyle;
  configPath: string;
  scheduleIntent: CliScheduleIntent;
}

export interface LoadCliConfigOptions {
  /** Override fixed config path (tests only). */
  configPath?: string;
  env?: Record<string, string | undefined>;
  platform?: NodeJS.Platform;
  readText?: (path: string) => Promise<string>;
  homedir?: () => string;
}

export class CliConfigError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "CliConfigError";
  }
}

export async function loadCliConfig(
  opts: LoadCliConfigOptions = {},
): Promise<CliRuntimeConfig> {
  const env = opts.env ?? process.env;
  const platform = opts.platform ?? process.platform;
  const configPath = opts.configPath ?? resolveCliConfigPath(env, platform);
  const readText = opts.readText ?? ((p: string) => fs.readFile(p, "utf8"));

  let raw: string;
  try {
    raw = await readText(configPath);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") {
      throw new CliConfigError(
        `CLI config not found: ${configPath}\nRun: arxiv-daily init`,
      );
    }
    throw new CliConfigError(`failed to read CLI config: ${configPath}`, e);
  }

  let parsed: Record<string, unknown>;
  try {
    const value = parseToml(raw);
    if (!isRecord(value)) {
      throw new Error("config root must be a table");
    }
    parsed = value;
  } catch (e) {
    throw new CliConfigError(`failed to parse CLI config: ${configPath}`, e);
  }

  const settings = mapTomlToSettings(parsed);
  settings.detailSelection = sanitizeDetailSelection(
    detailSelectionPresetBalanced(),
  );

  const home = opts.homedir?.() ?? os.homedir();
  const vaultRoot = resolveUserPath(
    requireString(parsed.vault_root, "vault_root"),
    home,
  );
  const cacheRaw =
    typeof parsed.cache_dir === "string" && parsed.cache_dir.trim()
      ? parsed.cache_dir.trim()
      : path.join(vaultRoot, ".cache", "arxiv-daily");
  const cacheDir = resolveUserPath(cacheRaw, home);

  settings.output.dailyDir = normalizeOutputDirectory(
    "daily_dir",
    settings.output.dailyDir,
  );
  settings.output.papersDir = normalizeOutputDirectory(
    "papers_dir",
    settings.output.papersDir,
  );
  if (
    vaultRelativeDirectoriesCollide(
      settings.output.dailyDir,
      settings.output.papersDir,
    )
  ) {
    throw new CliConfigError("daily_dir and papers_dir must be different");
  }

  const linkStyle = settings.output.linkStyle ?? "wikilink";
  settings.output.linkStyle = linkStyle;
  settings.output.summaryLanguage = normalizeSummaryLanguage(
    settings.output.summaryLanguage ?? "zh",
  );

  return {
    settings,
    vaultRoot,
    cacheDir,
    linkStyle,
    configPath,
    scheduleIntent: mapScheduleIntent(parsed.schedule),
  };
}

function detailSelectionPresetBalanced(): PluginSettings["detailSelection"] {
  return sanitizeDetailSelection({ profile: "balanced" });
}

function mapTomlToSettings(root: Record<string, unknown>): PluginSettings {
  const base = structuredClone(DEFAULT_SETTINGS);
  base.detailSelection = detailSelectionPresetBalanced();

  const llm = asTable(root.llm);
  if (llm) {
    base.llm.apiKey = stringField(llm, "api_key", base.llm.apiKey);
    base.llm.provider = stringField(llm, "provider", base.llm.provider);
    base.llm.baseUrl = stringField(llm, "base_url", base.llm.baseUrl);
    base.llm.model = stringField(llm, "model", base.llm.model);
    if (typeof llm.thinking_mode === "boolean") {
      base.llm.thinkingMode = llm.thinking_mode;
    }
    base.llm.reasoningEffort = stringField(
      llm,
      "reasoning_effort",
      base.llm.reasoningEffort,
    );
  }

  const embedding = asTable(root.embedding);
  if (embedding) {
    const mode = stringField(embedding, "mode", base.embedding.mode);
    base.embedding.mode = mode === "remote" ? "remote" : "local";
    base.embedding.provider = stringField(embedding, "provider", base.embedding.provider);
    base.embedding.baseUrl = stringField(embedding, "base_url", base.embedding.baseUrl);
    base.embedding.apiKey = stringField(embedding, "api_key", base.embedding.apiKey);
    base.embedding.model = stringField(embedding, "model", base.embedding.model);
    const dimension = embedding.dimension;
    if (typeof dimension === "number" && Number.isInteger(dimension) && dimension > 0) {
      base.embedding.dimension = dimension;
    }
  }

  const arxiv = asTable(root.arxiv);
  if (arxiv) {
    if (Array.isArray(arxiv.categories)) {
      base.arxiv.categories = normalizeCategoryList(
        arxiv.categories,
        base.arxiv.categories,
      );
    }
    base.arxiv.timezone = stringField(arxiv, "timezone", base.arxiv.timezone);
    if (Array.isArray(arxiv.topics)) {
      base.arxiv.topics = arxiv.topics.map((item, index) =>
        mapTopic(item, index),
      );
    }
  }
  base.arxiv.category =
    arxivCategories(base.arxiv)[0] ?? base.arxiv.category;

  const output = asTable(root.output);
  if (output) {
    base.output.dailyDir = stringField(output, "daily_dir", base.output.dailyDir);
    base.output.papersDir = stringField(
      output,
      "papers_dir",
      base.output.papersDir,
    );
    const link = stringField(output, "link_style", base.output.linkStyle ?? "wikilink");
    if (link !== "wikilink" && link !== "relative") {
      throw new CliConfigError(`invalid link_style: ${link}`);
    }
    base.output.linkStyle = link;
    base.output.summaryLanguage = normalizeSummaryLanguage(
      stringField(output, "summary_language", base.output.summaryLanguage ?? "zh"),
    );
  }

  const email = asTable(root.email);
  if (email) {
    if (typeof email.enabled === "boolean") base.email.enabled = email.enabled;
    const mode = stringField(email, "mode", base.email.mode);
    if (mode !== "self" && mode !== "hosted") {
      throw new CliConfigError(`invalid email.mode: ${mode}`);
    }
    base.email.mode = mode;
    base.email.to = stringField(email, "to", base.email.to);
    base.email.fromEmail = stringField(email, "from_email", base.email.fromEmail);
    base.email.fromName = stringField(
      email,
      "from_name",
      base.email.fromName ?? "arXiv Daily",
    );
    base.email.apiKey = stringField(email, "api_key", base.email.apiKey ?? "");
    base.email.hostedToken = stringField(
      email,
      "hosted_token",
      base.email.hostedToken ?? "",
    );
    // hosted_base_url intentionally ignored if present
  }

  const advanced = asTable(root.advanced);
  if (advanced) {
    const level = stringField(advanced, "log_level", base.advanced.logLevel);
    if (
      level !== "debug" &&
      level !== "info" &&
      level !== "warn" &&
      level !== "error"
    ) {
      throw new CliConfigError(`invalid log_level: ${level}`);
    }
    base.advanced.logLevel = level;
    if (advanced.request_delay_ms !== undefined) {
      if (
        typeof advanced.request_delay_ms !== "number" ||
        !Number.isFinite(advanced.request_delay_ms) ||
        advanced.request_delay_ms < 0
      ) {
        throw new CliConfigError(
          "invalid advanced.request_delay_ms: expected a non-negative finite number",
        );
      }
      base.advanced.requestDelayMs = Math.max(3_000, advanced.request_delay_ms);
    }
    if (typeof advanced.cache_expiry_days === "number") {
      base.advanced.cacheExpiryDays = advanced.cache_expiry_days;
    }
    if (typeof advanced.section_char_limit === "number") {
      base.advanced.sectionCharLimit = advanced.section_char_limit;
    }
    if (typeof advanced.paper_char_limit === "number") {
      base.advanced.paperCharLimit = advanced.paper_char_limit;
    }
    if (typeof advanced.daily_char_limit === "number") {
      base.advanced.dailyCharLimit = advanced.daily_char_limit;
    }
  }

  // Plugin schedule table ignored for CLI runtime; scheduleIntent is separate.
  base.schedule = { ...DEFAULT_SETTINGS.schedule, enabled: false };

  return base;
}

function mapTopic(raw: unknown, index: number): Topic {
  if (!isRecord(raw)) {
    throw new CliConfigError(`arxiv.topics[${index}] must be a table`);
  }
  const name = stringField(raw, "name", "");
  const tag = stringField(raw, "tag", "");
  const description = stringField(raw, "description", "");
  const detail =
    typeof raw.detail === "boolean" ? raw.detail : true;
  const id =
    typeof raw.id === "string" && raw.id.trim()
      ? raw.id.trim()
      : `topic-${index + 1}`;
  return { id, name, tag, description, detail };
}

function mapScheduleIntent(raw: unknown): CliScheduleIntent {
  const d = { ...DEFAULT_CLI_SCHEDULE };
  if (!isRecord(raw)) return d;
  if (typeof raw.enabled === "boolean") d.enabled = raw.enabled;
  d.on = stringField(raw, "on", d.on);
  if (typeof raw.interval_hours === "number" && Number.isFinite(raw.interval_hours)) {
    d.intervalHours = Math.max(0, Math.floor(raw.interval_hours));
  }
  d.until = stringField(raw, "until", d.until);
  if (typeof raw.weekdays_only === "boolean") d.weekdaysOnly = raw.weekdays_only;
  if (!/^\d{1,2}:\d{2}$/.test(d.on)) {
    throw new CliConfigError(`invalid schedule.on: ${d.on} (use HH:MM)`);
  }
  if (!/^\d{1,2}:\d{2}$/.test(d.until)) {
    throw new CliConfigError(`invalid schedule.until: ${d.until} (use HH:MM)`);
  }
  return d;
}

function normalizeOutputDirectory(name: string, value: unknown): string {
  const result = validateVaultRelativeDirectory(value);
  if (!result.ok || !result.value) {
    throw new CliConfigError(`invalid ${name}: ${result.reason}`);
  }
  return result.value;
}

function normalizeSummaryLanguage(value: unknown): SummaryLanguage {
  if (value === "zh" || value === undefined) return "zh";
  if (value === "en") return "en";
  throw new CliConfigError(`invalid summary_language: ${String(value)}`);
}

function resolveUserPath(input: string, home: string): string {
  const trimmed = input.trim();
  if (!trimmed) throw new CliConfigError("path must not be empty");
  if (trimmed === "~") return home;
  if (trimmed.startsWith("~/") || trimmed.startsWith("~\\")) {
    return path.resolve(home, trimmed.slice(2));
  }
  return path.resolve(trimmed);
}

function requireString(value: unknown, key: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new CliConfigError(`missing required ${key}`);
  }
  return value.trim();
}

function stringField(
  table: Record<string, unknown>,
  key: string,
  fallback: string,
): string {
  const value = table[key];
  if (typeof value === "string") return value;
  return fallback;
}

function asTable(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

/** Expand HH:MM schedule slots for cron generation. */
export function scheduleFireSlots(intent: CliScheduleIntent): string[] {
  const start = parseHm(intent.on);
  const end = parseHm(intent.until);
  if (intent.intervalHours <= 0) {
    return [formatHm(start.h, start.m)];
  }
  const slots: string[] = [];
  let minutes = start.h * 60 + start.m;
  const endMin = end.h * 60 + end.m;
  const step = intent.intervalHours * 60;
  while (minutes <= endMin) {
    const h = Math.floor(minutes / 60) % 24;
    const m = minutes % 60;
    slots.push(formatHm(h, m));
    minutes += step;
    if (slots.length > 48) break;
  }
  return slots;
}

function parseHm(value: string): { h: number; m: number } {
  const [hs, ms] = value.split(":");
  const h = Number(hs);
  const m = Number(ms);
  if (!Number.isInteger(h) || !Number.isInteger(m) || h < 0 || h > 23 || m < 0 || m > 59) {
    throw new CliConfigError(`invalid time ${value}`);
  }
  return { h, m };
}

function formatHm(h: number, m: number): string {
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
