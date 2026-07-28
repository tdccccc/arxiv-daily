import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import {
  ARXIV_CATEGORIES,
  DEFAULT_SETTINGS,
  PROVIDER_PRESETS,
  startHostedEmailVerification,
} from "@arxiv-daily/core";
import { buildNodeHostAdapters } from "@arxiv-daily/node-runtime";
import { resolveCliConfigPath } from "./config-path";
import type { WritableTextStream } from "./main-types";

export interface InitOptions {
  env?: Record<string, string | undefined>;
  platform?: NodeJS.Platform;
  configPath?: string;
  /** Injected answers for tests: async (prompt) => line */
  ask?: (prompt: string) => Promise<string>;
  writeFile?: (path: string, body: string) => Promise<void>;
  readFile?: (path: string) => Promise<string>;
  mkdir?: (path: string) => Promise<void>;
  stdout?: WritableTextStream;
  stderr?: WritableTextStream;
  isTTY?: boolean;
  /** Injected for tests; defaults to real hosted verify call. */
  startHostedVerify?: (email: string) => Promise<void>;
}

const COMMON_TIMEZONES = [
  "Asia/Shanghai",
  "Asia/Tokyo",
  "Asia/Singapore",
  "Asia/Kolkata",
  "Europe/London",
  "Europe/Berlin",
  "Europe/Paris",
  "America/New_York",
  "America/Los_Angeles",
  "America/Chicago",
  "UTC",
] as const;

export async function runInit(opts: InitOptions = {}): Promise<number> {
  const env = opts.env ?? process.env;
  const platform = opts.platform ?? process.platform;
  const configPath = opts.configPath ?? resolveCliConfigPath(env, platform);
  const write =
    opts.writeFile ??
    ((p: string, body: string) => fs.writeFile(p, body, "utf8"));
  const read =
    opts.readFile ?? ((p: string) => fs.readFile(p, "utf8"));
  const mkdir =
    opts.mkdir ??
    ((p: string) => fs.mkdir(p, { recursive: true }).then(() => undefined));
  const stdout = opts.stdout ?? process.stdout;
  const stderr = opts.stderr ?? process.stderr;
  const tty =
    opts.isTTY ??
    (Boolean((input as NodeJS.ReadStream).isTTY) &&
      Boolean((output as NodeJS.WriteStream).isTTY));

  if (!tty && !opts.ask) {
    writeLine(stderr, "init requires an interactive terminal");
    return 2;
  }

  writeLine(stdout, "");
  writeLine(stdout, "arXiv Daily CLI setup");
  writeLine(stdout, `Config file: ${configPath}`);
  writeLine(
    stdout,
    "Press Enter to accept values in [brackets]. Type ? on some steps for more help.",
  );
  writeLine(stdout, "");

  let existing: string | null = null;
  try {
    existing = await read(configPath);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }

  let mode: "write" | "merge" | "cancel" = "write";
  if (existing !== null) {
    writeLine(stdout, `A config already exists at:\n  ${configPath}`);
    const choice = (
      await prompt(
        opts,
        "  [o]verwrite all  [m]erge (keep existing file)  [c]ancel [o]: ",
      )
    )
      .trim()
      .toLowerCase();
    if (choice === "c" || choice === "cancel") {
      writeLine(stdout, "init cancelled");
      return 0;
    }
    if (choice === "m" || choice === "merge") mode = "merge";
    else mode = "write";
    writeLine(stdout, "");
  }

  // --- Paths ---
  writeLine(stdout, "── Paths ──");
  writeLine(
    stdout,
    "Vault root = folder where daily reports are written (Obsidian vault root, or any directory).",
  );
  const vaultRoot = (
    await promptUntil(
      opts,
      `Vault root path (absolute, e.g. /home/you/Notes) []: `,
      (v) => (v.trim() ? null : "Path is required."),
    )
  ).trim();

  // --- LLM ---
  writeLine(stdout, "");
  writeLine(stdout, "── AI / LLM ──");
  writeLine(
    stdout,
    "Used to filter and summarize papers. Need an API key from your provider.",
  );
  const providerKeys = Object.keys(PROVIDER_PRESETS);
  writeLine(stdout, "Providers:");
  providerKeys.forEach((key, i) => {
    const p = PROVIDER_PRESETS[key]!;
    writeLine(stdout, `  ${i + 1}) ${p.name}  (${key})`);
  });
  const providerAns = (
    await prompt(
      opts,
      `Choose provider 1–${providerKeys.length} or id [${DEFAULT_SETTINGS.llm.provider}]: `,
    )
  ).trim();
  const providerId = resolveProviderId(providerAns, providerKeys);
  const preset = PROVIDER_PRESETS[providerId] ?? PROVIDER_PRESETS.custom!;
  const baseUrlDefault = preset.baseUrl || DEFAULT_SETTINGS.llm.baseUrl;
  const modelDefault =
    preset.models[0]?.value || DEFAULT_SETTINGS.llm.model;

  writeLine(
    stdout,
    "API key is stored in the config file (plaintext). Do not commit this file.",
  );
  const apiKey = (
    await promptUntil(
      opts,
      "LLM API key []: ",
      (v) => (v.trim() ? null : "API key is required for run commands."),
    )
  ).trim();

  if (preset.models.length > 0 && providerId !== "custom") {
    writeLine(stdout, "Models for this provider:");
    preset.models.forEach((m, i) => {
      writeLine(stdout, `  ${i + 1}) ${m.label}  →  ${m.value}`);
    });
  }
  const modelAns = (
    await prompt(opts, `Model name or number [${modelDefault}]: `)
  ).trim();
  const model = resolveModel(modelAns, preset.models, modelDefault);

  const baseUrl = (
    await prompt(opts, `API base URL [${baseUrlDefault}]: `)
  ).trim() || baseUrlDefault;

  // --- Email (optional) ---
  writeLine(stdout, "");
  writeLine(stdout, "── Email (optional) ──");
  writeLine(
    stdout,
    "After a successful daily run, send a short digest. Skip if you only want local files.",
  );
  writeLine(stdout, "  1) Skip for now (default)");
  writeLine(stdout, "  2) Send yourself (your Resend API key)");
  writeLine(stdout, "  3) Official delivery Beta (verify email; shared quota)");
  const emailChoice = (
    await prompt(opts, "Email setup [1]: ")
  ).trim() || "1";

  let emailEnabled = false;
  let emailMode: "self" | "hosted" = "self";
  let emailTo = "";
  let emailApiKey = "";
  let hostedToken = "";

  if (emailChoice === "2") {
    emailMode = "self";
    writeLine(
      stdout,
      "Create a key at https://resend.com — free tier is fine for personal digests.",
    );
    writeLine(
      stdout,
      "With From empty, you can usually only send TO your Resend account email.",
    );
    emailTo = (
      await promptUntil(
        opts,
        "Your inbox (To) []: ",
        (v) => (v.trim() ? null : "Email address required."),
      )
    ).trim();
    emailApiKey = (
      await promptUntil(
        opts,
        "Resend API key (re_…) []: ",
        (v) => (v.trim() ? null : "Resend API key required for Send yourself."),
      )
    ).trim();
    const enableNow = (
      await prompt(
        opts,
        "Turn on daily auto-send after runs? (test first recommended) [y/N]: ",
      )
    )
      .trim()
      .toLowerCase();
    emailEnabled = enableNow === "y" || enableNow === "yes";
  } else if (emailChoice === "3") {
    emailMode = "hosted";
    writeLine(
      stdout,
      "Official delivery: we email you a link; open it, copy the LONG code from the web page,",
    );
    writeLine(
      stdout,
      "then paste it here. Shared free Beta — a few messages per inbox per UTC day.",
    );
    emailTo = (
      await promptUntil(
        opts,
        "Email to verify (To) []: ",
        (v) => (v.trim() ? null : "Email address required."),
      )
    ).trim();

    const sendNow = (
      await prompt(opts, "Send verification email now? [Y/n]: ")
    )
      .trim()
      .toLowerCase();
    if (sendNow !== "n" && sendNow !== "no") {
      try {
        const start =
          opts.startHostedVerify ??
          (async (email: string) => {
            const host = buildNodeHostAdapters({ rootDir: process.cwd() });
            await startHostedEmailVerification({
              http: host.http,
              email,
            });
          });
        await start(emailTo);
        writeLine(
          stdout,
          `Sent. Check ${emailTo} (and spam). Open the link, copy the long code from the page.`,
        );
      } catch (e) {
        writeLine(
          stderr,
          `Could not send verification email: ${(e as Error).message}`,
        );
        writeLine(
          stdout,
          "You can finish later: arxiv-daily email verify-start",
        );
      }
    } else {
      writeLine(
        stdout,
        "Skipped send. Later: arxiv-daily email verify-start",
      );
    }
    hostedToken = (
      await prompt(
        opts,
        "Paste verification code (long token), or Enter to fill later: ",
      )
    )
      .trim()
      .replace(/\s+/g, "");
    if (hostedToken) {
      const enableNow = (
        await prompt(
          opts,
          "Turn on daily auto-send after runs? (test first recommended) [y/N]: ",
        )
      )
        .trim()
        .toLowerCase();
      emailEnabled = enableNow === "y" || enableNow === "yes";
    } else {
      writeLine(
        stdout,
        "Token empty — set email.hosted_token in config after verify, then email test.",
      );
    }
  }

  // --- arXiv ---
  writeLine(stdout, "");
  writeLine(stdout, "── arXiv sources ──");
  writeLine(
    stdout,
    "Categories = which arXiv boards to fetch. You can pick from the list or type ids.",
  );
  const categories = await pickCategories(opts, stdout);
  writeLine(stdout, `Selected: ${categories.join(", ")}`);

  writeLine(stdout, "");
  writeLine(stdout, "Timezone for “today” and schedule dates (IANA name).");
  COMMON_TIMEZONES.forEach((tz, i) => {
    writeLine(stdout, `  ${i + 1}) ${tz}`);
  });
  const tzAns = (
    await prompt(
      opts,
      `Timezone number or name [${DEFAULT_SETTINGS.arxiv.timezone}]: `,
    )
  ).trim();
  const timezone = resolveTimezone(tzAns);

  writeLine(stdout, "");
  writeLine(stdout, "Language for daily reports and paper notes.");
  writeLine(stdout, "  1) zh (Chinese)");
  writeLine(stdout, "  2) en (English)");
  const langAns = (await prompt(opts, "Language [1]: ")).trim() || "1";
  const summaryLanguage =
    langAns === "2" || langAns.toLowerCase() === "en" ? "en" : "zh";

  // --- Topic placeholder ---
  writeLine(stdout, "");
  writeLine(stdout, "── Research topic (placeholder) ──");
  writeLine(
    stdout,
    "Daily reports group papers by topic. Edit [[arxiv.topics]] in the config anytime",
  );
  writeLine(
    stdout,
    "(name / tag / description). Description quality matters most for filtering.",
  );
  const topicName = (
    await prompt(opts, 'Topic display name [My research (edit me)]: ')
  ).trim() || "My research (edit me)";
  const topicTag = (
    await prompt(opts, "Topic tag slug [my-research]: ")
  ).trim() || "my-research";
  const topicDescription = (
    await prompt(
      opts,
      "Topic description (what papers belong here; Enter for placeholder): ",
    )
  ).trim() ||
    "Describe in natural language what papers belong in this topic (problems, methods, objects; what to exclude).";

  const cacheDir = path.join(vaultRoot, ".cache", "arxiv-daily");
  const body = renderInitToml({
    vaultRoot,
    cacheDir,
    apiKey,
    baseUrl,
    model,
    provider: providerId,
    thinkingMode: preset.thinkingMode,
    categories,
    timezone,
    summaryLanguage,
    topic: {
      name: topicName,
      tag: topicTag,
      description: topicDescription,
    },
    email: {
      enabled: emailEnabled,
      mode: emailMode,
      to: emailTo,
      apiKey: emailApiKey,
      hostedToken,
    },
  });

  let finalBody = body;
  if (mode === "merge" && existing) {
    finalBody = mergeTomlPreferExisting(existing, body);
  }

  await mkdir(path.dirname(configPath));
  await write(configPath, finalBody);

  writeLine(stdout, "");
  writeLine(stdout, `Wrote ${configPath}`);
  writeLine(stdout, "");
  writeLine(stdout, "Next steps:");
  writeLine(
    stdout,
    "  1. Edit [[arxiv.topics]] description if the placeholder is still generic.",
  );
  writeLine(stdout, "  2. arxiv-daily run --today");
  if (emailTo && !emailEnabled) {
    writeLine(
      stdout,
      "  3. Optional: arxiv-daily email test   then set email.enabled = true",
    );
  }
  writeLine(
    stdout,
    "  Optional: set [schedule] enabled = true → arxiv-daily schedule install",
  );
  writeLine(stdout, "");
  return 0;
}

async function pickCategories(
  opts: InitOptions,
  stdout: WritableTextStream,
): Promise<string[]> {
  writeLine(stdout, "Category groups:");
  ARXIV_CATEGORIES.forEach((g, i) => {
    writeLine(stdout, `  ${i + 1}) ${g.label}  (${g.categories.length} ids)`);
  });
  writeLine(stdout, "  0) Type category ids myself (e.g. cs.LG, astro-ph)");
  const groupAns = (
    await prompt(
      opts,
      `Group number(s), comma-separated, or 0 [1]: `,
    )
  ).trim() || "1";

  if (groupAns === "0") {
    const raw = (
      await prompt(
        opts,
        `Category ids (comma-separated) [${DEFAULT_SETTINGS.arxiv.categories.join(", ")}]: `,
      )
    ).trim();
    return raw
      ? raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)
      : [...DEFAULT_SETTINGS.arxiv.categories];
  }

  const groupIndexes = parseIndexList(groupAns, ARXIV_CATEGORIES.length);
  const pool: Array<{ id: string; name: string; group: string }> = [];
  for (const gi of groupIndexes) {
    const g = ARXIV_CATEGORIES[gi - 1];
    if (!g) continue;
    for (const c of g.categories) {
      pool.push({ id: c.id, name: c.name, group: g.label });
    }
  }
  if (pool.length === 0) {
    return [...DEFAULT_SETTINGS.arxiv.categories];
  }

  writeLine(stdout, "Categories in selected group(s):");
  pool.forEach((c, i) => {
    writeLine(stdout, `  ${i + 1}) ${c.id.padEnd(14)} ${c.name}`);
  });
  writeLine(
    stdout,
    "Enter numbers (e.g. 1,3,5), a single id, or Enter for first item only.",
  );
  const pickAns = (
    await prompt(opts, `Select [1]: `)
  ).trim() || "1";

  // Free-form id(s) without numbers?
  if (/[a-z]/i.test(pickAns) && !/^\d/.test(pickAns)) {
    return pickAns.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
  }
  const indexes = parseIndexList(pickAns, pool.length);
  const selected = indexes
    .map((i) => pool[i - 1]?.id)
    .filter((id): id is string => Boolean(id));
  return selected.length > 0
    ? [...new Set(selected)]
    : [...DEFAULT_SETTINGS.arxiv.categories];
}

function parseIndexList(raw: string, max: number): number[] {
  const parts = raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
  const out: number[] = [];
  for (const p of parts) {
    const n = Number(p);
    if (Number.isInteger(n) && n >= 1 && n <= max && !out.includes(n)) {
      out.push(n);
    }
  }
  return out;
}

function resolveProviderId(ans: string, keys: string[]): string {
  if (!ans) return DEFAULT_SETTINGS.llm.provider;
  const asNum = Number(ans);
  if (Number.isInteger(asNum) && asNum >= 1 && asNum <= keys.length) {
    return keys[asNum - 1]!;
  }
  const lower = ans.toLowerCase();
  if (keys.includes(lower)) return lower;
  if (keys.includes(ans)) return ans;
  return DEFAULT_SETTINGS.llm.provider;
}

function resolveModel(
  ans: string,
  models: Array<{ label: string; value: string }>,
  fallback: string,
): string {
  if (!ans) return fallback;
  const asNum = Number(ans);
  if (
    Number.isInteger(asNum) &&
    asNum >= 1 &&
    asNum <= models.length
  ) {
    return models[asNum - 1]!.value;
  }
  return ans;
}

function resolveTimezone(ans: string): string {
  if (!ans) return DEFAULT_SETTINGS.arxiv.timezone;
  const asNum = Number(ans);
  if (
    Number.isInteger(asNum) &&
    asNum >= 1 &&
    asNum <= COMMON_TIMEZONES.length
  ) {
    return COMMON_TIMEZONES[asNum - 1]!;
  }
  return ans;
}

async function promptUntil(
  opts: InitOptions,
  message: string,
  validate: (value: string) => string | null,
): Promise<string> {
  for (;;) {
    const value = await prompt(opts, message);
    const err = validate(value);
    if (!err) return value;
    writeLine(opts.stderr ?? process.stderr, `  ${err}`);
  }
}

async function prompt(
  opts: InitOptions,
  message: string,
): Promise<string> {
  if (opts.ask) return opts.ask(message);
  const rl = readline.createInterface({ input, output });
  try {
    return await rl.question(message);
  } finally {
    rl.close();
  }
}

function writeLine(stream: WritableTextStream, line: string): void {
  stream.write(`${line}\n`);
}

export function renderInitToml(input: {
  vaultRoot: string;
  cacheDir: string;
  apiKey: string;
  baseUrl: string;
  model: string;
  provider: string;
  thinkingMode: boolean;
  categories: string[];
  timezone: string;
  summaryLanguage: string;
  topic: { name: string; tag: string; description: string };
  email: {
    enabled: boolean;
    mode: "self" | "hosted";
    to: string;
    apiKey: string;
    hostedToken: string;
  };
}): string {
  const cats = input.categories.map((c) => tomlString(c)).join(", ");
  return `# =============================================================================
# arXiv Daily — CLI config
# Path: ~/.config/arxiv-daily/config.toml  (or $XDG_CONFIG_HOME/...)
# May contain secrets — do not commit or share this file publicly.
#
# Editing tips (or hand to an agent):
#   - Research focus: [arxiv].categories and [[arxiv.topics]]
#   - Do not invent service URLs; do not add hosted_base_url
#   - Keep key names; only replace secret values
# Run: arxiv-daily run --today
# =============================================================================

schema_version = 1

# Folder where daily/ and papers/ are written (absolute path recommended)
vault_root = ${tomlString(input.vaultRoot)}
# Fetch cache (safe to delete; will be recreated)
cache_dir = ${tomlString(input.cacheDir)}

[llm]
api_key = ${tomlString(input.apiKey)}
base_url = ${tomlString(input.baseUrl)}
model = ${tomlString(input.model)}
# Preset id (deepseek, openai, anthropic, zhipu, custom, ...)
provider = ${tomlString(input.provider)}
thinking_mode = ${input.thinkingMode}
reasoning_effort = ${tomlString(DEFAULT_SETTINGS.llm.reasoningEffort)}

[arxiv]
# arXiv category ids, e.g. ["astro-ph"], ["cs.LG", "cs.AI"]
categories = [${cats}]
# IANA timezone for "today" (e.g. Asia/Shanghai, America/New_York, UTC)
timezone = ${tomlString(input.timezone)}

# Duplicate the whole [[arxiv.topics]] block to add more topics.
# Clearer description → better filtering.
[[arxiv.topics]]
name = ${tomlString(input.topic.name)}
tag = ${tomlString(input.topic.tag)}
description = ${tomlString(input.topic.description)}
# true = this topic may get longer paper notes under papers/
detail = true

[output]
# "zh" or "en"
summary_language = ${tomlString(input.summaryLanguage)}
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
link_style = "wikilink"

[email]
# true = email a digest after a successful daily run (failures never fail the report)
enabled = ${input.email.enabled}
# "self" = your Resend API key; "hosted" = Official delivery Beta (verification token)
mode = ${tomlString(input.email.mode)}
to = ${tomlString(input.email.to)}
# Leave empty for Resend test sender (often only delivers to your Resend account email)
from_email = ""
from_name = "arXiv Daily"
# mode = "self"
api_key = ${tomlString(input.email.apiKey)}
# mode = "hosted" — long code from the verification web page
hosted_token = ${tomlString(input.email.hostedToken)}

[schedule]
# Defaults only; init does not ask. Set enabled = true then: arxiv-daily schedule install
enabled = false
# First daily fire (machine local time HH:MM)
on = "09:30"
# 0 = once per day at on; e.g. 4 = every 4 hours from on until until
interval_hours = 0
until = "18:00"
weekdays_only = true

[advanced]
# debug | info | warn | error
log_level = "info"
`;
}

function tomlString(value: string): string {
  return JSON.stringify(value);
}

function mergeTomlPreferExisting(existing: string, generated: string): string {
  if (/^\s*vault_root\s*=/m.test(existing)) {
    return `${existing.trimEnd()}\n\n# --- init merge: kept existing file; re-run with overwrite to replace ---\n`;
  }
  return generated;
}
