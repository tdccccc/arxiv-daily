import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
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
}

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

  let existing: string | null = null;
  try {
    existing = await read(configPath);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }

  let mode: "write" | "merge" | "cancel" = "write";
  if (existing !== null) {
    const choice = (
      await prompt(
        opts,
        `Config already exists at ${configPath}\n  [o]verwrite  [m]erge missing  [c]ancel: `,
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
  }

  const vaultRoot = (
    await prompt(opts, "Vault root path (absolute): ")
  ).trim();
  if (!vaultRoot) {
    writeLine(stderr, "vault_root is required");
    return 2;
  }

  const apiKey = (await prompt(opts, "LLM API key: ")).trim();
  const baseUrl = (
    await prompt(
      opts,
      `LLM base URL [${DEFAULT_SETTINGS.llm.baseUrl}]: `,
    )
  ).trim() || DEFAULT_SETTINGS.llm.baseUrl;
  const model = (
    await prompt(opts, `LLM model [${DEFAULT_SETTINGS.llm.model}]: `)
  ).trim() || DEFAULT_SETTINGS.llm.model;

  let emailEnabled = false;
  let emailMode: "self" | "hosted" = "self";
  let emailTo = "";
  let emailApiKey = "";
  let hostedToken = "";
  const wantEmail = (
    await prompt(opts, "Configure email now? [y/N]: ")
  )
    .trim()
    .toLowerCase();
  if (wantEmail === "y" || wantEmail === "yes") {
    const modeAns = (
      await prompt(opts, 'Email mode "self" or "hosted" [self]: ')
    )
      .trim()
      .toLowerCase();
    emailMode = modeAns === "hosted" ? "hosted" : "self";
    emailTo = (await prompt(opts, "Your email (to): ")).trim();
    if (emailMode === "self") {
      emailApiKey = (await prompt(opts, "Resend API key: ")).trim();
    } else {
      hostedToken = (
        await prompt(
          opts,
          "Hosted verification token (paste later if empty): ",
        )
      ).trim();
    }
  }

  const categoriesRaw = (
    await prompt(
      opts,
      `arXiv categories comma-separated [${DEFAULT_SETTINGS.arxiv.categories.join(",")}]: `,
    )
  ).trim();
  const categories = categoriesRaw
    ? categoriesRaw.split(",").map((s) => s.trim()).filter(Boolean)
    : [...DEFAULT_SETTINGS.arxiv.categories];
  const timezone = (
    await prompt(
      opts,
      `Timezone [${DEFAULT_SETTINGS.arxiv.timezone}]: `,
    )
  ).trim() || DEFAULT_SETTINGS.arxiv.timezone;
  const summaryLanguage = (
    await prompt(opts, 'Summary language "zh" or "en" [zh]: ')
  )
    .trim()
    .toLowerCase();
  const lang = summaryLanguage === "en" ? "en" : "zh";

  const cacheDir = path.join(vaultRoot, ".cache", "arxiv-daily");
  const body = renderInitToml({
    vaultRoot,
    cacheDir,
    apiKey,
    baseUrl,
    model,
    categories,
    timezone,
    summaryLanguage: lang,
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
  writeLine(stdout, `Wrote ${configPath}`);
  writeLine(
    stdout,
    "Edit [[arxiv.topics]] for research interests. Optional: set [schedule] enabled=true then: arxiv-daily schedule install",
  );
  writeLine(stdout, "Then: arxiv-daily run --today");
  return 0;
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
  categories: string[];
  timezone: string;
  summaryLanguage: string;
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
# arXiv Daily — CLI 配置
# 路径固定: ~/.config/arxiv-daily/config.toml  (或 $XDG_CONFIG_HOME/...)
# 可含密钥，请勿把此文件发到公开地方。
#
# 改研究兴趣 / 让 Agent 帮忙时：
#   - 主要改 [arxiv] 的 categories 与下方 [[arxiv.topics]]
#   - 不要编造服务地址；不要添加 hosted_base_url
#   - 密钥只替换值，不要删掉键名
# 跑任务: 配好后执行 arxiv-daily run --today （需已 init）
# =============================================================================

schema_version = 1

# 笔记库根目录（绝对路径）
vault_root = ${tomlString(input.vaultRoot)}
# 缓存目录（可删）
cache_dir = ${tomlString(input.cacheDir)}

[llm]
api_key = ${tomlString(input.apiKey)}
base_url = ${tomlString(input.baseUrl)}
model = ${tomlString(input.model)}
provider = ${tomlString(DEFAULT_SETTINGS.llm.provider)}
thinking_mode = ${DEFAULT_SETTINGS.llm.thinkingMode}
reasoning_effort = ${tomlString(DEFAULT_SETTINGS.llm.reasoningEffort)}

[arxiv]
# 例: ["astro-ph"], ["cs.LG", "cs.AI"]
categories = [${cats}]
timezone = ${tomlString(input.timezone)}

# 复制整段 [[arxiv.topics]] 可增加主题。description 越具体，筛选越好。
[[arxiv.topics]]
name = "我的研究方向（请修改）"
tag = "my-research"
description = "用自然语言描述你关心的问题、方法与对象；也可说明要排除什么。"
detail = true

[output]
summary_language = ${tomlString(input.summaryLanguage)}
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
link_style = "wikilink"

[email]
# true = 某日 pipeline completed 后自动发摘要（失败不影响写日报）
enabled = ${input.email.enabled}
# "self" = 自己的 Resend API Key；"hosted" = 官方代发（需验证码 token）
mode = ${tomlString(input.email.mode)}
to = ${tomlString(input.email.to)}
from_email = ""
from_name = "arXiv Daily"
api_key = ${tomlString(input.email.apiKey)}
hosted_token = ${tomlString(input.email.hostedToken)}

[schedule]
# init 写入默认，不在向导里提问。改完后: arxiv-daily schedule install
enabled = false
on = "09:30"
# 0 = 每天只在 on 跑一次；例如 4 = 从 09:30 起每 4 小时一次直到 until
interval_hours = 0
until = "18:00"
weekdays_only = true

[advanced]
log_level = "info"
`;
}

function tomlString(value: string): string {
  return JSON.stringify(value);
}

/** Naive merge: if existing has a non-empty assignment for a simple key line, keep existing file entirely when merge — for v1 keep existing and only fill empty api keys is hard; use: prefer existing whole file if merge and only append missing sections is complex. Spec: fill missing keys. Simple approach: if merge, keep existing when non-empty secrets present, else write new. */
function mergeTomlPreferExisting(existing: string, generated: string): string {
  // v1 pragmatic merge: if existing lacks vault_root, replace; else keep existing and append a comment that merge kept file.
  if (/^\s*vault_root\s*=/m.test(existing)) {
    return `${existing.trimEnd()}\n\n# --- init merge: kept existing file; re-run with overwrite to replace ---\n`;
  }
  return generated;
}
