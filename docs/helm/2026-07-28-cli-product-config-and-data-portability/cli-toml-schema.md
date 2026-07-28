# CLI TOML schema draft

status: draft (docs only; not implemented)  
decision_ref: `docs/adr/0003-two-products-cli-config-and-data-portability.md`  
goal_ref: `./goal.md`  
phase_ref: `./phases/01-cli-toml-init.md`

Hand-edited config at `$XDG_CONFIG_HOME/arxiv-daily/config.toml`  
(default `~/.config/arxiv-daily/config.toml`; Windows `%APPDATA%\arxiv-daily\config.toml`).

**UX goals for this file**

1. A person can fill it by reading **inline comments**.
2. An **agent** can be given this file + a short research description and safely edit topics/categories.
3. Only fields a normal user (or agent) should touch appear. Internal/service knobs stay out.

TOML keys are **snake_case**. Loader maps into core camelCase `PluginSettings` + CLI deployment fields.

---

## 1. File identity

| Item | Rule |
|---|---|
| Path | Fixed XDG/APPDATA only (ADR 0003) |
| Encoding | UTF-8 |
| Format | TOML 1.0 |
| Comments | **Required in files written by `init`** — full annotated sample below |
| Secrets | Inline (`llm.api_key`, `email.api_key`, `email.hosted_token`) |
| Not read | cwd JSON, any `ARXIV_DAILY_*` env |

```toml
schema_version = 1
```

Missing `schema_version` → treat as `1`.

---

## 2. What belongs in the user file (and what does not)

### Include (user / agent)

| Area | Keys |
|---|---|
| Paths | `vault_root`, `cache_dir` |
| LLM | `api_key`, `base_url`, `model` (+ optional provider / thinking) |
| arXiv | `categories`, `timezone`, `[[arxiv.topics]]` |
| Output | `summary_language`, optional dirs / `link_style` |
| Email | `enabled`, `mode`, `to`, `from_email`, `from_name`, `api_key` *or* `hosted_token` |
| Schedule | `enabled`, `on`, `interval_hours`, `until`, `weekdays_only` (defaults from `init`; **not** asked in wizard) |
| Advanced (optional) | `log_level`, delays / char limits if ever needed |

### Exclude from user-facing TOML (v1)

| Omitted | Why |
|---|---|
| `email.hosted_base_url` | Service URL is a product default in core, not user config. |
| `detail_selection.profile` / whole profile UX | No “profile” product surface on CLI. Engine uses core default (`balanced`) unless we later add a single plain knob. |
| `detail_selection.*` thresholds | Power-user only; omit from `init` output. Default deep-dive policy = core balanced. |
| Plugin-style `tick_interval_min` / in-process tick | CLI does **not** sleep inside the process; interval only shapes **generated cron** lines. |
| `arxiv.category` singular | Derived as `categories[0]` on load. |
| Plugin-only UI state | Not applicable. |

If a key is omitted in the file, load **core defaults** (`DEFAULT_SETTINGS` + deployment defaults).

---

## 3. Comments policy (`init` must write them)

`init` does **not** write a bare key dump. It writes a **commented template**:

1. **File header** — what this file is, that it may contain secrets, that CLI reads only this path.
2. **Agent blurb** — short block: what to edit for research interests (`categories`, `[[arxiv.topics]]`), what not to invent (`hosted_base_url`, random URLs).
3. **Per-section comments** — one or two lines above each table: purpose + allowed values.
4. **Per-field comments** — on non-obvious keys (email modes, empty `from_email`, topic `description`).

Comments are the primary help; we do not require a separate schema website for v1.

Suggested languages: **中文为主** in comments (user audience), keys remain English snake_case (stable for agents/tools).

---

## 4. Top-level deployment (CLI-only)

| TOML key | Type | Required | Default (`init`) | Maps to |
|---|---|---|---|---|
| `vault_root` | string | **yes** | user answer (prefer absolute) | `vaultRoot` |
| `cache_dir` | string | no | `{vault_root}/.cache/arxiv-daily` | `cacheDir` |

```toml
# 笔记库根目录（Obsidian vault 或任意输出根）。请用绝对路径。
vault_root = "/home/alice/Notes"

# HTML/全文缓存（可删，会重建）。默认在 vault 内、日报目录外。
cache_dir = "/home/alice/Notes/.cache/arxiv-daily"
```

---

## 5. `[llm]`

| TOML key | Type | Need for run | Default | Core |
|---|---|---|---|---|
| `api_key` | string | **yes** | `""` | `llm.apiKey` |
| `base_url` | string | **yes** | deepseek default | `llm.baseUrl` |
| `model` | string | **yes** | deepseek default | `llm.model` |
| `provider` | string | no | `"deepseek"` | `llm.provider` |
| `thinking_mode` | bool | no | `true` | `llm.thinkingMode` |
| `reasoning_effort` | string | no | `"high"` | `llm.reasoningEffort` |

`init` asks for key (and optionally model); writes the rest with comments.

```toml
[llm]
# 大模型 API Key（密钥，勿分享此文件）
api_key = "sk-..."
# OpenAI 兼容接口的 Base URL
base_url = "https://api.deepseek.com/v1"
model = "deepseek-v4-pro"
# provider 仅作标记/预设名，真正请求看 base_url + model
provider = "deepseek"
thinking_mode = true
reasoning_effort = "high"
```

---

## 6. `[arxiv]` and topics

| TOML key | Type | Required | Default | Core |
|---|---|---|---|---|
| `categories` | string[] | **yes** | e.g. `["astro-ph"]` | `arxiv.categories` |
| `timezone` | string | no | `"Asia/Shanghai"` | `arxiv.timezone` |

### `[[arxiv.topics]]` — main hand-edit / agent surface

| TOML key | Type | Required | Notes | Core |
|---|---|---|---|---|
| `name` | string | **yes** | section title in daily report | `name` |
| `tag` | string | **yes** | short slug, unique | `tag` |
| `description` | string | **yes** for good runs | natural-language inclusion rule; **best place for agent help** | `description` |
| `detail` | bool | no | default `true` — eligible for auto deep-dive notes | `detail` |
| `id` | string | no | auto UUID on load if missing | `id` |

**No topic templates forced in the file.** `init` writes **one placeholder** topic with comments telling the user (or agent) how to replace it.

```toml
[arxiv]
# arXiv 分类，可多个。例: "astro-ph", "cs.LG", "hep-th"
categories = ["astro-ph"]
# 用于“今天”与调度日期的时区（IANA）
timezone = "Asia/Shanghai"

# ---------------------------------------------------------------------------
# 研究主题：日报按 topic 分节。可复制整块 [[arxiv.topics]] 增加多条。
# 把 description 写清楚（或交给 Agent：粘贴本文件 + 用自然语言描述兴趣）。
# ---------------------------------------------------------------------------
[[arxiv.topics]]
name = "我的研究方向（请修改）"
tag = "my-research"
description = "用自然语言写：你希望哪些论文进入本主题（问题、方法、对象、不要什么）。"
detail = true
```

**Agent-oriented comment (in file header or above topics):**

```text
若使用 AI 助手修改配置：请主要改 arxiv.categories 与 [[arxiv.topics]] 的
name / tag / description；不要编造 email.hosted_base_url 或未知服务地址；
不要删除 llm.api_key / email 密钥字段名，只需替换值。
```

---

## 7. Automatic deep dives (no profile in TOML)

CLI **does not** expose `detail_selection.profile` (no Fewer/Recommended/More / conservative/balanced/broad in the file).

| Behavior | Rule |
|---|---|
| Default | Load core **balanced** detail-selection preset always |
| User control in v1 | Per-topic `detail = true/false` (whether that topic is eligible) |
| Later (optional, not in init) | If needed, a single plain key could be added after product naming—not “profile” |

Do **not** write a `[detail_selection]` table in `init` output.

---

## 8. `[output]`

| TOML key | Type | Default | Core |
|---|---|---|---|
| `summary_language` | `"zh"` \| `"en"` | `"zh"` | `summaryLanguage` |
| `daily_dir` | vault-relative | `"arxiv-daily/daily"` | `dailyDir` |
| `papers_dir` | vault-relative | `"arxiv-daily/papers"` | `papersDir` |
| `link_style` | `"wikilink"` \| `"relative"` | `"wikilink"` | `linkStyle` |

```toml
[output]
# 摘要语言: "zh" 或 "en"
summary_language = "zh"
# 相对 vault_root 的目录（一般保持默认）
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
link_style = "wikilink"
```

---

## 9. `[email]` — user fields only

| TOML key | Type | Default | Core |
|---|---|---|---|
| `enabled` | bool | `false` | `enabled` |
| `mode` | `"self"` \| `"hosted"` | `"self"` | `mode` |
| `to` | string | `""` | `to` |
| `from_email` | string | `""` | `fromEmail` |
| `from_name` | string | `"arXiv Daily"` | `fromName` |
| `api_key` | string | `""` | `apiKey` (mode `self`) |
| `hosted_token` | string | `""` | `hostedToken` (mode `hosted`) |

**Not in file:** `hosted_base_url` — core uses the built-in Official delivery base URL.

```toml
[email]
# true = 某日 pipeline completed 后自动发摘要（失败不影响写日报）
enabled = false
# "self" = 自己的 Resend API Key；"hosted" = 官方代发（需验证码 token）
mode = "self"
to = "you@example.com"
# self 且 from_email 留空时，使用 Resend 测试发件地址（通常只能发给 Resend 账号邮箱）
from_email = ""
from_name = "arXiv Daily"
# mode = "self" 时填写；mode = "hosted" 时可留空
api_key = "re_..."
# mode = "hosted" 时填写验证后的长 token；self 时留空
hosted_token = ""
```

`init`: ask configure email? → no defaults / yes → mode + to + secret field for that mode; leave `enabled = false` and comment to run `email-test` then set `enabled = true`.

---

## 10. `[schedule]` — system timer intent (not an in-process daemon)

CLI remains **one-shot** (ADR 0001). `[schedule]` does **not** make `arxiv-daily` sleep or tick by itself.

It describes when the **OS** should invoke `run --today`, applied by:

```text
arxiv-daily schedule show       # print managed crontab lines (no write)
arxiv-daily schedule install    # install/replace managed lines in user crontab
arxiv-daily schedule uninstall  # remove managed lines only
```

| TOML key | Type | Default (`init` writes these; **wizard does not ask**) | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | If `true`, `schedule install` installs jobs; if `false`, install no-ops or refuses with a message to set `enabled = true` |
| `on` | string `HH:MM` | `"09:30"` | First daily fire (machine local time used by cron) |
| `interval_hours` | number | `0` | **`0` = once per day at `on`**. **`> 0` = also repeat every N hours** after `on` while still before `until` |
| `until` | string `HH:MM` | `"18:00"` | Last time a repeated fire may be scheduled (only used when `interval_hours > 0`) |
| `weekdays_only` | bool | `true` | Cron dow `1-5` vs `*` |

**Not in CLI TOML:** plugin `runAtLocal`/`runUntilLocal`/`tickIntervalMin` names. Map only if we ever import plugin dumps; do not document them as CLI keys.

### Interval semantics

| `interval_hours` | Generated fires (example `on=09:30`, `until=18:00`) |
|---|---|
| `0` (default) | Only **09:30** once per selected day |
| `4` | **09:30, 13:30, 17:30** (step 4h from `on`, while `time <= until`) |
| `1` | hourly from 09:30 through last slot ≤ 18:00 |

Rules:

- Times are **wall clock on the machine** that runs cron (document skew vs `arxiv.timezone`).
- Each fire runs: `arxiv-daily run --today` (full path to the installed binary when known).
- Only **today** (in `arxiv.timezone`) is attempted per fire—**no** multi-day lookback batch on CLI. Missed days need `run --date` by hand. Same-day re-entry stays idempotent if already completed.
- Changing TOML does **not** update crontab until the user runs `schedule install` again.
- Managed lines are marked with a stable comment marker, e.g. `# arxiv-daily-managed`, so uninstall is safe.

### `init` behavior

- **Do not ask** any schedule questions.
- **Always write** a commented `[schedule]` block with the defaults above (`enabled = false`).
- End of `init` may print one line: “定时：编辑 `[schedule]` 后执行 `arxiv-daily schedule install`”.

### Cron generation sketch

```text
if !enabled → install exits 0 with “schedule.enabled is false”
slots = [on] if interval_hours == 0
        else times from on stepping interval_hours while <= until
dow = weekdays_only ? "1-5" : "*"
for each slot HH:MM → "MM HH * * {dow}  <bin> run --today"
```

No crontab on the system → print the lines and tell the user to paste (containers).

```toml
[schedule]
# false = 不安装系统定时；改 true 后执行: arxiv-daily schedule install
enabled = false
# 每天第一次触发（机器本地时间 HH:MM）
on = "09:30"
# 0 = 每天只跑 on 一次；>0 = 从 on 起每隔 N 小时再跑，直到 until
interval_hours = 0
# 仅 interval_hours > 0 时有意义
until = "18:00"
# true = 仅周一到周五
weekdays_only = true
```

**Windows:** native `schedule install` / `uninstall` are **not** supported (no Task Scheduler integration). Prefer **WSL** for CLI + cron, or use the **Obsidian plugin** on the desktop. Config path on Windows remains `%APPDATA%\arxiv-daily\config.toml` for `init` / `run`.

---

## 11. `[advanced]` (optional, lightly commented)

Only what operators sometimes change:

| TOML key | Default | Core |
|---|---|---|
| `log_level` | `"info"` | `logLevel` |
| `request_delay_ms` | `3000` | `requestDelayMs` |
| `cache_expiry_days` | `7` | `cacheExpiryDays` |

Char limits may exist in core defaults without being written by `init`. If present in file, map them; if absent, defaults.

```toml
[advanced]
# debug | info | warn | error
log_level = "info"
```

---

## 12. Load algorithm

```text
1. Path = XDG/APPDATA …/arxiv-daily/config.toml
2. Missing → error: run init
3. Parse TOML (ignore comments)
4. DEFAULT_SETTINGS + default deployment
5. Apply known keys only; unknown keys → warn (recommended)
6. Never read hosted_base_url from file (ignore if somehow present, or warn unused)
7. detailSelection ← always sanitize to balanced preset (v1)
8. category ← categories[0]
9. Parse [schedule] into CLI schedule intent (defaults if table missing)
10. Expand paths; validate output dirs
```

`run` **does not** consult `[schedule]` to sleep. Only `schedule install|show|uninstall` use it.

---

## 13. Full sample written by `init` (comments included)

```toml
# =============================================================================
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
vault_root = "/home/alice/Notes"
# 缓存目录（可删）
cache_dir = "/home/alice/Notes/.cache/arxiv-daily"

[llm]
api_key = ""
base_url = "https://api.deepseek.com/v1"
model = "deepseek-v4-pro"
provider = "deepseek"
thinking_mode = true
reasoning_effort = "high"

[arxiv]
# 例: ["astro-ph"], ["cs.LG", "cs.AI"]
categories = ["astro-ph"]
timezone = "Asia/Shanghai"

# 复制整段 [[arxiv.topics]] 可增加主题。description 越具体，筛选越好。
[[arxiv.topics]]
name = "我的研究方向（请修改）"
tag = "my-research"
description = "用自然语言描述你关心的问题、方法与对象；也可说明要排除什么。"
detail = true

[output]
summary_language = "zh"
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
link_style = "wikilink"

[email]
enabled = false
mode = "self"
to = ""
from_email = ""
from_name = "arXiv Daily"
api_key = ""
hosted_token = ""

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
```

---

## 14. Day-to-day edits

| Goal | Edit |
|---|---|
| 研究兴趣 | `[[arxiv.topics]]`（可交给 Agent） |
| 看哪些板 | `arxiv.categories` |
| 摘要中/英 | `output.summary_language` |
| 某主题是否自动深潜 | 该 topic 的 `detail` |
| 开邮件 | `email.*` 后 `enabled = true` |
| 系统定时 | `[schedule]` → `enabled = true` → `schedule install` |
| 跑得更勤 | `interval_hours`（如 `2` 或 `4`）+ 再 `install` |
| 换模型 | `llm.model` / `base_url` / `api_key` |
| 换库 | `vault_root`（及按需 `cache_dir`） |

---

## 15. JSON → TOML hand-migration (short)

| Old | New |
|---|---|
| `vaultRoot` | `vault_root` |
| `settings.llm.apiKey` | `llm.api_key` |
| `settings.arxiv.categories` / `topics` | `arxiv.categories` / `[[arxiv.topics]]` |
| `settings.output.*` | `output.*` snake_case |
| `settings.email.*` | `email.*` snake_case；**drop** `hostedBaseUrl` |
| `settings.detailSelection` | **omit** (CLI uses default balanced) |
| `settings.schedule.*` (plugin tick window) | **do not copy 1:1**; use CLI `[schedule]` (`on` / `interval_hours` / `until`) |
| `ARXIV_DAILY_*` env | **delete**; put values in this file |

---

## 16. Implementer notes

1. `init` output **must** match the commented sample style (not uncommented minimal dump).
2. Strip or ignore `hosted_base_url` if old drafts appear.
3. Do not serialize `detail_selection` in `init`. **Do** write default `[schedule]` (`enabled = false`); **do not** ask schedule questions in the wizard.
4. `schedule install` is the only writer to the user crontab; use a managed marker comment; never delete unrelated cron lines.
5. Unknown keys: prefer warn, don’t fail the whole file for a typo comment-adjacent key—optional strict mode later.
6. P2 email verify only fills `hosted_token` (and docs), never teaches `hosted_base_url`.
7. Phase placement: config load of `[schedule]` can ship with P1; `schedule install|show|uninstall` may be P1.5 or early P2—must exist before docs claim “auto cron”.

---

## 17. Out of scope

- Plugin `data.json`
- Data export zip (P3)
- Changing core Official delivery default URL (code constant, not TOML)
- In-process CLI daemon / sleeping tick loop
