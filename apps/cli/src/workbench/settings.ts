import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as os from "node:os";
import { parse, stringify } from "smol-toml";
import { DEFAULT_SETTINGS, arxivCategories, sha256Hex, validateFilterConfig, type Topic } from "@arxiv-daily/core";
import { NodeFileLock, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { loadCliConfig, type CliRuntimeConfig } from "../config";
import { WorkbenchError } from "./documents";

export interface WorkbenchSettingsValues {
  vaultRoot: string; baseUrl: string; provider: string; model: string; apiKeyConfigured: boolean;
  categories: string[]; timezone: string; summaryLanguage: "zh" | "en"; topics: Topic[]; dailyDir: string; papersDir: string;
}
export interface WorkbenchSettings { setupRequired: boolean; revision: string | null; configPath: string; values: WorkbenchSettingsValues }

/** Safe editable projection. Secret fields are write-only. */
export async function readWorkbenchSettings(configPath: string): Promise<WorkbenchSettings> {
  const raw = await readOptional(configPath);
  const config = raw === null ? null : await decode(configPath, raw);
  const settings = config?.settings ?? DEFAULT_SETTINGS;
  let baseUrl = settings.llm.baseUrl;
  // Existing hand-written URLs may contain credentials; never reveal them in a form.
  try { const url = new URL(baseUrl); url.username = ""; url.password = ""; url.search = ""; url.hash = ""; baseUrl = url.toString(); } catch { baseUrl = ""; }
  return {
    setupRequired: config === null, revision: config?.configRevision ?? null, configPath,
    values: { vaultRoot: config?.vaultRoot ?? "", baseUrl, provider: settings.llm.provider, model: settings.llm.model,
      apiKeyConfigured: Boolean(settings.llm.apiKey.trim()), categories: arxivCategories(settings.arxiv), timezone: settings.arxiv.timezone,
      summaryLanguage: settings.output.summaryLanguage ?? "zh", topics: structuredClone(settings.arxiv.topics),
      dailyDir: settings.output.dailyDir, papersDir: settings.output.papersDir },
  };
}

/** Same target and lock identity as the CLI's library connection writer. */
export async function saveWorkbenchSettings(configPath: string, body: unknown): Promise<CliRuntimeConfig> {
  if (!record(body) || !(body.revision === null || typeof body.revision === "string") || !record(body.values)) invalid("设置请求无效。");
  const values = validateValues(body.values);
  const expected = body.revision;
  await fs.mkdir(path.dirname(configPath), { recursive: true, mode: 0o700 });
  let target: string;
  try { target = await fs.realpath(configPath); } catch (error) {
    if (!missing(error)) throw error;
    // A dangling symlink must not be treated as a missing configuration file.
    try { await fs.lstat(configPath); throw new WorkbenchError(409, "配置路径已改变，请检查后重试。"); } catch (statError) { if (!missing(statError)) throw statError; }
    target = path.join(await fs.realpath(path.dirname(configPath)), path.basename(configPath));
  }
  const directory = path.dirname(target), fileName = path.basename(target);
  const locks = new NodeFileLock(directory, { lockRoot: path.join(directory, ".arxiv-daily-config-locks") });
  const lease = await locks.acquire(`cli-config:${fileName}`, { wait: true });
  if (!lease) throw new WorkbenchError(409, "配置正在修改，请稍后重试。");
  try {
    const raw = await readOptional(target);
    assertRevision(raw, expected);
    if (raw !== null) await decode(configPath, raw);
    const document = raw === null ? {} : parse(raw);
    const llm = table(document.llm), arxiv = table(document.arxiv), output = table(document.output);
    const content = stringify({ ...document, vault_root: values.vaultRoot,
      llm: { ...llm, base_url: values.baseUrl, provider: values.provider, model: values.model,
        api_key: values.apiKey || llm.api_key || "" },
      arxiv: { ...arxiv, categories: values.categories, timezone: values.timezone, topics: values.topics },
      output: { ...output, summary_language: values.summaryLanguage, daily_dir: values.dailyDir, papers_dir: values.papersDir },
    });
    const next = await decode(configPath, content);
    const validation = validateFilterConfig(next.settings);
    if (!validation.ok) invalid("请填写模型密钥、有效主题和输出目录。");
    assertRevision(await readOptional(target), expected);
    await new NodeStorageAdapter(directory).writeTextAtomic(fileName, content, 0o600);
    return await loadCliConfig({ configPath });
  } finally { await lease.release(); }
}

function validateValues(value: Record<string, unknown>) {
  const text = (key: string) => {
    const input = value[key];
    if (typeof input !== "string" || !input.trim() || input.length > 20000 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/u.test(input)) invalid(`请填写有效的 ${key}。`);
    return input.trim();
  };
  let vaultRoot = text("vaultRoot");
  if (vaultRoot === "~") vaultRoot = os.homedir();
  else if (vaultRoot.startsWith("~/")) vaultRoot = path.join(os.homedir(), vaultRoot.slice(2));
  if (!path.isAbsolute(vaultRoot)) invalid("保存目录必须使用绝对路径。");
  const baseUrl = text("baseUrl");
  try {
    const url = new URL(baseUrl);
    if (!["http:", "https:"].includes(url.protocol) || !url.hostname || url.username || url.password || url.search || url.hash) invalid("模型地址须为不含凭证、查询参数的 HTTP(S) 地址。");
  } catch { invalid("模型地址须为不含凭证、查询参数的 HTTP(S) 地址。"); }
  const timezone = text("timezone");
  try { new Intl.DateTimeFormat("en", { timeZone: timezone }).format(); } catch { invalid("请选择有效时区。"); }
  if (value.summaryLanguage !== "zh" && value.summaryLanguage !== "en") invalid("请选择中文或英文。");
  if (!Array.isArray(value.categories) || !value.categories.length || value.categories.length > 200 || value.categories.some(c => typeof c !== "string" || !/^[a-zA-Z][a-zA-Z0-9.-]*$/.test(c))) invalid("请填写有效 arXiv 分类。");
  if (new Set(value.categories).size !== value.categories.length) invalid("arXiv 分类不能重复。");
  if (!Array.isArray(value.topics) || !value.topics.length || value.topics.length > 100) invalid("请至少填写一个研究主题。");
  const ids = new Set<string>();
  const topics: Topic[] = value.topics.map(topic => {
    if (!record(topic) || ["id", "name", "tag", "description"].some(key => typeof topic[key] !== "string" || !topic[key].trim() || topic[key].length > 20000) || typeof topic.detail !== "boolean") invalid("研究主题格式无效。");
    const id = (topic.id as string).trim();
    if (ids.has(id)) invalid("研究主题标识重复。");
    ids.add(id);
    return { id, name: (topic.name as string).trim(), tag: (topic.tag as string).trim(), description: (topic.description as string).trim(), detail: topic.detail as boolean };
  });
  if (value.apiKey !== undefined && (typeof value.apiKey !== "string" || value.apiKey.length > 20000)) invalid("模型密钥格式无效。");
  return { vaultRoot, baseUrl, provider: text("provider"), model: text("model"), categories: value.categories as string[], timezone,
    summaryLanguage: value.summaryLanguage, topics, dailyDir: text("dailyDir"), papersDir: text("papersDir"), apiKey: typeof value.apiKey === "string" ? value.apiKey.trim() : "" };
}
function record(value: unknown): value is Record<string, unknown> { return Boolean(value) && typeof value === "object" && !Array.isArray(value); }
function table(value: unknown): Record<string, unknown> { return record(value) ? value : {}; }
function invalid(message: string): never { throw new WorkbenchError(400, message); }
function missing(error: unknown): boolean { return (error as NodeJS.ErrnoException)?.code === "ENOENT"; }
async function readOptional(file: string): Promise<string | null> { try { return await fs.readFile(file, "utf8"); } catch (error) { if (missing(error)) return null; throw error; } }
async function decode(configPath: string, raw: string): Promise<CliRuntimeConfig> {
  try { return await loadCliConfig({ configPath, readText: async () => raw }); } catch { return invalid("配置格式无效，请检查配置文件或设置内容。"); }
}
function assertRevision(raw: string | null, expected: unknown) {
  const actual = raw === null ? null : `sha256:${sha256Hex(raw)}`;
  if (actual !== expected) throw new WorkbenchError(409, "配置已改变，请重新打开设置后再保存。");
}
