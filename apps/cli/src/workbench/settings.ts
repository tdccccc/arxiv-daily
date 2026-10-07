import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as os from "node:os";
import { parse, stringify } from "smol-toml";
import { DEFAULT_SETTINGS, arxivCategories, sha256Hex, type Topic } from "@arxiv-daily/core";
import { NodeFileLock, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { loadCliConfig, type CliRuntimeConfig } from "../config";
import { defaultCliVaultRoot } from "../config-path";
import { extendedSettings, type ExtendedSettingsValues } from "./settings-fields";
import { patchWorkbenchBusinessSettings } from "./settings-adapter";
import { WorkbenchError } from "./documents";

export interface WorkbenchSettingsValues extends ExtendedSettingsValues {
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
    // First run suggests the same save root as `init`, so edits can autosave from the start.
    values: { ...extendedSettings(config), vaultRoot: config?.vaultRoot ?? defaultCliVaultRoot(), baseUrl, provider: settings.llm.provider, model: settings.llm.model,
      apiKeyConfigured: Boolean(settings.llm.apiKey.trim()), categories: arxivCategories(settings.arxiv), timezone: settings.arxiv.timezone,
      summaryLanguage: settings.output.summaryLanguage ?? "zh", topics: structuredClone(settings.arxiv.topics),
      dailyDir: settings.output.dailyDir, papersDir: settings.output.papersDir },
  };
}

/** Same target and lock identity as the CLI's library connection writer. */
export async function saveWorkbenchSettings(configPath: string, body: unknown): Promise<CliRuntimeConfig> {
  if (!record(body) || !(body.revision === null || typeof body.revision === "string") || !record(body.values)) invalid("设置请求无效。");

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
    const previous = raw !== null ? await decode(configPath, raw) : null;
    const document = raw === null ? {} : parse(raw);
    const vaultRoot = resolveVaultRoot(body.values.vaultRoot ?? previous?.vaultRoot);
    const updated: Record<string, unknown> = { ...document, vault_root: vaultRoot };
    try { patchWorkbenchBusinessSettings(updated, body.values, previous); }
    catch (error) { throw new WorkbenchError(400, error instanceof Error ? error.message : "设置内容无效。"); }
    const content = stringify(updated);
    await decode(configPath, content);
    assertRevision(await readOptional(target), expected);
    await new NodeStorageAdapter(directory).writeTextAtomic(fileName, content, 0o600);
    return await loadCliConfig({ configPath });
  } finally { await lease.release(); }
}

// Equivalent to /[\u0000-\u001f]/u: any C0 control character (DEL is
// deliberately allowed, matching the original pattern). Scanning char codes
// avoids tripping no-control-regex while matching the same set of strings.
function hasC0ControlCharacter(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    if (value.charCodeAt(index) <= 0x1f) return true;
  }
  return false;
}

function resolveVaultRoot(input: unknown): string {
  if (typeof input !== "string" || !input.trim() || input.length > 20000 || hasC0ControlCharacter(input)) invalid("请填写有效的 vaultRoot。");
  let value = input.trim();
  if (value === "~") value = os.homedir();
  else if (value.startsWith("~/")) value = path.join(os.homedir(), value.slice(2));
  if (!path.isAbsolute(value)) invalid("保存目录必须使用绝对路径。");
  return value;
}
function record(value: unknown): value is Record<string, unknown> { return Boolean(value) && typeof value === "object" && !Array.isArray(value); }
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
