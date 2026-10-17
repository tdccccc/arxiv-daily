import { mkdir, readFile, realpath } from "node:fs/promises";
import path from "node:path";
import { DEFAULT_UI_APPEARANCE, normalizeUiAppearancePreferences, validateUiAppearancePreferences, type UiAppearancePreferences } from "@arxiv-daily/core";
import { NodeFileLock, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { WorkbenchError } from "./documents";

export interface WorkbenchPreferences { sidebarWidth: number | null; sidebarCollapsed: boolean; appearance: UiAppearancePreferences }
const filename = "workbench-ui.json";
const defaults = (): WorkbenchPreferences => ({ sidebarWidth: null, sidebarCollapsed: false, appearance: { ...DEFAULT_UI_APPEARANCE } });
export async function readPreferences(configPath: string, warn: (message: string) => void = defaultWarn): Promise<WorkbenchPreferences> {
  const file = path.join(path.dirname(configPath), filename);
  let text: string;
  try { text = await readFile(file, "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return defaults(); throw error; }
  return { ...defaults(), ...salvage(text, file, warn) };
}

function defaultWarn(message: string): void { process.stderr.write(`${message}\n`); }

/**
 * These are cosmetic settings, so a damaged file (hand edit, sync conflict, empty file) must never
 * take the workbench down or block saving: keep every field that is still valid and warn about the rest.
 * The next save rewrites the file in full.
 */
function salvage(text: string, file: string, warn: (message: string) => void): Partial<WorkbenchPreferences> {
  let raw: unknown;
  try { raw = JSON.parse(text); }
  catch { warn(`Ignoring unreadable workbench preferences ${file}; using defaults until they are saved again.`); return {}; }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    warn(`Ignoring workbench preferences ${file}: expected a JSON object; using defaults until they are saved again.`);
    return {};
  }
  const kept: Partial<WorkbenchPreferences> = {};
  for (const [key, value] of Object.entries(raw)) {
    try { Object.assign(kept, validatePatch({ [key]: value })); }
    catch (error) {
      if (!(error instanceof WorkbenchError)) throw error;
      warn(`Ignoring invalid "${key}" in workbench preferences ${file}.`);
    }
  }
  return kept;
}

/** Merge under a shared per-file lock so layout and appearance saves cannot erase each other. */
export async function savePreferences(configPath: string, body: Record<string, unknown>): Promise<WorkbenchPreferences> {
  const patch = validatePatch(body);
  await mkdir(path.dirname(configPath), { recursive: true, mode: 0o700 });
  const directory = await realpath(path.dirname(configPath));
  const locks = new NodeFileLock(directory, { lockRoot: path.join(directory, ".arxiv-daily-config-locks") });
  const lease = await locks.acquire(`workbench-preferences:${filename}`, { wait: true });
  if (!lease) throw new WorkbenchError(409, "界面偏好正在修改，请稍后重试。");
  try {
    const value = { ...await readPreferences(path.join(directory, path.basename(configPath))), ...patch };
    await new NodeStorageAdapter(directory).writeTextAtomic(filename, `${JSON.stringify(value)}\n`, 0o600);
    return value;
  } finally { await lease.release(); }
}
function validatePatch(body: unknown): Partial<WorkbenchPreferences> {
  if (!body || typeof body !== "object" || Array.isArray(body)) throw new WorkbenchError(400, "界面偏好无效。");
  const fields = body as Record<string, unknown>;
  if (Object.keys(fields).some(key => !["sidebarWidth", "sidebarCollapsed", "appearance"].includes(key))) throw new WorkbenchError(400, "界面偏好包含未知字段。");
  const patch: Partial<WorkbenchPreferences> = {};
  if (Object.hasOwn(fields, "sidebarWidth")) {
    const width = fields.sidebarWidth;
    if (width !== null && (typeof width !== "number" || !Number.isFinite(width) || width < 280 || width > 900)) throw new WorkbenchError(400, "侧栏宽度无效。");
    patch.sidebarWidth = width;
  }
  if (Object.hasOwn(fields, "sidebarCollapsed")) {
    if (typeof fields.sidebarCollapsed !== "boolean") throw new WorkbenchError(400, "侧栏折叠偏好无效。");
    patch.sidebarCollapsed = fields.sidebarCollapsed;
  }
  if (Object.hasOwn(fields, "appearance")) {
    if (!validateUiAppearancePreferences(fields.appearance)) throw new WorkbenchError(400, "请选择有效的主题和界面语言。");
    patch.appearance = normalizeUiAppearancePreferences(fields.appearance);
  }
  return patch;
}
