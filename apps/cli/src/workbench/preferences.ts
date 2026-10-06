import { mkdir, readFile, realpath } from "node:fs/promises";
import path from "node:path";
import { DEFAULT_UI_APPEARANCE, normalizeUiAppearancePreferences, validateUiAppearancePreferences, type UiAppearancePreferences } from "@arxiv-daily/core";
import { NodeFileLock, NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { WorkbenchError } from "./documents";

export interface WorkbenchPreferences { sidebarWidth: number | null; sidebarCollapsed: boolean; appearance: UiAppearancePreferences }
const filename = "workbench-ui.json";
const defaults = (): WorkbenchPreferences => ({ sidebarWidth: null, sidebarCollapsed: false, appearance: { ...DEFAULT_UI_APPEARANCE } });
export async function readPreferences(configPath: string): Promise<WorkbenchPreferences> {
  try {
    const raw: unknown = JSON.parse(await readFile(path.join(path.dirname(configPath), filename), "utf8"));
    return { ...defaults(), ...validatePatch(raw) };
  } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return defaults(); throw error; }
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
    patch.sidebarWidth = width as number | null;
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
