import { readFile } from "node:fs/promises";
import path from "node:path";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { WorkbenchError } from "./documents";

export interface WorkbenchPreferences { sidebarWidth: number | null; sidebarCollapsed: boolean }
const filename = "workbench-ui.json";
export async function readPreferences(configPath: string): Promise<WorkbenchPreferences> {
  try { return validate(JSON.parse(await readFile(path.join(path.dirname(configPath), filename), "utf8"))); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return { sidebarWidth: null, sidebarCollapsed: false }; throw error; }
}
export async function savePreferences(configPath: string, body: Record<string, unknown>): Promise<WorkbenchPreferences> {
  const value = validate(body);
  await new NodeStorageAdapter(path.dirname(configPath)).writeTextAtomic(filename, `${JSON.stringify(value)}\n`, 0o600);
  return value;
}
function validate(body: unknown): WorkbenchPreferences {
  if (!body || typeof body !== "object") throw new WorkbenchError(400, "布局偏好无效。");
  const { sidebarWidth, sidebarCollapsed } = body as Record<string, unknown>;
  if ((sidebarWidth !== null && (typeof sidebarWidth !== "number" || !Number.isFinite(sidebarWidth) || sidebarWidth < 280 || sidebarWidth > 900)) || typeof sidebarCollapsed !== "boolean") throw new WorkbenchError(400, "布局偏好无效。");
  return { sidebarWidth: sidebarWidth as number | null, sidebarCollapsed };
}
