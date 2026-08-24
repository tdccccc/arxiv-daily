import { Notice, setIcon, type App, type TFile } from "obsidian";
import {
  modernArxivResources,
  type DashboardAction,
  type DashboardRow,
} from "@arxiv-daily/core";
import type { RunStateEntry } from "@arxiv-daily/core";
import { ObsidianResourceOpener } from "../hosts/obsidian/resource-opener";
import type ArxivDailyPlugin from "../../main";
import {
  markdownPathFromLeaf,
  normalizeVaultPath,
} from "./files";

/** Dashboard status/trash helpers (moved out of view.ts). */
export function dashboardHeaderStatusText(input: {
  isRunning: boolean;
  lastCompletedDate?: string;
}): string {
  if (input.isRunning) return "Running…";
  return `Last run: ${input.lastCompletedDate ?? "never"}`;
}

export function latestCompletedRunDate(
  runState: Record<string, RunStateEntry | undefined>,
): string | undefined {
  const completed = Object.entries(runState)
    .filter(([, entry]) => entry?.status === "completed")
    .map(([date]) => date)
    .sort();
  return completed[completed.length - 1];
}

export async function trashFileWithUserPreference(
  app: Pick<App, "fileManager" | "vault">,
  file: TFile,
): Promise<void> {
  const fileManager = app.fileManager as unknown as {
    trashFile?: (target: TFile) => Promise<void>;
  };
  if (typeof fileManager.trashFile === "function") {
    await fileManager.trashFile(file);
    return;
  }

  // Obsidian before 1.6.6 does not expose the user's trash preference.
  // Prefer system trash; Vault.trash falls back to the local .trash folder.
  const legacyTrash = Reflect.get(app.vault, "trash");
  if (typeof legacyTrash !== "function") {
    throw new Error("No compatible Obsidian trash API is available");
  }
  await Reflect.apply(legacyTrash, app.vault, [file, true]);
}

export async function openMarkdownFileOnce(
  app: {
    workspace: {
      getLeavesOfType?(type: string): unknown[];
      setActiveLeaf?(leaf: unknown, options?: { focus?: boolean }): void;
      openLinkText(path: string, sourcePath: string, newLeaf?: boolean): Promise<void>;
    };
  },
  path: string,
): Promise<void> {
  const target = normalizeVaultPath(path);
  const leaves = app.workspace.getLeavesOfType?.("markdown") ?? [];
  for (const leaf of leaves) {
    const leafPath = markdownPathFromLeaf(leaf);
    if (leafPath && normalizeVaultPath(leafPath) === target) {
      if (app.workspace.setActiveLeaf) {
        app.workspace.setActiveLeaf(leaf, { focus: true });
      } else {
        await app.workspace.openLinkText(path, "", false);
      }
      return;
    }
  }
  await app.workspace.openLinkText(path, "", false);
}

export function appendSettingsButton(
  parent: HTMLElement,
  onClick: () => void,
): HTMLButtonElement {
  const button = parent.createEl("button", {
    cls: "arxiv-daily-dashboard__settings-btn",
    attr: {
      type: "button",
      "aria-label": "Open settings",
    },
  });
  setIcon(button, "settings");
  button.createSpan({ text: "Settings" });
  button.addEventListener("click", onClick);
  return button;
}

export function applyStarButtonState(
  button: HTMLButtonElement,
  starred: boolean,
): void {
  button.classList.toggle("is-starred", starred);
  button.setAttribute("aria-pressed", String(starred));
  button.setAttribute(
    "aria-label",
    starred ? "Unstar paper" : "Star paper",
  );
  button.replaceChildren();
  setIcon(button, "star");
}

export function topicOptions(entries: DashboardRow["entry"][]): string[] {
  const topics = new Set<string>();
  for (const entry of entries) {
    for (const topic of [entry.primaryTopic, ...entry.topics]) {
      const trimmed = topic.trim();
      if (trimmed) topics.add(trimmed);
    }
  }
  return [...topics].sort((a, b) => a.localeCompare(b));
}

export async function openArxivResource(
  rawArxivId: string,
  kind: "abs" | "pdf",
  plugin: ArxivDailyPlugin,
): Promise<void> {
  const resources = modernArxivResources(rawArxivId);
  const label = kind === "pdf" ? "PDF" : "arXiv";
  if (!resources) {
    plugin.logger.warn(`dashboard: refused invalid arXiv ID for ${label}`);
    new Notice(`arXiv Daily: invalid arXiv ID; ${label} was not opened`);
    return;
  }
  const url = kind === "pdf" ? resources.pdfUrl : resources.absUrl;
  plugin.logger.info(`dashboard: opening ${label} URL ${url}`);
  await new ObsidianResourceOpener(plugin.app).openUrl(url);
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
}

export function describeDashboardAction(action: DashboardAction): string {
  const count = action.arxivIds.length;
  if (action.type === "set_priority") {
    return `set_priority:${action.priority}:${count}`;
  }
  if (action.type === "set_status") {
    return `set_status:${action.status}:${count}`;
  }
  return `${action.type}:${count}`;
}

export function isStarredEntry(entry: DashboardRow["entry"]): boolean {
  return entry.status !== "ignored" && entry.priority === "high";
}

export function deferDashboardAction(action: () => void): void {
  window.setTimeout(action, 0);
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return Boolean(
    value &&
      (typeof value === "object" || typeof value === "function") &&
      typeof (value as { then?: unknown }).then === "function",
  );
}

export async function executeObsidianCommand(
  app: unknown,
  commandId: string,
  pluginId?: string,
): Promise<boolean> {
  const commands = (app as {
    commands?: {
      executeCommandById?: (id: string) => unknown;
      commands?: Record<
        string,
        {
          callback?: () => unknown;
          checkCallback?: (checking: boolean) => unknown;
        }
      >;
    };
  })?.commands;
  if (!commands) return false;
  const ids = commandId.includes(":")
    ? [commandId]
    : uniqueCommandIds([
        pluginId ? `${pluginId}:${commandId}` : "",
        commandId,
      ]);
  const registeredIds = ids.filter((id) => commands.commands?.[id]);
  const executableIds = registeredIds.length ? registeredIds : ids;

  if (typeof commands.executeCommandById === "function") {
    for (const id of executableIds) {
      const result = commands.executeCommandById(id);
      if (isPromiseLike(result)) await result;
      if (result !== false) return true;
    }
    return false;
  }

  const id = registeredIds[0];
  const command = id ? commands.commands?.[id] : undefined;
  if (!command) return false;
  const callback = command.callback;
  if (typeof callback === "function") {
    const result = callback();
    if (isPromiseLike(result)) await result;
    return result !== false;
  }
  const checkCallback = command.checkCallback;
  if (typeof checkCallback === "function") {
    const result = checkCallback(false);
    if (isPromiseLike(result)) await result;
    return result !== false;
  }
  return false;
}

function uniqueCommandIds(ids: string[]): string[] {
  const out: string[] = [];
  for (const id of ids) {
    if (!id || out.includes(id)) continue;
    out.push(id);
  }
  return out;
}

export function recentDatesFallbackNotice(refreshedAt: number): string {
  if (!refreshedAt) {
    return "arXiv recent dates are still refreshing in the background.";
  }
  const refreshed = new Date(refreshedAt).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  return `arXiv recent dates are still refreshing in the background; using cached data from ${refreshed}.`;
}
