import type { App, TFile } from "obsidian";
import type { RunStateEntry } from "@arxiv-daily/core";

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
