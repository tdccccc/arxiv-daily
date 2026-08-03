import type { App } from "obsidian";
import type { HostAdapters, ProgressReporter } from "@arxiv-daily/core";
import { NoopProgressReporter } from "@arxiv-daily/core";
import type { PluginSettings } from "@arxiv-daily/core";
import { ObsidianHttpClient, type ObsidianRequestImpl } from "./http-client";
import { ObsidianResourceOpener } from "./resource-opener";
import { ObsidianSettingsSecretProvider } from "./secrets";
import { ObsidianStorageAdapter } from "./storage-adapter";
import { ObsidianMarkupParser } from "./markup-parser";

export interface ObsidianHostAdapterOptions {
  app: App;
  getSettings: () => PluginSettings;
  persistSettings?: () => Promise<void> | void;
  progress?: ProgressReporter;
  request?: ObsidianRequestImpl;
}

export function buildObsidianHostAdapters(
  opts: ObsidianHostAdapterOptions,
): HostAdapters {
  return {
    http: new ObsidianHttpClient(opts.request),
    storage: new ObsidianStorageAdapter(opts.app.vault),
    secrets: new ObsidianSettingsSecretProvider(
      opts.getSettings,
      opts.persistSettings,
    ),
    progress: opts.progress ?? new NoopProgressReporter(),
    opener: new ObsidianResourceOpener(opts.app),
    markupParser: new ObsidianMarkupParser(),
  };
}

export {
  ObsidianHttpClient,
  type ObsidianRequestImpl,
  type ObsidianRequestResponse,
} from "./http-client";
export { ObsidianResourceOpener } from "./resource-opener";
export { ObsidianSettingsSecretProvider } from "./secrets";
export { ObsidianStorageAdapter } from "./storage-adapter";
export { ObsidianMarkupParser } from "./markup-parser";
export { openObsidianLibrarySource } from "./library-source";
export {
  ObsidianLibraryDirectoryPicker,
  type DirectoryDialog,
  type DirectoryDialogResult,
  type LibraryDirectorySelection,
} from "./library-directory-picker";
