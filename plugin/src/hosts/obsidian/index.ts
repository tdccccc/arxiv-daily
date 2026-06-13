import type { App } from "obsidian";
import type { HostAdapters, ProgressReporter } from "../../core/adapters";
import { NoopProgressReporter } from "../../services/progress";
import type { PluginSettings } from "../../settings/types";
import { ObsidianHttpClient } from "./http-client";
import { ObsidianResourceOpener } from "./resource-opener";
import { ObsidianSettingsSecretProvider } from "./secrets";
import { ObsidianStorageAdapter } from "./storage-adapter";

export interface ObsidianHostAdapterOptions {
  app: App;
  getSettings: () => PluginSettings;
  persistSettings?: () => Promise<void> | void;
  progress?: ProgressReporter;
}

export function buildObsidianHostAdapters(
  opts: ObsidianHostAdapterOptions,
): HostAdapters {
  return {
    http: new ObsidianHttpClient(),
    storage: new ObsidianStorageAdapter(opts.app.vault),
    secrets: new ObsidianSettingsSecretProvider(
      opts.getSettings,
      opts.persistSettings,
    ),
    progress: opts.progress ?? new NoopProgressReporter(),
    opener: new ObsidianResourceOpener(opts.app),
  };
}

export { ObsidianHttpClient } from "./http-client";
export { ObsidianResourceOpener } from "./resource-opener";
export { ObsidianSettingsSecretProvider } from "./secrets";
export { ObsidianStorageAdapter } from "./storage-adapter";
