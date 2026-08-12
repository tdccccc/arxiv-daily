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
  changeSettingValue?: (key: string, value: unknown) => Promise<void> | void;
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
      opts.changeSettingValue,
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
  ObsidianPdfTextExtractor,
  type PdfJsLib,
  type PdfJsLoadingTask,
  type PdfJsDocument,
  type PdfJsPage,
  type PdfJsTextContent,
  type PdfJsTextItem,
} from "./pdf-text-extractor";
export {
  ObsidianLibraryDirectoryPicker,
  type DirectoryDialog,
  type DirectoryDialogResult,
  type LibraryDirectorySelection,
} from "./library-directory-picker";
export {
  createTransformersEmbeddingModel,
  EMBEDDING_MODEL_ID,
  inspectTransformersEnv,
  alignElectronReleaseProbe,
  describeRuntimeProbe,
  type TransformersEmbeddingModelOptions,
  type TransformersEnvFacts,
} from "./embedding-model";
