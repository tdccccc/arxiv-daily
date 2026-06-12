import type { HostAdapters } from "../../core/adapters";
import { NodeHttpClient, type FetchLike } from "./http-client";
import { StreamProgressReporter, type WritableTextStream } from "./progress";
import { StreamResourceOpener } from "./resource-opener";
import { EnvSecretProvider } from "./secrets";
import { NodeStorageAdapter } from "./storage-adapter";

export interface NodeHostAdapterOptions {
  rootDir?: string;
  env?: Record<string, string | undefined>;
  fetch?: FetchLike;
  progressStream?: WritableTextStream;
  openerStream?: WritableTextStream;
}

export function buildNodeHostAdapters(
  opts: NodeHostAdapterOptions = {},
): HostAdapters {
  return {
    http: new NodeHttpClient(opts.fetch),
    storage: new NodeStorageAdapter(opts.rootDir),
    secrets: new EnvSecretProvider(opts.env),
    progress: new StreamProgressReporter(opts.progressStream),
    opener: new StreamResourceOpener(opts.openerStream),
  };
}

export { NodeHttpClient, type FetchLike } from "./http-client";
export { StreamProgressReporter } from "./progress";
export { StreamResourceOpener } from "./resource-opener";
export { EnvSecretProvider } from "./secrets";
export { NodeStorageAdapter } from "./storage-adapter";
