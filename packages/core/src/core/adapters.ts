export type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

export interface HttpRequest {
  url: string;
  method?: HttpMethod;
  headers?: Record<string, string>;
  body?: string | ArrayBuffer;
  responseType?: "text" | "arrayBuffer";
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface HttpResponse {
  status: number;
  headers: Record<string, string>;
  bodyText: string;
  bodyBuffer?: ArrayBuffer;
}

export interface HttpClient {
  request(req: HttpRequest): Promise<HttpResponse>;
}

export type HttpTransportErrorKind = "network" | "timeout";

const HTTP_TRANSPORT_ERROR_NAME = "HttpTransportError";

export interface HttpTransportErrorOptions extends ErrorOptions {
  /** True only when the host knows the physical attempt has stopped before retry. */
  retryableAttempt?: boolean;
}

/** Host-neutral failure raised when an HTTP transport cannot produce a response. */
export class HttpTransportError extends Error {
  readonly kind: HttpTransportErrorKind;
  readonly retryableAttempt: boolean;

  constructor(
    kind: HttpTransportErrorKind,
    message: string,
    options: HttpTransportErrorOptions = {},
  ) {
    super(message, options);
    this.name = HTTP_TRANSPORT_ERROR_NAME;
    this.kind = kind;
    this.retryableAttempt = options.retryableAttempt === true;
  }
}

/** Structural guard so errors remain recognizable across package/realm boundaries. */
export function isHttpTransportError(error: unknown): error is HttpTransportError {
  if (!error || typeof error !== "object") return false;
  const candidate = error as Partial<HttpTransportError>;
  return (
    candidate.name === HTTP_TRANSPORT_ERROR_NAME &&
    typeof candidate.message === "string" &&
    (candidate.kind === "network" || candidate.kind === "timeout") &&
    (candidate.retryableAttempt === undefined ||
      typeof candidate.retryableAttempt === "boolean")
  );
}

export interface StorageEntry {
  path: string;
  type: "file" | "folder";
}

export interface StorageAdapter {
  normalizePath(path: string): string;
  readText(path: string): Promise<string>;
  writeText(path: string, content: string): Promise<void>;
  /** Writes host-private data with a restrictive mode when the host supports it. */
  writeTextWithMode?(path: string, content: string, mode: number): Promise<void>;
  writeTextAtomic?(path: string, content: string): Promise<void>;
  appendText?(path: string, content: string): Promise<void>;
  exists(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
  remove(path: string): Promise<void>;
  rename(from: string, to: string): Promise<void>;
  list?(dir: string): Promise<StorageEntry[]>;
  readBinary?(path: string): Promise<ArrayBuffer>;
  writeBinary?(path: string, content: ArrayBuffer): Promise<void>;
}

export interface SecretProvider {
  getSecret(key: string): Promise<string | null>;
  setSecret?(key: string, value: string): Promise<void>;
  deleteSecret?(key: string): Promise<void>;
}

export type ProgressStage =
  | "fetch-metadata"
  | "fetch-recent"
  | "enrich-abstract"
  | "filter"
  | "personal-novelty"
  | "fetch-content"
  | "summarize-daily"
  | "summarize-detail"
  | "write-detail";

export type IdleReason = "weekend" | "disabled";

export interface ProgressReporter {
  setTask(title: string, detail?: string): void;
  setBatch(currentDay: number, totalDays: number, date: string): void;
  setStage(stage: ProgressStage, current?: number, total?: number): void;
  setComplete(message?: string): void;
  setError(message: string): void;
  setIdle(lastCompletedDate?: string, reason?: IdleReason): void;
  setDisabled(): void;
}

export interface ResourceOpenOptions {
  newLeaf?: boolean;
}

export interface ResourceOpener {
  openNote(path: string, opts?: ResourceOpenOptions): Promise<void>;
  openDailyReport(path: string, opts?: ResourceOpenOptions): Promise<void>;
  openUrl(url: string): Promise<void>;
}

export interface MarkupParser {
  parseFromString(markup: string, mimeType: "text/html" | "text/xml"): Document;
}

export interface HostAdapters {
  http: HttpClient;
  storage: StorageAdapter;
  secrets: SecretProvider;
  progress: ProgressReporter;
  opener: ResourceOpener;
  markupParser: MarkupParser;
}
