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

export interface StorageEntry {
  path: string;
  type: "file" | "folder";
}

export interface StorageAdapter {
  normalizePath(path: string): string;
  readText(path: string): Promise<string>;
  writeText(path: string, content: string): Promise<void>;
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

export interface HostAdapters {
  http: HttpClient;
  storage: StorageAdapter;
  secrets: SecretProvider;
  progress: ProgressReporter;
  opener: ResourceOpener;
}
