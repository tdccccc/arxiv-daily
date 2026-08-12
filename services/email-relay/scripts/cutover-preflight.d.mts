// Type declarations for the untranspiled preflight script, used by vitest tests.
export interface PreflightCheckResult {
  name: string;
  status: "pass" | "fail" | "skip";
  detail?: string;
}

export interface RunOutcome {
  status: number;
  stdout: string;
  stderr: string;
}

export interface PreflightDeps {
  runCommand: (cmd: string, args: string[], opts?: { cwd?: string }) => RunOutcome;
  readTextFile: (path: string) => string;
  httpGet: (url: string, timeoutMs?: number) => Promise<{ status: number; text: string }>;
  tomlPath?: string;
}

export interface PreflightOptions {
  remoteBaseUrl?: string;
  skipLogin?: boolean;
}

export const RELAY_ROOT: string;
export const REQUIRED_TOML_NAMES: string[];
export const REQUIRED_SECRET_NAMES: string[];

export function checkSourceReadonly(source: string): string[];

export function createPreflight(
  deps: PreflightDeps,
): (options?: PreflightOptions) => Promise<{ results: PreflightCheckResult[]; ok: boolean }>;

export function main(): Promise<void>;
