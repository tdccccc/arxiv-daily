// Type declarations for the untranspiled deploy script, used by vitest tests.
export interface DeployStep {
  name: string;
  status: "pass" | "fail" | "warn" | "skip";
  detail?: string;
}

export interface DeployRunOutcome {
  status: number;
  stdout: string;
  stderr: string;
}

export interface DeployDeps {
  runCommand: (
    cmd: string,
    args: string[],
    opts?: { cwd?: string; inherit?: boolean },
  ) => DeployRunOutcome;
  readTextFile: (path: string) => string;
  httpGet: (url: string, timeoutMs?: number) => Promise<{ status: number; text: string }>;
  tomlPath?: string;
}

export interface DeployOptions {
  dryRun?: boolean;
  identity?: string;
  remoteBaseUrl?: string;
}

export interface DeployResult {
  ok: boolean;
  steps: DeployStep[];
  identity?: string;
  command?: string;
}

export interface ParsedDeployArgs extends DeployOptions {
  errors: string[];
}

export const RELAY_ROOT: string;
export const BUILD_IDENTITY_PATTERN: RegExp;

export function parseArgs(argv: string[]): ParsedDeployArgs;

export function createDeploy(
  deps: DeployDeps,
): (options?: DeployOptions) => Promise<DeployResult>;

export function main(): Promise<void>;
