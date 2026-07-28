import * as os from "node:os";
import * as path from "node:path";

/** Directory containing config.toml (XDG / APPDATA). */
export function resolveCliConfigDir(
  env: Record<string, string | undefined> = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  if (platform === "win32") {
    const appData = env.APPDATA?.trim() || path.join(os.homedir(), "AppData", "Roaming");
    return path.join(appData, "arxiv-daily");
  }
  const xdg = env.XDG_CONFIG_HOME?.trim();
  const base = xdg || path.join(os.homedir(), ".config");
  return path.join(base, "arxiv-daily");
}

/** Fixed path to the CLI TOML config file. */
export function resolveCliConfigPath(
  env: Record<string, string | undefined> = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  return path.join(resolveCliConfigDir(env, platform), "config.toml");
}
