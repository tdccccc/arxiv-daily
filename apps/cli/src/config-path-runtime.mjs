import * as os from "node:os";
import * as path from "node:path";

/** Directory containing config.toml (XDG / APPDATA). */
export function resolveCliConfigDir(
  env = process.env,
  platform = process.platform,
) {
  if (platform === "win32") {
    const appData = env.APPDATA?.trim() || path.join(os.homedir(), "AppData", "Roaming");
    return path.join(appData, "arxiv-daily");
  }
  const xdg = env.XDG_CONFIG_HOME?.trim();
  const base = xdg || path.join(os.homedir(), ".config");
  return path.join(base, "arxiv-daily");
}

/** Save root suggested on first setup, shared by `init` and the workbench. */
export function defaultCliVaultRoot() {
  return path.join(os.homedir(), "arxiv-daily");
}

/** Fixed path to the CLI TOML config file. */
export function resolveCliConfigPath(
  env = process.env,
  platform = process.platform,
) {
  return path.join(resolveCliConfigDir(env, platform), "config.toml");
}
