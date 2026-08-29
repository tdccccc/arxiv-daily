import fsPromises from "node:fs/promises";
import path from "node:path";

/**
 * A settings store shaped the way it was before the sidecar existed. Loading it
 * is what gives the migration something to prove.
 */
export function legacySettingsFixture() {
  return {
    settings: {
      llm: { provider: "openai", model: "gpt-4o-mini", apiKey: "" },
      arxiv: { categories: ["astro-ph.GA"], timezone: "UTC" },
      output: { dailyDir: "arxiv-daily/daily", papersDir: "arxiv-daily/papers" },
      schedule: { enabled: false },
      advanced: {},
      email: {},
      detailSelection: {},
    },
  };
}

export async function installSettingsFixture({ vaultPath, pluginId, fs = fsPromises, data }) {
  if (typeof vaultPath !== "string" || !path.isAbsolute(vaultPath)) {
    throw new TypeError(`vault path must be absolute: ${String(vaultPath)}`);
  }
  const target = path.join(vaultPath, ".obsidian", "plugins", pluginId, "data.json");
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, JSON.stringify(data, null, 2));
  return target;
}
