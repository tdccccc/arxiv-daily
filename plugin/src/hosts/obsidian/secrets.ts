import type { SecretProvider } from "../../core/adapters";
import type { PluginSettings } from "../../settings/types";

export class ObsidianSettingsSecretProvider implements SecretProvider {
  constructor(
    private getSettings: () => PluginSettings,
    private persistSettings?: () => Promise<void> | void,
  ) {}

  async getSecret(key: string): Promise<string | null> {
    if (!isApiKeySecret(key)) return null;
    const value = this.getSettings().llm.apiKey.trim();
    return value || null;
  }

  async setSecret(key: string, value: string): Promise<void> {
    if (!isApiKeySecret(key)) {
      throw new Error(`unsupported Obsidian secret key: ${key}`);
    }
    this.getSettings().llm.apiKey = value;
    await this.persistSettings?.();
  }

  async deleteSecret(key: string): Promise<void> {
    if (!isApiKeySecret(key)) return;
    this.getSettings().llm.apiKey = "";
    await this.persistSettings?.();
  }
}

function isApiKeySecret(key: string): boolean {
  const normalized = key.replace(/[^A-Za-z0-9]+/g, "").toLowerCase();
  return normalized === "apikey" || normalized === "llmapikey";
}
