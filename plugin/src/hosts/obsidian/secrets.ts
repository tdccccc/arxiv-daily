import type { SecretProvider } from "@arxiv-daily/core";
import type { PluginSettings } from "@arxiv-daily/core";

export class ObsidianSettingsSecretProvider implements SecretProvider {
  constructor(
    private getSettings: () => PluginSettings,
    private persistSettings?: () => Promise<void> | void,
    private changeSettingValue?: (
      key: string,
      value: unknown,
    ) => Promise<void> | void,
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
    await this.writeApiKey(value);
  }

  async deleteSecret(key: string): Promise<void> {
    if (!isApiKeySecret(key)) return;
    await this.writeApiKey("");
  }

  private async writeApiKey(value: string): Promise<void> {
    if (this.changeSettingValue) {
      await this.changeSettingValue("llm.apiKey", value);
      return;
    }

    const settings = this.getSettings();
    const previous = settings.llm.apiKey;
    settings.llm.apiKey = value;
    try {
      await this.persistSettings?.();
    } catch (error) {
      settings.llm.apiKey = previous;
      throw error;
    }
  }
}

function isApiKeySecret(key: string): boolean {
  const normalized = key.replace(/[^A-Za-z0-9]+/g, "").toLowerCase();
  return normalized === "apikey" || normalized === "llmapikey";
}
