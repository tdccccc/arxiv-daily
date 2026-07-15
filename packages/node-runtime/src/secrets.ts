import type { SecretProvider } from "@arxiv-daily/core";

const MEMORY_ONLY_WARNING =
  "Node EnvSecretProvider stores secrets in memory only; they will not persist across restarts";

export class EnvSecretProvider implements SecretProvider {
  private readonly memory = new Map<string, string>();

  constructor(
    private env: Record<string, string | undefined> = process.env,
    private prefix = "ARXIV_DAILY",
  ) {}

  async getSecret(key: string): Promise<string | null> {
    for (const candidate of this.candidates(key)) {
      const memoryValue = this.memory.get(candidate);
      if (memoryValue) return memoryValue;
      const value = this.env[candidate];
      if (value) return value;
    }
    return null;
  }

  async setSecret(key: string, value: string): Promise<void> {
    console.warn(MEMORY_ONLY_WARNING);
    this.memory.set(key, value);
  }

  async deleteSecret(key: string): Promise<void> {
    for (const candidate of this.candidates(key)) {
      this.memory.delete(candidate);
    }
    this.memory.delete(key);
  }

  private candidates(key: string): string[] {
    const envKey = toEnvKey(key);
    return [key, envKey, `${this.prefix}_${envKey}`];
  }
}

function toEnvKey(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/[^A-Za-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toUpperCase();
}
