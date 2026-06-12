import type { SecretProvider } from "../../core/adapters";

export class EnvSecretProvider implements SecretProvider {
  constructor(
    private env: Record<string, string | undefined> = process.env,
    private prefix = "ARXIV_DAILY",
  ) {}

  async getSecret(key: string): Promise<string | null> {
    for (const candidate of this.candidates(key)) {
      const value = this.env[candidate];
      if (value) return value;
    }
    return null;
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
