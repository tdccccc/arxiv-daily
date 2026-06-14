import { normalizePath, type Vault } from "obsidian";
import type { StorageAdapter, StorageEntry } from "../../core/adapters";

export class ObsidianStorageAdapter implements StorageAdapter {
  constructor(private vault: Vault) {}

  normalizePath(path: string): string {
    return normalizePath(path);
  }

  async readText(path: string): Promise<string> {
    return await this.vault.adapter.read(this.normalizePath(path));
  }

  async writeText(path: string, content: string): Promise<void> {
    await this.vault.adapter.write(this.normalizePath(path), content);
  }

  async exists(path: string): Promise<boolean> {
    return await this.vault.adapter.exists(this.normalizePath(path));
  }

  async mkdir(path: string): Promise<void> {
    await this.vault.adapter.mkdir(this.normalizePath(path));
  }

  async remove(path: string): Promise<void> {
    await this.vault.adapter.remove(this.normalizePath(path));
  }

  async rename(from: string, to: string): Promise<void> {
    await this.vault.adapter.rename(
      this.normalizePath(from),
      this.normalizePath(to),
    );
  }

  async list(path: string): Promise<StorageEntry[]> {
    const listed = await this.vault.adapter.list(this.normalizePath(path));
    return [
      ...listed.files.map((entry) => ({ path: entry, type: "file" as const })),
      ...listed.folders.map((entry) => ({
        path: entry,
        type: "folder" as const,
      })),
    ];
  }

  async readBinary(path: string): Promise<ArrayBuffer> {
    return await this.vault.adapter.readBinary(this.normalizePath(path));
  }

  async writeBinary(path: string, content: ArrayBuffer): Promise<void> {
    await this.vault.adapter.writeBinary(this.normalizePath(path), content);
  }
}
