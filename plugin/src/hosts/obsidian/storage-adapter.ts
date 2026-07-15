import { normalizePath, type Vault } from "obsidian";
import type { StorageAdapter, StorageEntry } from "@arxiv-daily/core";

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

  async writeTextAtomic(path: string, content: string): Promise<void> {
    const target = this.normalizePath(path);
    const tmp = this.normalizePath(`${path}.tmp`);
    const bak = this.normalizePath(`${path}.bak`);
    await this.vault.adapter.remove(tmp).catch(() => undefined);
    await this.vault.adapter.write(tmp, content);
    try {
      await this.vault.adapter.rename(tmp, target);
      await this.vault.adapter.remove(bak).catch(() => undefined);
      return;
    } catch (e) {
      if (!(await this.vault.adapter.exists(target))) throw e;
    }

    await this.vault.adapter.remove(bak).catch(() => undefined);
    await this.vault.adapter.rename(target, bak);
    try {
      await this.vault.adapter.rename(tmp, target);
      await this.vault.adapter.remove(bak).catch(() => undefined);
    } catch (e) {
      if (await this.vault.adapter.exists(bak)) {
        await this.vault.adapter.rename(bak, target);
      }
      throw e;
    }
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
