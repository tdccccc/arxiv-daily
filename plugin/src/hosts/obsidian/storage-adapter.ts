import { normalizePath, type Vault } from "obsidian";
import type { StorageAdapter, StorageEntry } from "@arxiv-daily/core";
import {
  createDesktopTextExclusive,
  guardDesktopClaimNamespace,
  isFileSystemDataAdapter,
  recoverDesktopTextAtomicPrivate,
  supportsDesktopExclusiveCreate,
  writeDesktopTextAtomicPrivate,
  type DesktopAtomicWriteOptions,
  type FileSystemDataAdapter,
} from "../../../node-fs-exclusive";

export interface ObsidianStorageAdapterOptions {
  /** Test-only seams for inspecting private atomic-write artifacts. */
  privateAtomicWrite?: DesktopAtomicWriteOptions;
}

export class ObsidianStorageAdapter implements StorageAdapter {
  readonly createTextExclusive?: (
    path: string,
    content: string,
  ) => Promise<boolean>;
  readonly guardClaimNamespace?: StorageAdapter["guardClaimNamespace"];
  readonly recoverTextAtomic?: StorageAdapter["recoverTextAtomic"];

  constructor(
    private vault: Vault,
    private readonly options: ObsidianStorageAdapterOptions = {},
  ) {
    const adapter = vault.adapter as unknown as FileSystemDataAdapter;
    if (isFileSystemDataAdapter(adapter) && supportsDesktopExclusiveCreate()) {
      this.createTextExclusive = (path, content) =>
        createDesktopTextExclusive(adapter, this.normalizePath(path), content);
      this.guardClaimNamespace = (path) =>
        guardDesktopClaimNamespace(adapter, this.normalizePath(path));
      this.recoverTextAtomic = (path, mode) =>
        recoverDesktopTextAtomicPrivate(adapter, this.normalizePath(path), mode)
          .catch((error) => {
            if (
              error instanceof Error &&
              /ENOENT|no such file/i.test(error.message)
            ) {
              return;
            }
            throw error;
          });
    }
  }

  normalizePath(path: string): string {
    return normalizePath(path);
  }

  async readText(path: string): Promise<string> {
    return await this.vault.adapter.read(this.normalizePath(path));
  }

  async writeText(path: string, content: string): Promise<void> {
    await this.vault.adapter.write(this.normalizePath(path), content);
  }

  async writeTextAtomic(
    path: string,
    content: string,
    mode?: number,
  ): Promise<void> {
    if (mode !== undefined) {
      const adapter = this.vault.adapter as unknown as FileSystemDataAdapter;
      if (!isFileSystemDataAdapter(adapter) || !supportsDesktopExclusiveCreate()) {
        throw new Error("private atomic storage is unavailable on this host");
      }
      await writeDesktopTextAtomicPrivate(
        adapter,
        this.normalizePath(path),
        content,
        mode,
        this.options.privateAtomicWrite,
      );
      return;
    }
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

  private async ensureDirDeep(path: string): Promise<void> {
    const parts = this.normalizePath(path).split("/").filter(Boolean);
    let current = "";
    for (const part of parts) {
      current = current ? `${current}/${part}` : part;
      if (await this.vault.adapter.exists(current)) continue;
      try {
        await this.vault.adapter.mkdir(current);
      } catch (error) {
        if (!(await this.vault.adapter.exists(current))) throw error;
      }
    }
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
