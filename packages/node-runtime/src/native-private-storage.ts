import type { StorageNamespaceGuard } from "@arxiv-daily/core";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { randomUUID } from "node:crypto";
import { NodeFileLock } from "./file-lock";
export { getNativeStorageBinding } from "./native-storage-loader";

export interface NativeFile {
  writeText(content: string): void;
  restrict(): void;
  isPrivate(): boolean;
  close(): void;
}
export interface NativeDirectory {
  assertCurrent(): void;
  createFile(name: string): NativeFile | null;
  openFile(name: string): NativeFile | null;
  remove(name: string): boolean;
  link(from: string, to: string): boolean;
  rename(from: string, to: string): void;
  list(): string[];
  sync(): void;
  close(): void;
}
export interface NativeStorageBinding {
  version: 1;
  openDirectory(root: string, parent: string, create: boolean): NativeDirectory;
}
export interface NativePrivateStorageOptions {
  lockRoot?: string;
  afterFinalParentOpened?: () => Promise<void> | void;
  afterTargetCreated?: () => Promise<void> | void;
  afterTemporaryFileReady?: (path: string) => Promise<void> | void;
  afterBackupFileReady?: (path: string) => Promise<void> | void;
  renameAtomic?: (from: string, to: string) => Promise<void>;
}

export class NativePrivateStorage {
  constructor(
    private readonly root: string,
    private readonly binding: NativeStorageBinding,
    private readonly options: NativePrivateStorageOptions = {},
  ) {
    this.root = path.resolve(root);
  }

  async createTextExclusive(input: string, content: string): Promise<boolean> {
    const { parent, name } = storagePath(input);
    await fs.mkdir(this.root, { recursive: true, mode: 0o700 });
    const directory = this.binding.openDirectory(this.root, parent, true);
    let file: NativeFile | null = null;
    try {
      await this.options.afterFinalParentOpened?.();
      directory.assertCurrent();
      file = directory.createFile(name);
      if (!file) {
        directory.assertCurrent();
        return false;
      }
      await this.options.afterTargetCreated?.();
      directory.assertCurrent();
      file.writeText(content);
      directory.sync();
      directory.assertCurrent();
      return true;
    } catch (error) {
      if (file) removeBestEffort(directory, name);
      throw error;
    } finally {
      file?.close();
      directory.close();
    }
  }

  async writeTextAtomic(input: string, content: string, mode: number): Promise<void> {
    requirePrivateMode(mode);
    const { normalized, parent, name } = storagePath(input);
    await this.withStateLock(normalized, async () => {
      const directory = this.binding.openDirectory(this.root, parent, true);
      const suffix = randomUUID().replace(/-/g, "");
      const temporary = `${name}.tmp-${suffix}`;
      const backup = `${name}.bak-${suffix}`;
      let file: NativeFile | null = null;
      let hasTemporary = false;
      let hasBackup = false;
      const logicalPath = (leaf: string) => path.join(this.root, parent, leaf);
      try {
        recover(directory, name);
        directory.assertCurrent();
        file = directory.createFile(temporary);
        if (!file) throw new Error("private temporary file already exists");
        hasTemporary = true;
        file.writeText(content);
        await this.options.afterTemporaryFileReady?.(logicalPath(temporary));
        directory.assertCurrent();
        const previous = directory.openFile(name);
        if (previous) {
          try { previous.restrict(); } finally { previous.close(); }
          if (!directory.link(name, backup)) throw new Error("private backup already exists");
          hasBackup = true;
          await this.options.afterBackupFileReady?.(logicalPath(backup));
          directory.assertCurrent();
        }
        if (this.options.renameAtomic) {
          await this.options.renameAtomic(logicalPath(temporary), logicalPath(name));
        } else {
          directory.rename(temporary, name);
        }
        hasTemporary = false;
        directory.sync();
        directory.assertCurrent();
      } finally {
        file?.close();
        if (hasTemporary) removeBestEffort(directory, temporary);
        if (hasBackup) removeBestEffort(directory, backup);
        directory.close();
      }
    });
  }

  async recoverTextAtomic(input: string, mode: number): Promise<void> {
    requirePrivateMode(mode);
    const { normalized, parent, name } = storagePath(input);
    try { await fs.stat(this.root); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return; throw error; }
    await this.withStateLock(normalized, async () => {
      let directory: NativeDirectory;
      try { directory = this.binding.openDirectory(this.root, parent, false); }
      catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return; throw error; }
      try {
        directory.assertCurrent();
        recover(directory, name);
        directory.sync();
        directory.assertCurrent();
      } finally { directory.close(); }
    });
  }

  async guardClaimNamespace(input: string): Promise<StorageNamespaceGuard> {
    const { parent } = storagePath(input);
    const directory = this.binding.openDirectory(this.root, parent, false);
    return {
      assertCurrent: () => directory.assertCurrent(),
      release: async () => { directory.close(); },
    };
  }

  private async withStateLock<T>(key: string, work: () => Promise<T>): Promise<T> {
    const lease = await new NodeFileLock(this.root, this.options).acquire(`private-state:${key}`, { wait: true });
    if (!lease) throw new Error("private state lock is unavailable");
    try { return await work(); } finally { await lease.release(); }
  }
}

function storagePath(input: string) {
  const normalized = input.replace(/\\/g, "/");
  const parts = normalized.split("/");
  if (path.isAbsolute(input) || normalized.includes("\0") || parts.some(part => !part || part === "." || part === ".." || part.includes(":"))) {
    throw new Error("private storage path escapes root (invalid vault-relative path)");
  }
  return { normalized, name: parts.pop()!, parent: parts.join("/") };
}

function requirePrivateMode(mode: number): void {
  if (mode !== 0o600) throw new Error("native private storage requires mode 0600");
}

function removeBestEffort(directory: NativeDirectory, name: string): void {
  try { directory.remove(name); } catch { /* Keep the original operation failure. */ }
}

function recover(directory: NativeDirectory, name: string): void {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const backupPattern = new RegExp(`^${escaped}\\.bak(?:-[0-9a-f]+)?$`);
  const temporaryPattern = new RegExp(`^${escaped}\\.tmp(?:-[0-9a-f]+)?$`);
  const names = directory.list();
  const primary = directory.openFile(name);
  let hasPrimary = primary !== null;
  if (primary) { try { primary.restrict(); } finally { primary.close(); } }
  for (const backup of names.filter(value => backupPattern.test(value)).sort().reverse()) {
    const file = directory.openFile(backup);
    if (!file) continue;
    try { file.restrict(); } finally { file.close(); }
    if (!hasPrimary) {
      directory.rename(backup, name);
      hasPrimary = true;
    } else {
      directory.remove(backup);
    }
  }
  for (const temporary of names.filter(value => temporaryPattern.test(value))) {
    const file = directory.openFile(temporary);
    if (!file) continue;
    try { file.restrict(); } finally { file.close(); }
    directory.remove(temporary);
  }
}
