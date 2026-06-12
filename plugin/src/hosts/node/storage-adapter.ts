import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { StorageAdapter, StorageEntry } from "../../core/adapters";

export class NodeStorageAdapter implements StorageAdapter {
  private rootDir: string;

  constructor(rootDir: string = process.cwd()) {
    this.rootDir = path.resolve(rootDir);
  }

  normalizePath(input: string): string {
    return normalizeStoragePath(input);
  }

  async readText(storagePath: string): Promise<string> {
    return await fs.readFile(this.toFsPath(storagePath), "utf8");
  }

  async writeText(storagePath: string, content: string): Promise<void> {
    await fs.writeFile(this.toFsPath(storagePath), content, "utf8");
  }

  async exists(storagePath: string): Promise<boolean> {
    try {
      await fs.stat(this.toFsPath(storagePath));
      return true;
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return false;
      throw e;
    }
  }

  async mkdir(storagePath: string): Promise<void> {
    await fs.mkdir(this.toFsPath(storagePath), { recursive: true });
  }

  async remove(storagePath: string): Promise<void> {
    await fs.rm(this.toFsPath(storagePath), { recursive: true, force: true });
  }

  async rename(from: string, to: string): Promise<void> {
    const target = this.toFsPath(to);
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.rename(this.toFsPath(from), target);
  }

  async list(dir: string): Promise<StorageEntry[]> {
    const normalizedDir = this.normalizePath(dir);
    const entries = await fs.readdir(this.toFsPath(normalizedDir), {
      withFileTypes: true,
    });
    return entries.map((entry) => ({
      path: normalizedDir ? `${normalizedDir}/${entry.name}` : entry.name,
      type: entry.isDirectory() ? "folder" : "file",
    }));
  }

  async readBinary(storagePath: string): Promise<ArrayBuffer> {
    const buffer = await fs.readFile(this.toFsPath(storagePath));
    return buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength,
    );
  }

  async writeBinary(storagePath: string, content: ArrayBuffer): Promise<void> {
    await fs.writeFile(this.toFsPath(storagePath), Buffer.from(content));
  }

  private toFsPath(storagePath: string): string {
    const normalized = this.normalizePath(storagePath);
    const resolved = path.resolve(this.rootDir, normalized);
    const relative = path.relative(this.rootDir, resolved);
    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new Error(`storage path escapes root: ${storagePath}`);
    }
    return resolved;
  }
}

function normalizeStoragePath(input: string): string {
  return input
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part && part !== ".")
    .join("/");
}
