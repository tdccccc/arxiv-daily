import {
  constants as fsConstants,
  existsSync,
  fstatSync,
  lstatSync,
  realpathSync,
  statSync,
} from "node:fs";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import type {
  StorageAdapter,
  StorageEntry,
  StorageNamespaceGuard,
} from "@arxiv-daily/core";

export interface NodeStorageAdapterOptions {
  /** Test-only deterministic seam after the final parent fd is opened. */
  afterFinalParentOpened?: () => Promise<void> | void;
  /** Test-only deterministic seam after the target is created but before write. */
  afterTargetCreated?: () => Promise<void> | void;
  /** Test-only fault-injection seam for the atomic install rename. */
  renameAtomic?: (from: string, to: string) => Promise<void>;
}

export class NodeStorageAdapter implements StorageAdapter {
  private rootDir: string;
  readonly createTextExclusive?: (
    storagePath: string,
    content: string,
  ) => Promise<boolean>;

  constructor(
    rootDir: string = process.cwd(),
    private readonly options: NodeStorageAdapterOptions = {},
  ) {
    this.rootDir = path.resolve(rootDir);
    if (supportsDescriptorAnchoredCreate()) {
      this.createTextExclusive = (storagePath, content) =>
        this.createTextExclusiveLinux(storagePath, content);
    }
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

  async writeTextWithMode(
    storagePath: string,
    content: string,
    mode: number,
  ): Promise<void> {
    const target = this.toFsPath(storagePath);
    await fs.writeFile(target, content, { encoding: "utf8", mode });
    await fs.chmod(target, mode);
  }

  async writeTextAtomic(
    storagePath: string,
    content: string,
    mode?: number,
  ): Promise<void> {
    const target = this.toFsPath(storagePath);
    const suffix = crypto.randomUUID().replace(/-/g, "");
    const tmp = this.toFsPath(`${storagePath}.tmp-${suffix}`);
    const bak = this.toFsPath(`${storagePath}.bak-${suffix}`);
    await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
    await fs.writeFile(tmp, content, {
      encoding: "utf8",
      ...(mode === undefined ? {} : { mode }),
    });
    if (mode !== undefined) await fs.chmod(tmp, mode);
    try {
      await (this.options.renameAtomic ?? fs.rename)(tmp, target);
      if (mode !== undefined) await fs.chmod(target, mode);
    } finally {
      await fs.rm(tmp, { force: true }).catch(() => undefined);
      await fs.rm(bak, { force: true }).catch(() => undefined);
    }
  }

  async recoverTextAtomic(
    storagePath: string,
    mode: number,
  ): Promise<void> {
    if (!supportsDescriptorAnchoredCreate() || mode !== 0o600) {
      throw new Error("private descriptor-anchored recovery is unavailable");
    }
    const normalized = validateExclusiveStoragePath(storagePath);
    const parts = normalized.split("/");
    const fileName = parts.pop();
    if (!fileName) throw new Error("private recovery target is invalid");
    const handles: fs.FileHandle[] = [];
    try {
      let rootHandle: fs.FileHandle;
      try {
        rootHandle = await openDirectoryNoFollow(this.rootDir);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
        throw error;
      }
      handles.push(rootHandle);
      const realRoot = await validatedDescriptorPath(rootHandle, storagePath);
      const configuredRealRoot = await fs.realpath(this.rootDir);
      if (realRoot !== configuredRealRoot) {
        throw new Error("private recovery root descriptor is inconsistent");
      }

      let parentHandle = rootHandle;
      for (const part of parts) {
        try {
          const nextHandle = await openDirectoryNoFollow(
            `/proc/self/fd/${parentHandle.fd}/${part}`,
          );
          handles.push(nextHandle);
          parentHandle = nextHandle;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
          throw error;
        }
        assertContained(
          realRoot,
          await validatedDescriptorPath(parentHandle, storagePath),
          storagePath,
        );
      }
      await assertDescriptorCurrent(
        rootHandle,
        parentHandle,
        this.rootDir,
        realRoot,
        parts,
        storagePath,
      );

      let targetHandle: fs.FileHandle;
      try {
        targetHandle = await fs.open(
          `/proc/self/fd/${parentHandle.fd}/${fileName}`,
          fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
        );
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
        throw error;
      }
      handles.push(targetHandle);
      if (!(await targetHandle.stat()).isFile()) {
        throw new Error("private recovery primary is not a file");
      }
      await targetHandle.chmod(mode);
      const actualMode = (await targetHandle.stat()).mode & 0o777;
      if (actualMode !== mode) {
        throw new Error(
          `private recovery mode was not enforced: expected ${mode.toString(8)}, got ${actualMode.toString(8)}`,
        );
      }
      await targetHandle.sync();
      await assertDescriptorCurrent(
        rootHandle,
        parentHandle,
        this.rootDir,
        realRoot,
        parts,
        storagePath,
      );
    } finally {
      for (const handle of handles.reverse()) {
        await handle.close().catch(() => undefined);
      }
    }
  }

  async guardClaimNamespace(claimPath: string): Promise<StorageNamespaceGuard> {
    if (!supportsDescriptorAnchoredCreate()) {
      throw new Error("descriptor-backed claim namespace guard is unavailable");
    }
    const normalized = validateExclusiveStoragePath(claimPath);
    const relativeParent = path.posix.dirname(normalized);
    const logicalParent = path.join(this.rootDir, ...relativeParent.split("/"));
    const handle = await openDirectoryNoFollow(logicalParent);
    const expectedRoot = await fs.realpath(this.rootDir);
    const expectedParent = await validatedDescriptorPath(handle, claimPath);
    assertContained(expectedRoot, expectedParent, claimPath);
    const info = await handle.stat();
    let released = false;
    return {
      assertCurrent: () => {
        if (released) throw new Error("delivery claim namespace guard was released");
        assertNoSymlinkComponentsSync(this.rootDir, relativeParent);
        const currentRoot = realpathSync(this.rootDir);
        const currentParent = realpathSync(logicalParent);
        const logicalInfo = statSync(logicalParent);
        const descriptorInfo = fstatSync(handle.fd);
        const descriptorPath = realpathSync(`/proc/self/fd/${handle.fd}`);
        if (
          currentRoot !== expectedRoot ||
          currentParent !== expectedParent ||
          descriptorPath !== expectedParent ||
          logicalInfo.dev !== info.dev ||
          logicalInfo.ino !== info.ino ||
          descriptorInfo.dev !== info.dev ||
          descriptorInfo.ino !== info.ino
        ) {
          throw new Error("delivery claim namespace was replaced");
        }
      },
      release: async () => {
        if (released) return;
        released = true;
        await handle.close();
      },
    };
  }

  private async createTextExclusiveLinux(
    storagePath: string,
    content: string,
  ): Promise<boolean> {
    const normalized = validateExclusiveStoragePath(storagePath);
    await fs.mkdir(this.rootDir, { recursive: true, mode: 0o700 });

    const handles: fs.FileHandle[] = [];
    let createdTarget: string | undefined;
    try {
      const rootHandle = await openDirectoryNoFollow(this.rootDir);
      handles.push(rootHandle);
      const realRoot = await validatedDescriptorPath(rootHandle, this.rootDir);
      const configuredRealRoot = await fs.realpath(this.rootDir);
      if (realRoot !== configuredRealRoot) {
        throw new Error("exclusive-create root descriptor is inconsistent");
      }

      const parts = normalized.split("/");
      const fileName = parts.pop();
      if (!fileName) throw new Error("exclusive-create target is invalid");
      let parentHandle = rootHandle;

      for (const part of parts) {
        const descriptorParent = `/proc/self/fd/${parentHandle.fd}`;
        const next = `${descriptorParent}/${part}`;
        try {
          const nextHandle = await openDirectoryNoFollow(next);
          handles.push(nextHandle);
          parentHandle = nextHandle;
        } catch (error) {
          const code = (error as NodeJS.ErrnoException).code;
          if (code !== "ENOENT") {
            if (code === "ELOOP" || code === "ENOTDIR") {
              throw new Error("exclusive-create parent is a symlink or not a directory");
            }
            throw error;
          }
          try {
            await fs.mkdir(next, { mode: 0o700 });
          } catch (mkdirError) {
            if ((mkdirError as NodeJS.ErrnoException).code !== "EEXIST") {
              throw mkdirError;
            }
          }
          const nextHandle = await openDirectoryNoFollow(next);
          handles.push(nextHandle);
          parentHandle = nextHandle;
        }
        const realParent = await validatedDescriptorPath(parentHandle, storagePath);
        assertContained(realRoot, realParent, storagePath);
      }

      await this.options.afterFinalParentOpened?.();
      await assertDescriptorCurrent(
        rootHandle,
        parentHandle,
        this.rootDir,
        realRoot,
        parts,
        storagePath,
      );
      const target = `/proc/self/fd/${parentHandle.fd}/${fileName}`;
      const flags =
        fsConstants.O_CREAT |
        fsConstants.O_EXCL |
        fsConstants.O_WRONLY |
        fsConstants.O_NOFOLLOW;
      let fileHandle: fs.FileHandle;
      try {
        fileHandle = await fs.open(target, flags, 0o600);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "EEXIST") {
          await assertDescriptorCurrent(
            rootHandle,
            parentHandle,
            this.rootDir,
            realRoot,
            parts,
            storagePath,
          );
          return false;
        }
        throw error;
      }
      handles.push(fileHandle);
      createdTarget = target;
      await this.options.afterTargetCreated?.();
      await assertDescriptorCurrent(
        rootHandle,
        parentHandle,
        this.rootDir,
        realRoot,
        parts,
        storagePath,
      );
      await fileHandle.writeFile(content, "utf8");
      await fileHandle.chmod(0o600);
      await fileHandle.sync();
      await assertDescriptorCurrent(
        rootHandle,
        parentHandle,
        this.rootDir,
        realRoot,
        parts,
        storagePath,
      );
      return true;
    } catch (error) {
      if (createdTarget) {
        await fs.rm(createdTarget, { force: true }).catch(() => undefined);
      }
      throw error;
    } finally {
      for (const handle of handles.reverse()) {
        await handle.close().catch(() => undefined);
      }
    }
  }

  async appendText(storagePath: string, content: string): Promise<void> {
    const target = this.toFsPath(storagePath);
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.appendFile(target, content, "utf8");
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

function supportsDescriptorAnchoredCreate(): boolean {
  return (
    process.platform === "linux" &&
    typeof fsConstants.O_DIRECTORY === "number" &&
    typeof fsConstants.O_NOFOLLOW === "number" &&
    existsSync("/proc/self/fd")
  );
}

async function openDirectoryNoFollow(target: string): Promise<fs.FileHandle> {
  return fs.open(
    target,
    fsConstants.O_RDONLY | fsConstants.O_DIRECTORY | fsConstants.O_NOFOLLOW,
  );
}

async function assertDescriptorCurrent(
  rootDescriptor: fs.FileHandle,
  parentDescriptor: fs.FileHandle,
  configuredRoot: string,
  expectedRealRoot: string,
  logicalParts: string[],
  input: string,
): Promise<void> {
  const logicalPath = path.join(configuredRoot, ...logicalParts);
  let logical: fs.FileHandle | undefined;
  try {
    const configuredRealRoot = await fs.realpath(configuredRoot);
    if (configuredRealRoot !== expectedRealRoot) {
      throw new Error(`exclusive-create root was replaced: ${input}`);
    }
    logical = await openDirectoryNoFollow(logicalPath);
    const [
      rootInfo,
      configuredRootInfo,
      descriptorInfo,
      logicalInfo,
      descriptorPath,
      logicalRealPath,
    ] = await Promise.all([
      rootDescriptor.stat(),
      fs.stat(configuredRoot),
      parentDescriptor.stat(),
      logical.stat(),
      fs.realpath(`/proc/self/fd/${parentDescriptor.fd}`),
      fs.realpath(logicalPath),
    ]);
    if (
      rootInfo.dev !== configuredRootInfo.dev ||
      rootInfo.ino !== configuredRootInfo.ino ||
      descriptorInfo.dev !== logicalInfo.dev ||
      descriptorInfo.ino !== logicalInfo.ino ||
      descriptorPath !== logicalRealPath ||
      descriptorPath !== path.join(expectedRealRoot, ...logicalParts)
    ) {
      throw new Error(`exclusive-create parent was replaced: ${input}`);
    }
    assertContained(expectedRealRoot, descriptorPath, input);
  } catch (error) {
    if (
      (error as NodeJS.ErrnoException).code === "ELOOP" ||
      (error as NodeJS.ErrnoException).code === "ENOTDIR" ||
      (error as NodeJS.ErrnoException).code === "ENOENT"
    ) {
      throw new Error(`exclusive-create parent was replaced: ${input}`);
    }
    throw error;
  } finally {
    await logical?.close().catch(() => undefined);
  }
}

async function validatedDescriptorPath(
  handle: fs.FileHandle,
  input: string,
): Promise<string> {
  const descriptorPath = `/proc/self/fd/${handle.fd}`;
  const real = await fs.realpath(descriptorPath);
  const info = await handle.stat();
  if (!info.isDirectory()) {
    throw new Error(`exclusive-create descriptor is not a directory: ${input}`);
  }
  return real;
}

function assertNoSymlinkComponentsSync(
  root: string,
  relativeParent: string,
): void {
  if (lstatSync(root).isSymbolicLink()) {
    throw new Error("delivery claim namespace root is a symlink");
  }
  let current = root;
  for (const part of relativeParent.split("/").filter(Boolean)) {
    current = path.join(current, part);
    if (lstatSync(current).isSymbolicLink()) {
      throw new Error("delivery claim namespace contains a symlink");
    }
  }
}

function assertContained(base: string, candidate: string, input: string): void {
  const relative = path.relative(base, candidate);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`exclusive-create parent escapes root: ${input}`);
  }
}

function validateExclusiveStoragePath(input: string): string {
  const candidate = input.replace(/\\/g, "/");
  if (
    !candidate ||
    candidate.startsWith("/") ||
    path.isAbsolute(input) ||
    candidate.includes("\u0000")
  ) {
    throw new Error(`exclusive-create path escapes root: ${input}`);
  }
  const parts = candidate.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error(`exclusive-create path escapes root: ${input}`);
  }
  const normalized = path.posix.normalize(candidate);
  if (normalized !== candidate || normalized.startsWith("../")) {
    throw new Error(`exclusive-create path escapes root: ${input}`);
  }
  return normalized;
}

function normalizeStoragePath(input: string): string {
  return input
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part && part !== ".")
    .join("/");
}
