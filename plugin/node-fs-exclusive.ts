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
import type { StorageNamespaceGuard } from "@arxiv-daily/core";

export interface FileSystemDataAdapter {
  getBasePath(): string;
  read(path: string): Promise<string>;
  write(path: string, content: string): Promise<void>;
  exists(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
  rename(from: string, to: string): Promise<void>;
  remove(path: string): Promise<void>;
  list(path: string): Promise<unknown>;
}

export interface DesktopExclusiveCreateOptions {
  /** Test-only deterministic seam after the final parent fd is opened. */
  afterFinalParentOpened?: () => Promise<void> | void;
  /** Test-only deterministic seam after the target is created but before write. */
  afterTargetCreated?: () => Promise<void> | void;
}

export interface DesktopAtomicWriteOptions {
  /** Test-only seam after the private temporary file is durable. */
  afterTemporaryFileReady?: (path: string) => Promise<void> | void;
  /** Test-only seam after a private hard-link backup exists beside the target. */
  afterBackupFileReady?: (path: string) => Promise<void> | void;
}

export function supportsDesktopExclusiveCreate(): boolean {
  return (
    process.platform === "linux" &&
    typeof fsConstants.O_DIRECTORY === "number" &&
    typeof fsConstants.O_NOFOLLOW === "number" &&
    existsSync("/proc/self/fd")
  );
}

/** Structural desktop FileSystemAdapter check; mobile DataAdapters lack getBasePath. */
export function isFileSystemDataAdapter(
  value: unknown,
): value is FileSystemDataAdapter {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.getBasePath === "function" &&
    typeof candidate.read === "function" &&
    typeof candidate.write === "function" &&
    typeof candidate.exists === "function" &&
    typeof candidate.mkdir === "function" &&
    typeof candidate.rename === "function" &&
    typeof candidate.remove === "function" &&
    typeof candidate.list === "function"
  );
}

export async function writeDesktopTextAtomicPrivate(
  adapter: FileSystemDataAdapter,
  vaultPath: string,
  content: string,
  mode: number,
  options: DesktopAtomicWriteOptions = {},
): Promise<void> {
  if (!supportsDesktopExclusiveCreate() || mode !== 0o600) {
    throw new Error("private descriptor-anchored atomic write is unavailable");
  }
  const normalized = validateVaultRelativePath(vaultPath);
  const basePath = adapter.getBasePath();
  if (!basePath || typeof basePath !== "string") {
    throw new Error("filesystem adapter returned an invalid vault base path");
  }
  const configuredBase = path.resolve(basePath);
  await fs.mkdir(configuredBase, { recursive: true, mode: 0o700 });
  const handles: fs.FileHandle[] = [];
  let tmpTarget: string | undefined;
  let backupTarget: string | undefined;
  try {
    const baseHandle = await openDirectoryNoFollow(configuredBase);
    handles.push(baseHandle);
    const realBase = await validatedDescriptorPath(baseHandle, vaultPath);
    if (realBase !== await fs.realpath(configuredBase)) {
      throw new Error("vault base descriptor is inconsistent");
    }

    const parts = normalized.split("/");
    const fileName = parts.pop();
    if (!fileName) throw new Error("invalid vault-relative atomic-write path");
    let parentHandle = baseHandle;
    for (const part of parts) {
      const next = `/proc/self/fd/${parentHandle.fd}/${part}`;
      try {
        const nextHandle = await openDirectoryNoFollow(next);
        handles.push(nextHandle);
        parentHandle = nextHandle;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
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
      assertContained(
        realBase,
        await validatedDescriptorPath(parentHandle, vaultPath),
        vaultPath,
      );
    }

    await assertDescriptorCurrent(
      baseHandle,
      parentHandle,
      configuredBase,
      realBase,
      parts,
      vaultPath,
    );
    const descriptorParent = `/proc/self/fd/${parentHandle.fd}`;
    const target = `${descriptorParent}/${fileName}`;
    await recoverPrivateAtomicArtifacts(descriptorParent, fileName, mode);

    const suffix = crypto.randomUUID().replace(/-/g, "");
    tmpTarget = `${descriptorParent}/${fileName}.tmp-${suffix}`;
    const tmpHandle = await fs.open(
      tmpTarget,
      fsConstants.O_CREAT |
        fsConstants.O_EXCL |
        fsConstants.O_WRONLY |
        fsConstants.O_NOFOLLOW,
      0o600,
    );
    handles.push(tmpHandle);
    await tmpHandle.writeFile(content, "utf8");
    await tmpHandle.chmod(mode);
    await assertPrivateFileMode(tmpHandle, mode);
    await tmpHandle.sync();
    await options.afterTemporaryFileReady?.(tmpTarget);
    await assertDescriptorCurrent(
      baseHandle,
      parentHandle,
      configuredBase,
      realBase,
      parts,
      vaultPath,
    );

    try {
      const targetHandle = await fs.open(
        target,
        fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
      );
      handles.push(targetHandle);
      if (!(await targetHandle.stat()).isFile()) {
        throw new Error("private atomic-write target is not a file");
      }
      await targetHandle.chmod(mode);
      await assertPrivateFileMode(targetHandle, mode);
      backupTarget = `${descriptorParent}/${fileName}.bak-${suffix}`;
      await fs.link(target, backupTarget);
      await fs.chmod(backupTarget, mode);
      await options.afterBackupFileReady?.(backupTarget);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      backupTarget = undefined;
    }

    // This is the only operation that replaces the primary. The previous target
    // remains present until this atomic rename commits the new inode.
    await fs.rename(tmpTarget, target);
    tmpTarget = undefined;
    await tmpHandle.chmod(mode);
    await assertPrivateFileMode(tmpHandle, mode);
    await assertDescriptorCurrent(
      baseHandle,
      parentHandle,
      configuredBase,
      realBase,
      parts,
      vaultPath,
    );
    await parentHandle.sync();
    if (backupTarget) {
      await fs.rm(backupTarget, { force: true });
      backupTarget = undefined;
      await parentHandle.sync();
    }
  } finally {
    if (tmpTarget) await fs.rm(tmpTarget, { force: true }).catch(() => undefined);
    if (backupTarget) {
      await fs.rm(backupTarget, { force: true }).catch(() => undefined);
    }
    for (const handle of handles.reverse()) {
      await handle.close().catch(() => undefined);
    }
  }
}

export async function recoverDesktopTextAtomicPrivate(
  adapter: FileSystemDataAdapter,
  vaultPath: string,
  mode = 0o600,
): Promise<void> {
  if (!supportsDesktopExclusiveCreate() || mode !== 0o600) {
    throw new Error("private descriptor-anchored recovery is unavailable");
  }
  const normalized = validateVaultRelativePath(vaultPath);
  const configuredBase = path.resolve(adapter.getBasePath());
  const parts = normalized.split("/");
  const fileName = parts.pop();
  if (!fileName) throw new Error("invalid vault-relative recovery path");
  const handles: fs.FileHandle[] = [];
  try {
    const baseHandle = await openDirectoryNoFollow(configuredBase);
    handles.push(baseHandle);
    const expectedBase = await validatedDescriptorPath(baseHandle, vaultPath);
    let parentHandle = baseHandle;
    for (const part of parts) {
      const nextHandle = await openDirectoryNoFollow(
        `/proc/self/fd/${parentHandle.fd}/${part}`,
      );
      handles.push(nextHandle);
      parentHandle = nextHandle;
      assertContained(
        expectedBase,
        await validatedDescriptorPath(parentHandle, vaultPath),
        vaultPath,
      );
    }
    await assertDescriptorCurrent(
      baseHandle,
      parentHandle,
      configuredBase,
      expectedBase,
      parts,
      vaultPath,
    );
    await recoverPrivateAtomicArtifacts(
      `/proc/self/fd/${parentHandle.fd}`,
      fileName,
      mode,
    );
    await parentHandle.sync();
  } finally {
    for (const handle of handles.reverse()) {
      await handle.close().catch(() => undefined);
    }
  }
}

export async function guardDesktopClaimNamespace(
  adapter: FileSystemDataAdapter,
  vaultPath: string,
): Promise<StorageNamespaceGuard> {
  if (!supportsDesktopExclusiveCreate()) {
    throw new Error("descriptor-backed claim namespace guard is unavailable");
  }
  const normalized = validateVaultRelativePath(vaultPath);
  const configuredBase = path.resolve(adapter.getBasePath());
  const relativeParent = path.posix.dirname(normalized);
  const logicalParent = path.join(configuredBase, ...relativeParent.split("/"));
  const handle = await openDirectoryNoFollow(logicalParent);
  const expectedBase = await fs.realpath(configuredBase);
  const expectedParent = await validatedDescriptorPath(handle, vaultPath);
  assertContained(expectedBase, expectedParent, vaultPath);
  const info = await handle.stat();
  let released = false;
  return {
    assertCurrent: () => {
      if (released) throw new Error("delivery claim namespace guard was released");
      assertNoSymlinkComponentsSync(configuredBase, relativeParent);
      const currentBase = realpathSync(configuredBase);
      const currentParent = realpathSync(logicalParent);
      const logicalInfo = statSync(logicalParent);
      const descriptorInfo = fstatSync(handle.fd);
      const descriptorPath = realpathSync(`/proc/self/fd/${handle.fd}`);
      if (
        currentBase !== expectedBase ||
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

export async function createDesktopTextExclusive(
  adapter: FileSystemDataAdapter,
  vaultPath: string,
  content: string,
  options: DesktopExclusiveCreateOptions = {},
): Promise<boolean> {
  if (!supportsDesktopExclusiveCreate()) {
    throw new Error("descriptor-anchored exclusive create is unavailable");
  }
  const normalized = validateVaultRelativePath(vaultPath);
  const basePath = adapter.getBasePath();
  if (!basePath || typeof basePath !== "string") {
    throw new Error("filesystem adapter returned an invalid vault base path");
  }
  const configuredBase = path.resolve(basePath);
  await fs.mkdir(configuredBase, { recursive: true, mode: 0o700 });
  const handles: fs.FileHandle[] = [];
  let createdTarget: string | undefined;
  try {
    const baseHandle = await openDirectoryNoFollow(configuredBase);
    handles.push(baseHandle);
    const realBase = await validatedDescriptorPath(baseHandle, vaultPath);
    if (realBase !== await fs.realpath(configuredBase)) {
      throw new Error("vault base descriptor is inconsistent");
    }

    const parts = normalized.split("/");
    const fileName = parts.pop();
    if (!fileName) throw new Error("invalid vault-relative exclusive-create path");
    let parentHandle = baseHandle;
    for (const part of parts) {
      const next = `/proc/self/fd/${parentHandle.fd}/${part}`;
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
      const realParent = await validatedDescriptorPath(parentHandle, vaultPath);
      assertContained(realBase, realParent, vaultPath);
    }

    await options.afterFinalParentOpened?.();
    await assertDescriptorCurrent(
      baseHandle,
      parentHandle,
      configuredBase,
      realBase,
      parts,
      vaultPath,
    );
    const target = `/proc/self/fd/${parentHandle.fd}/${fileName}`;
    let fileHandle: fs.FileHandle;
    try {
      fileHandle = await fs.open(
        target,
        fsConstants.O_CREAT |
          fsConstants.O_EXCL |
          fsConstants.O_WRONLY |
          fsConstants.O_NOFOLLOW,
        0o600,
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") {
        await assertDescriptorCurrent(
          baseHandle,
          parentHandle,
          configuredBase,
          realBase,
          parts,
          vaultPath,
        );
        return false;
      }
      throw error;
    }
    handles.push(fileHandle);
    createdTarget = target;
    await options.afterTargetCreated?.();
    await assertDescriptorCurrent(
      baseHandle,
      parentHandle,
      configuredBase,
      realBase,
      parts,
      vaultPath,
    );
    await fileHandle.writeFile(content, "utf8");
    await fileHandle.chmod(0o600);
    await fileHandle.sync();
    await assertDescriptorCurrent(
      baseHandle,
      parentHandle,
      configuredBase,
      realBase,
      parts,
      vaultPath,
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

async function recoverPrivateAtomicArtifacts(
  descriptorParent: string,
  fileName: string,
  mode: number,
): Promise<void> {
  const target = `${descriptorParent}/${fileName}`;
  const entries = await fs.readdir(descriptorParent);
  const backups = entries
    .filter((entry) =>
      entry === `${fileName}.bak` ||
      new RegExp(`^${escapeRegExp(fileName)}\\.bak-[0-9a-f]+$`).test(entry),
    )
    .sort()
    .reverse();
  const temporaries = entries.filter((entry) =>
    entry === `${fileName}.tmp` ||
    new RegExp(`^${escapeRegExp(fileName)}\\.tmp-[0-9a-f]+$`).test(entry),
  );
  let targetExists = false;
  let targetHandle: fs.FileHandle | undefined;
  try {
    targetHandle = await fs.open(
      target,
      fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
    );
    if (!(await targetHandle.stat()).isFile()) {
      throw new Error("private atomic-write primary is not a file");
    }
    await targetHandle.chmod(mode);
    await assertPrivateFileMode(targetHandle, mode);
    targetExists = true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  } finally {
    await targetHandle?.close().catch(() => undefined);
  }

  for (const backupName of backups) {
    const backup = `${descriptorParent}/${backupName}`;
    const handle = await fs.open(
      backup,
      fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
    );
    try {
      if (!(await handle.stat()).isFile()) {
        throw new Error("private atomic-write backup is not a file");
      }
      await handle.chmod(mode);
      await assertPrivateFileMode(handle, mode);
    } finally {
      await handle.close();
    }
    if (!targetExists) {
      await fs.rename(backup, target);
      await fs.chmod(target, mode);
      targetExists = true;
    } else {
      await fs.rm(backup);
    }
  }

  for (const temporaryName of temporaries) {
    const temporary = `${descriptorParent}/${temporaryName}`;
    const handle = await fs.open(
      temporary,
      fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
    );
    try {
      if (!(await handle.stat()).isFile()) {
        throw new Error("private atomic-write temporary is not a file");
      }
      await handle.chmod(mode);
      await assertPrivateFileMode(handle, mode);
    } finally {
      await handle.close();
    }
    // A backup is authoritative recovery evidence. A lone temporary is an
    // uncommitted future value and must never replace an existing primary.
    await fs.rm(temporary);
  }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function assertPrivateFileMode(
  handle: fs.FileHandle,
  expectedMode: number,
): Promise<void> {
  const actualMode = (await handle.stat()).mode & 0o777;
  if (actualMode !== expectedMode) {
    throw new Error(
      `private atomic-write mode was not enforced: expected ${expectedMode.toString(8)}, got ${actualMode.toString(8)}`,
    );
  }
}

async function openDirectoryNoFollow(target: string): Promise<fs.FileHandle> {
  return fs.open(
    target,
    fsConstants.O_RDONLY | fsConstants.O_DIRECTORY | fsConstants.O_NOFOLLOW,
  );
}

async function assertDescriptorCurrent(
  baseDescriptor: fs.FileHandle,
  parentDescriptor: fs.FileHandle,
  configuredBase: string,
  expectedRealBase: string,
  logicalParts: string[],
  input: string,
): Promise<void> {
  const logicalPath = path.join(configuredBase, ...logicalParts);
  let logical: fs.FileHandle | undefined;
  try {
    const configuredRealBase = await fs.realpath(configuredBase);
    if (configuredRealBase !== expectedRealBase) {
      throw new Error(`vault base was replaced: ${input}`);
    }
    logical = await openDirectoryNoFollow(logicalPath);
    const [
      baseInfo,
      configuredBaseInfo,
      descriptorInfo,
      logicalInfo,
      descriptorPath,
      logicalRealPath,
    ] = await Promise.all([
      baseDescriptor.stat(),
      fs.stat(configuredBase),
      parentDescriptor.stat(),
      logical.stat(),
      fs.realpath(`/proc/self/fd/${parentDescriptor.fd}`),
      fs.realpath(logicalPath),
    ]);
    if (
      baseInfo.dev !== configuredBaseInfo.dev ||
      baseInfo.ino !== configuredBaseInfo.ino ||
      descriptorInfo.dev !== logicalInfo.dev ||
      descriptorInfo.ino !== logicalInfo.ino ||
      descriptorPath !== logicalRealPath ||
      descriptorPath !== path.join(expectedRealBase, ...logicalParts)
    ) {
      throw new Error(`exclusive-create parent was replaced: ${input}`);
    }
    assertContained(expectedRealBase, descriptorPath, input);
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
  const real = await fs.realpath(`/proc/self/fd/${handle.fd}`);
  if (!(await handle.stat()).isDirectory()) {
    throw new Error(`exclusive-create descriptor is not a directory: ${input}`);
  }
  return real;
}

function validateVaultRelativePath(input: string): string {
  const candidate = input.replace(/\\/g, "/");
  if (
    !candidate ||
    candidate.startsWith("/") ||
    path.isAbsolute(input) ||
    candidate.includes("\u0000")
  ) {
    throw new Error(`invalid vault-relative path: ${input}`);
  }
  const parts = candidate.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error(`invalid vault-relative path: ${input}`);
  }
  const normalized = path.posix.normalize(candidate);
  if (normalized !== candidate || normalized.startsWith("../")) {
    throw new Error(`vault path escapes vault: ${input}`);
  }
  return normalized;
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

function assertContained(
  base: string,
  candidate: string,
  input: string,
): void {
  const relative = path.relative(base, candidate);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    if (relative) throw new Error(`exclusive-create path escapes vault: ${input}`);
  }
}
