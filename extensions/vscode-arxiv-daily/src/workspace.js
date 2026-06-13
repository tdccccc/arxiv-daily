const ARXIV_DAILY_DIR = "arxiv-daily";
const PAPER_INDEX_PATH = "arxiv-daily/.index/papers.json";

async function findArxivDailyVault(vscodeApi) {
  const folders = vscodeApi.workspace.workspaceFolders ?? [];
  for (const folder of folders) {
    const rootUri = folder.uri;
    const arxivDailyUri = vscodeApi.Uri.joinPath(rootUri, ARXIV_DAILY_DIR);
    if (!(await isDirectoryUri(vscodeApi, arxivDailyUri))) continue;
    return {
      workspaceFolder: folder,
      vaultRootUri: rootUri,
      arxivDailyUri,
      paperIndexUri: vscodeApi.Uri.joinPath(rootUri, ...pathParts(PAPER_INDEX_PATH)),
      storage: createVsCodeStorageAdapter(vscodeApi, rootUri),
    };
  }
  return null;
}

function createVsCodeStorageAdapter(vscodeApi, rootUri) {
  const fs = vscodeApi.workspace.fs;
  return {
    normalizePath: normalizeStoragePath,
    async readText(storagePath) {
      const bytes = await fs.readFile(toUri(vscodeApi, rootUri, storagePath));
      return new TextDecoder().decode(bytes);
    },
    async writeText(storagePath, content) {
      await fs.writeFile(
        toUri(vscodeApi, rootUri, storagePath),
        new TextEncoder().encode(content),
      );
    },
    async exists(storagePath) {
      return await existsUri(fs, toUri(vscodeApi, rootUri, storagePath));
    },
    async mkdir(storagePath) {
      await fs.createDirectory(toUri(vscodeApi, rootUri, storagePath));
    },
    async remove(storagePath) {
      await fs.delete(toUri(vscodeApi, rootUri, storagePath), {
        recursive: true,
        useTrash: false,
      });
    },
    async rename(from, to) {
      await fs.rename(toUri(vscodeApi, rootUri, from), toUri(vscodeApi, rootUri, to), {
        overwrite: true,
      });
    },
    async list(dir) {
      const normalizedDir = normalizeStoragePath(dir);
      const entries = await fs.readDirectory(toUri(vscodeApi, rootUri, normalizedDir));
      return entries.map(([name, type]) => ({
        path: normalizedDir ? `${normalizedDir}/${name}` : name,
        type: type === vscodeApi.FileType.Directory ? "folder" : "file",
      }));
    },
    async readBinary(storagePath) {
      const bytes = await fs.readFile(toUri(vscodeApi, rootUri, storagePath));
      return toArrayBuffer(bytes);
    },
    async writeBinary(storagePath, content) {
      await fs.writeFile(toUri(vscodeApi, rootUri, storagePath), new Uint8Array(content));
    },
  };
}

function toUri(vscodeApi, rootUri, storagePath) {
  const parts = pathParts(storagePath);
  if (parts.length === 0) return rootUri;
  return vscodeApi.Uri.joinPath(rootUri, ...parts);
}

function normalizeStoragePath(input) {
  const parts = String(input)
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part && part !== ".");
  if (parts.some((part) => part === "..")) {
    throw new Error(`storage path escapes root: ${input}`);
  }
  return parts.join("/");
}

function pathParts(input) {
  const normalized = normalizeStoragePath(input);
  return normalized ? normalized.split("/") : [];
}

async function existsUri(fs, uri) {
  try {
    await fs.stat(uri);
    return true;
  } catch (error) {
    if (isFileNotFound(error)) return false;
    throw error;
  }
}

async function isDirectoryUri(vscodeApi, uri) {
  try {
    const stat = await vscodeApi.workspace.fs.stat(uri);
    return stat.type === vscodeApi.FileType.Directory;
  } catch (error) {
    if (isFileNotFound(error)) return false;
    throw error;
  }
}

function isFileNotFound(error) {
  return (
    error?.code === "FileNotFound" ||
    error?.name === "FileNotFound" ||
    /file not found|not found|enoent/i.test(String(error?.message ?? ""))
  );
}

function toArrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

module.exports = {
  ARXIV_DAILY_DIR,
  PAPER_INDEX_PATH,
  createVsCodeStorageAdapter,
  findArxivDailyVault,
  normalizeStoragePath,
};
