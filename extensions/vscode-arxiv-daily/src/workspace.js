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
    };
  }
  return null;
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

module.exports = {
  ARXIV_DAILY_DIR,
  PAPER_INDEX_PATH,
  findArxivDailyVault,
  normalizeStoragePath,
};
