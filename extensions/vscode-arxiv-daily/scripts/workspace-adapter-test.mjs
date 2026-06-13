import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const {
  PAPER_INDEX_PATH,
  createVsCodeStorageAdapter,
  findArxivDailyVault,
  normalizeStoragePath,
} = require("../src/workspace.js");

const FileType = {
  File: 1,
  Directory: 2,
};

const vscodeApi = createMockVscodeApi();
vscodeApi.fs.addDirectory("/workspace/fileonly");
vscodeApi.fs.addFile("/workspace/fileonly/arxiv-daily", "not a folder");
vscodeApi.fs.addDirectory("/workspace/plain");
vscodeApi.fs.addDirectory("/workspace/vault");
vscodeApi.fs.addDirectory("/workspace/vault/arxiv-daily");
vscodeApi.workspace.workspaceFolders = [
  { name: "fileonly", uri: uri("/workspace/fileonly") },
  { name: "plain", uri: uri("/workspace/plain") },
  { name: "vault", uri: uri("/workspace/vault") },
];

const vault = await findArxivDailyVault(vscodeApi);
assert.equal(vault.workspaceFolder.name, "vault");
assert.equal(vault.paperIndexUri.path, `/workspace/vault/${PAPER_INDEX_PATH}`);

const storage = createVsCodeStorageAdapter(vscodeApi, vault.vaultRootUri);
assert.equal(storage.normalizePath("\\arxiv-daily//daily/./today.md"), "arxiv-daily/daily/today.md");
assert.throws(() => normalizeStoragePath("../outside"), /escapes root/);

await storage.mkdir("arxiv-daily/.index");
await storage.writeText(PAPER_INDEX_PATH, '{"schemaVersion":2}');
assert.equal(await storage.exists(PAPER_INDEX_PATH), true);
assert.equal(await storage.readText(PAPER_INDEX_PATH), '{"schemaVersion":2}');

await storage.writeBinary("arxiv-daily/pdfs/example.pdf", new Uint8Array([1, 2]).buffer);
assert.deepEqual(
  Array.from(new Uint8Array(await storage.readBinary("arxiv-daily/pdfs/example.pdf"))),
  [1, 2],
);

assert.deepEqual(await storage.list("arxiv-daily"), [
  { path: "arxiv-daily/.index", type: "folder" },
  { path: "arxiv-daily/pdfs", type: "folder" },
]);

await storage.rename(PAPER_INDEX_PATH, "arxiv-daily/.index/papers-old.json");
assert.equal(await storage.exists(PAPER_INDEX_PATH), false);
assert.equal(await storage.readText("arxiv-daily/.index/papers-old.json"), '{"schemaVersion":2}');

console.log("arXiv Daily VS Code workspace adapter OK");

function createMockVscodeApi() {
  const fs = createMemoryFs();
  return {
    FileType,
    Uri: {
      joinPath(base, ...parts) {
        return uri([base.path, ...parts].join("/"));
      },
    },
    fs,
    workspace: {
      fs,
      workspaceFolders: [],
    },
  };
}

function createMemoryFs() {
  const files = new Map();
  const directories = new Set(["/"]);

  return {
    files,
    directories,
    addDirectory(path) {
      ensureParentDirectories(directories, path);
      directories.add(normalizeUriPath(path));
    },
    addFile(path, content = "") {
      const normalized = normalizeUriPath(path);
      ensureParentDirectories(directories, parentDir(normalized));
      files.set(normalized, new TextEncoder().encode(content));
    },
    async stat(target) {
      const path = normalizeUriPath(target.path);
      if (files.has(path)) return { type: FileType.File };
      if (directories.has(path)) return { type: FileType.Directory };
      throw fileNotFound(path);
    },
    async readFile(target) {
      const path = normalizeUriPath(target.path);
      if (!files.has(path)) throw fileNotFound(path);
      return files.get(path);
    },
    async writeFile(target, content) {
      const path = normalizeUriPath(target.path);
      ensureParentDirectories(directories, parentDir(path));
      files.set(path, new Uint8Array(content));
    },
    async createDirectory(target) {
      const path = normalizeUriPath(target.path);
      ensureParentDirectories(directories, path);
      directories.add(path);
    },
    async delete(target) {
      const path = normalizeUriPath(target.path);
      files.delete(path);
      for (const filePath of [...files.keys()]) {
        if (filePath.startsWith(`${path}/`)) files.delete(filePath);
      }
      for (const dirPath of [...directories]) {
        if (dirPath.startsWith(`${path}/`)) directories.delete(dirPath);
      }
      directories.delete(path);
    },
    async rename(from, to) {
      const fromPath = normalizeUriPath(from.path);
      const toPath = normalizeUriPath(to.path);
      if (!files.has(fromPath)) throw fileNotFound(fromPath);
      ensureParentDirectories(directories, parentDir(toPath));
      files.set(toPath, files.get(fromPath));
      files.delete(fromPath);
    },
    async readDirectory(target) {
      const path = normalizeUriPath(target.path);
      if (!directories.has(path)) throw fileNotFound(path);
      const children = new Map();
      for (const dirPath of directories) {
        const child = directChildName(path, dirPath);
        if (child) children.set(child, FileType.Directory);
      }
      for (const filePath of files.keys()) {
        const child = directChildName(path, filePath);
        if (child) children.set(child, FileType.File);
      }
      return [...children.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    },
  };
}

function directChildName(parent, child) {
  if (parent === child) return "";
  const prefix = parent === "/" ? "/" : `${parent}/`;
  if (!child.startsWith(prefix)) return "";
  const rest = child.slice(prefix.length);
  if (!rest || rest.includes("/")) return "";
  return rest;
}

function ensureParentDirectories(directories, path) {
  const parts = normalizeUriPath(path).split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur += `/${part}`;
    directories.add(cur);
  }
}

function parentDir(path) {
  const normalized = normalizeUriPath(path);
  const idx = normalized.lastIndexOf("/");
  return idx <= 0 ? "/" : normalized.slice(0, idx);
}

function uri(path) {
  return { path: normalizeUriPath(path) };
}

function normalizeUriPath(path) {
  return `/${String(path).split("/").filter(Boolean).join("/")}`;
}

function fileNotFound(path) {
  return Object.assign(new Error(`File not found: ${path}`), {
    code: "FileNotFound",
  });
}
