import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const { openDashboard } = require("../src/dashboard.js");
const { runForToday } = require("../src/pipeline-commands.js");

const vscodeApi = createMockVscodeApi();
seedVault(vscodeApi.fs);

const panel = await openDashboard(vscodeApi, vscodeApi.context);
assert(panel, "Dashboard panel should open for a workspace containing arxiv-daily/");
assert(panel.webview.html.includes("arXiv Daily"));
assert(panel.webview.html.includes("Search"));
assert(panel.webview.html.includes("Photometric Redshift Calibration"));

await panel.webview.receive({
  type: "setStatus",
  arxivId: "2606.12345",
  status: "reading",
});
const savedIndex = JSON.parse(
  new TextDecoder().decode(
    await vscodeApi.fs.readFile(uri("/workspace/vault/arxiv-daily/.index/papers.json")),
  ),
);
assert.equal(savedIndex.papers["2606.12345"].status, "reading");
assert(savedIndex.updatedAt, "status update should refresh index updatedAt");

await panel.webview.receive({
  type: "openResource",
  arxivId: "2606.12345",
  resource: "note",
});
assert.deepEqual(vscodeApi.commands.opened.map((item) => item.path), [
  "/workspace/vault/arxiv-daily/papers/2606.12345.md",
]);

const ran = await runForToday(
  vscodeApi,
  vscodeApi.context,
  () => new Date("2026-06-13T09:30:00"),
);
assert.equal(ran, true);
assert.equal(vscodeApi.terminals.length, 1);
assert.equal(vscodeApi.terminals[0].options.cwd, "/workspace/vault");
assert.equal(vscodeApi.terminals[0].options.env.ARXIV_DAILY_API_KEY, "sk-smoke");
assert.equal(vscodeApi.terminals[0].options.env.ARXIV_DAILY_LINK_STYLE, "relative");
assert.equal(
  vscodeApi.terminals[0].sent[0],
  "arxiv-daily run --date 2026-06-13 --vault-root /workspace/vault",
);

console.log("arXiv Daily VS Code extension smoke OK");

function seedVault(fs) {
  fs.addDirectory("/workspace/vault/arxiv-daily");
  fs.addDirectory("/workspace/vault/arxiv-daily/.index");
  fs.addDirectory("/workspace/vault/arxiv-daily/daily");
  fs.addDirectory("/workspace/vault/arxiv-daily/papers");
  fs.addFile(
    "/workspace/vault/arxiv-daily/.index/papers.json",
    JSON.stringify(
      {
        schemaVersion: 2,
        updatedAt: "2026-06-13T00:00:00.000Z",
        papers: {
          "2606.12345": {
            arxivId: "2606.12345",
            title: "Photometric Redshift Calibration",
            authors: ["Jane Doe"],
            published: "2026-06-13",
            updated: "2026-06-13",
            category: "astro-ph.CO",
            categories: ["astro-ph.CO"],
            summary: {
              coreProblem: "calibrating photo-z catalogs",
            },
            topics: ["photo-z"],
            primaryTopic: "photo-z",
            detail: false,
            status: "to_read",
            priority: "normal",
            seenDates: ["2026-06-13"],
            dailyReports: ["arxiv-daily/daily/2026-06-13.md"],
            paperPath: "arxiv-daily/papers/2606.12345.md",
            arxivUrl: "https://arxiv.org/abs/2606.12345",
            pdfUrl: "https://arxiv.org/pdf/2606.12345",
            pdfPath: "",
            zoteroKey: "",
            zoteroUri: "",
            citationKey: "",
            projects: [],
          },
        },
      },
      null,
      2,
    ),
  );
  fs.addFile("/workspace/vault/arxiv-daily/papers/2606.12345.md", "# Paper");
}

function createMockVscodeApi() {
  const fs = createMemoryFs();
  const terminals = [];
  const opened = [];
  return {
    FileType: {
      File: 1,
      Directory: 2,
    },
    ViewColumn: {
      One: 1,
    },
    Uri: {
      parse(value) {
        return { path: value, fsPath: value, toString: () => value };
      },
      joinPath(base, ...parts) {
        return uri([base.path, ...parts].join("/"));
      },
    },
    context: {
      secrets: {
        async get(key) {
          return key === "arxivDaily.llm.apiKey" ? "sk-smoke" : undefined;
        },
      },
      subscriptions: [],
    },
    workspace: {
      fs,
      workspaceFolders: [{ name: "vault", uri: uri("/workspace/vault") }],
      getConfiguration() {
        return {
          get(_key, fallback) {
            return fallback;
          },
        };
      },
    },
    window: {
      createWebviewPanel() {
        return createMockPanel();
      },
      showWarningMessage() {},
      showErrorMessage(message) {
        throw new Error(message);
      },
      createTerminal(options) {
        const terminal = {
          options,
          sent: [],
          show() {},
          sendText(text) {
            this.sent.push(text);
          },
        };
        terminals.push(terminal);
        return terminal;
      },
    },
    commands: {
      opened,
      async executeCommand(command, target) {
        assert.equal(command, "vscode.open");
        opened.push(target);
      },
    },
    env: {
      openedExternal: [],
      async openExternal(target) {
        this.openedExternal.push(target);
      },
    },
    fs,
    terminals,
  };
}

function createMockPanel() {
  let handler = null;
  return {
    webview: {
      html: "",
      onDidReceiveMessage(callback) {
        handler = callback;
      },
      async receive(message) {
        assert(handler, "webview message handler should be registered");
        await handler(message);
      },
    },
  };
}

function createMemoryFs() {
  const files = new Map();
  const directories = new Set(["/"]);
  return {
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
      if (files.has(path)) return { type: 1 };
      if (directories.has(path)) return { type: 2 };
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
      files.delete(normalizeUriPath(target.path));
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
        if (child) children.set(child, 2);
      }
      for (const filePath of files.keys()) {
        const child = directChildName(path, filePath);
        if (child) children.set(child, 1);
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
  const normalized = normalizeUriPath(path);
  return {
    path: normalized,
    fsPath: normalized,
  };
}

function normalizeUriPath(path) {
  return `/${String(path).split("/").filter(Boolean).join("/")}`;
}

function fileNotFound(path) {
  return Object.assign(new Error(`File not found: ${path}`), {
    code: "FileNotFound",
  });
}
