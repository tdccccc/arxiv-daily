import assert from "node:assert/strict";
import { readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { createTestVault } from "./test-vault.mjs";

const { loadExtensionBundle } = await import("./load-bundle.mjs");
const { openDashboard, runForToday } = loadExtensionBundle().internals;

const testVault = createTestVault({ index: seedIndex() });
writeFileSync(path.join(testVault.root, "arxiv-daily", "papers", "2606.12345.md"), "# Paper");
const vscodeApi = createMockVscodeApi(testVault.root);

const panel = await openDashboard(vscodeApi, vscodeApi.context, { storeOptions: testVault.storeOptions });
assert(panel, "Dashboard panel should open for a workspace containing arxiv-daily/");
assert(panel.webview.html.includes("arXiv Daily"));
assert(panel.webview.html.includes("Search"));
assert(panel.webview.html.includes("Photometric Redshift Calibration"));

const htmlAtOpen = panel.webview.html;
const readIndex = async () => testVault.readIndex();

// The page posts the map key of the row ("arxiv:<id>" in schema 5).
await panel.webview.receive({
  type: "setStatus",
  key: "arxiv:2606.12345",
  arxivId: "2606.12345",
  status: "reading",
});
let savedIndex = await readIndex();
assert.equal(savedIndex.schemaVersion, 5);
assert.equal(savedIndex.papers["arxiv:2606.12345"].status, "reading");
assert.equal(savedIndex.papers["arxiv:2606.12345"].updated, "2026-06-13");
assert(savedIndex.updatedAt !== "2026-06-13T00:00:00.000Z", "status update should refresh index updatedAt");
// The running page is updated by message, so its tab/search/filter state survives.
assert.equal(panel.webview.html, htmlAtOpen, "html must not be reassigned after the first render");
assert.equal(panel.webview.posted.length, 1);
assert.equal(panel.webview.posted[0].type, "state");
assert.equal(panel.webview.posted[0].state.allRows[0].entry.status, "reading");
// Core's atomic replace leaves only the index and its backup behind.
assert.deepEqual(
  readdirSync(path.dirname(testVault.indexFile)).sort(),
  ["papers.json", "papers.json.bak"],
);

// Older pages (or other callers) may only send the bare arXiv id.
await panel.webview.receive({ type: "setStatus", arxivId: "2606.12345", status: "saved" });
assert.equal((await readIndex()).papers["arxiv:2606.12345"].status, "saved");

await panel.webview.receive({
  type: "openResource",
  key: "arxiv:2606.12345",
  arxivId: "2606.12345",
  resource: "note",
});
assert.deepEqual(vscodeApi.commands.opened.map((item) => item.path), [
  `${testVault.root}/arxiv-daily/papers/2606.12345.md`,
]);
await panel.webview.receive({
  type: "openResource",
  key: "arxiv:2606.12345",
  resource: "arxiv",
});
assert.deepEqual(vscodeApi.env.openedExternal.map((item) => item.path), [
  "https://arxiv.org/abs/2606.12345",
]);

const runPromise = runForToday(vscodeApi);
assert.equal(vscodeApi.executedTasks.length, 1);
assert.equal(vscodeApi.taskExecutions.length, 1);
vscodeApi.endTaskProcess(vscodeApi.taskExecutions[0], 0);
vscodeApi.endTask(vscodeApi.taskExecutions[0]);
const ran = await runPromise;
assert.equal(ran, true);
assert.equal(vscodeApi.terminals.length, 0);
assert.equal(vscodeApi.activeTaskProcessListeners, 0);
assert.equal(vscodeApi.activeTaskListeners, 0);
assert.equal(vscodeApi.executedTasks[0].scope, vscodeApi.TaskScope.Workspace);
assert.equal(vscodeApi.executedTasks[0].execution.process, "arxiv-daily");
assert.deepEqual(vscodeApi.executedTasks[0].execution.args, ["run", "--today"]);

testVault.cleanup();

console.log("arXiv Daily VS Code extension smoke OK");

// Shape written by core's PaperIndexStore (schema 5): keyed "arxiv:<id>".
function seedIndex() {
  return {
    schemaVersion: 5,
    updatedAt: "2026-06-13T00:00:00.000Z",
    papers: {
      "arxiv:2606.12345": {
        paperKey: "arxiv:2606.12345",
        source: "arxiv",
        externalId: "2606.12345",
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
        discoveryProvenanceByReport: {},
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
  };
}

function createMockVscodeApi(vaultRoot) {
  const fs = createRealFs();
  const terminals = [];
  const executedTasks = [];
  const taskExecutions = [];
  const taskProcessListeners = new Set();
  const taskListeners = new Set();
  const opened = [];

  class ProcessExecution {
    constructor(process, args) {
      this.process = process;
      this.args = [...args];
    }
  }

  class Task {
    constructor(definition, scope, name, source, execution) {
      this.definition = definition;
      this.scope = scope;
      this.name = name;
      this.source = source;
      this.execution = execution;
    }
  }

  class TaskExecution {
    constructor(task) {
      this.task = task;
    }
  }

  const vscodeApi = {
    ProcessExecution,
    Task,
    TaskScope: {
      Workspace: 2,
    },
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
      subscriptions: [],
    },
    workspace: {
      fs,
      workspaceFolders: [{ name: "vault", uri: uri(vaultRoot) }],
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
    tasks: {
      onDidEndTaskProcess(listener) {
        taskProcessListeners.add(listener);
        return {
          dispose() {
            taskProcessListeners.delete(listener);
          },
        };
      },
      onDidEndTask(listener) {
        taskListeners.add(listener);
        return {
          dispose() {
            taskListeners.delete(listener);
          },
        };
      },
      async executeTask(task) {
        executedTasks.push(task);
        const taskExecution = new TaskExecution(task);
        taskExecutions.push(taskExecution);
        return taskExecution;
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
    executedTasks,
    taskExecutions,
    endTaskProcess(execution, exitCode) {
      for (const listener of [...taskProcessListeners]) {
        listener({ execution, exitCode });
      }
    },
    endTask(execution) {
      for (const listener of [...taskListeners]) {
        listener({ execution });
      }
    },
  };

  Object.defineProperties(vscodeApi, {
    activeTaskProcessListeners: {
      get: () => taskProcessListeners.size,
    },
    activeTaskListeners: {
      get: () => taskListeners.size,
    },
  });
  return vscodeApi;
}

function createMockPanel() {
  let handler = null;
  return {
    webview: {
      html: "",
      posted: [],
      async postMessage(message) {
        this.posted.push(message);
        return true;
      },
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

// The dashboard only stats the vault folder through VS Code's file system; the
// paper index itself is read and written by core on the real disk.
function createRealFs() {
  return {
    async stat(target) {
      try {
        return { type: statSync(target.fsPath).isDirectory() ? 2 : 1 };
      } catch (error) {
        throw error.code === "ENOENT" ? fileNotFound(target.fsPath) : error;
      }
    },
  };
}

function uri(target) {
  const normalized = String(target).replace(/\\/g, "/");
  return { path: normalized, fsPath: target, scheme: "file" };
}

function fileNotFound(path) {
  return Object.assign(new Error(`File not found: ${path}`), {
    code: "FileNotFound",
  });
}
