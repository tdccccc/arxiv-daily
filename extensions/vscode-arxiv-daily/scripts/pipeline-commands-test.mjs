import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const {
  buildCliCommand,
  formatDate,
  normalizeArxivId,
  runForToday,
  summarizeById,
} = require("../src/pipeline-commands.js");

assert.equal(formatDate(new Date("2026-06-13T23:00:00")), "2026-06-13");
assert.equal(normalizeArxivId("https://arxiv.org/abs/2606.12345v2"), "2606.12345");
assert.equal(normalizeArxivId("https://arxiv.org/pdf/2606.1234"), "2606.1234");
assert.equal(normalizeArxivId("bad"), "");
assert.equal(
  buildCliCommand("/path with spaces/arxiv-daily", ["run", "--vault-root", "/tmp/my vault"]),
  "'/path with spaces/arxiv-daily' run --vault-root '/tmp/my vault'",
);

const vscodeApi = createMockVscodeApi({
  apiKey: "sk-test",
  cliPath: "/tools/arxiv daily",
});
const ran = await runForToday(
  vscodeApi,
  vscodeApi.context,
  () => new Date("2026-06-13T09:30:00"),
);
assert.equal(ran, true);
assert.equal(vscodeApi.terminals[0].options.cwd, "/workspace/vault");
assert.equal(vscodeApi.terminals[0].options.env.ARXIV_DAILY_API_KEY, "sk-test");
assert.equal(vscodeApi.terminals[0].options.env.ARXIV_DAILY_LINK_STYLE, "relative");
assert.equal(
  vscodeApi.terminals[0].sent[0],
  "'/tools/arxiv daily' run --date 2026-06-13 --vault-root /workspace/vault",
);

const summarizeApi = createMockVscodeApi({
  apiKey: "sk-test",
  inputValue: "https://arxiv.org/abs/2606.54321v1",
});
const summarized = await summarizeById(summarizeApi, summarizeApi.context);
assert.equal(summarized, true);
assert.equal(
  summarizeApi.terminals[0].sent[0],
  "arxiv-daily summarize --id 2606.54321 --vault-root /workspace/vault",
);

const missingKeyApi = createMockVscodeApi({ apiKey: "" });
const missingKey = await runForToday(missingKeyApi, missingKeyApi.context);
assert.equal(missingKey, false);
assert.equal(missingKeyApi.terminals.length, 0);

console.log("arXiv Daily VS Code pipeline commands OK");

function createMockVscodeApi({ apiKey, cliPath = "arxiv-daily", inputValue = "" }) {
  const terminals = [];
  const fs = {
    async stat(target) {
      if (target.path === "/workspace/vault/arxiv-daily") {
        return { type: 2 };
      }
      throw Object.assign(new Error("File not found"), { code: "FileNotFound" });
    },
  };
  const context = {
    secrets: {
      async get(key) {
        return key === "arxivDaily.llm.apiKey" ? apiKey || undefined : undefined;
      },
    },
    subscriptions: [],
  };
  return {
    FileType: {
      File: 1,
      Directory: 2,
    },
    Uri: {
      joinPath(base, ...parts) {
        return uri([base.path, ...parts].join("/"));
      },
    },
    context,
    terminals,
    workspace: {
      fs,
      workspaceFolders: [{ name: "vault", uri: uri("/workspace/vault") }],
      getConfiguration(section) {
        assert.equal(section, "arxivDaily");
        return {
          get(key, fallback) {
            assert.equal(key, "cliPath");
            return cliPath || fallback;
          },
        };
      },
    },
    window: {
      async showInputBox() {
        return inputValue;
      },
      showWarningMessage() {},
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
  };
}

function uri(path) {
  const normalized = `/${String(path).split("/").filter(Boolean).join("/")}`;
  return {
    path: normalized,
    fsPath: normalized,
  };
}
