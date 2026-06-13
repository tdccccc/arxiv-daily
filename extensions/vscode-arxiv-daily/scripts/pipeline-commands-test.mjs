import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const {
  OBSIDIAN_PLUGIN_DATA_PATH,
  VSCODE_CLI_CONFIG_PATH,
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
  "'/tools/arxiv daily' run --date 2026-06-13 --config /workspace/vault/arxiv-daily/.index/vscode-cli.config.json --vault-root /workspace/vault",
);
const cliConfigText = new TextDecoder().decode(
  await vscodeApi.workspace.fs.readFile(uri(`/workspace/vault/${VSCODE_CLI_CONFIG_PATH}`)),
);
const cliConfig = JSON.parse(cliConfigText);
assert.equal(cliConfig.linkStyle, "relative");
assert.equal(cliConfig.settings.output.linkStyle, "relative");
assert.equal(cliConfig.settings.arxiv.topics[0].tag, "photo-z");
assert.equal(cliConfig.settings.llm.apiKey, "");
assert(!cliConfigText.includes("sk-test"));
assert(!cliConfigText.includes("obsidian-key"));

const summarizeApi = createMockVscodeApi({
  apiKey: "sk-test",
  inputValue: "https://arxiv.org/abs/2606.54321v1",
});
const summarized = await summarizeById(summarizeApi, summarizeApi.context);
assert.equal(summarized, true);
assert.equal(
  summarizeApi.terminals[0].sent[0],
  "arxiv-daily summarize --id 2606.54321 --config /workspace/vault/arxiv-daily/.index/vscode-cli.config.json --vault-root /workspace/vault",
);

const missingKeyApi = createMockVscodeApi({ apiKey: "" });
const missingKey = await runForToday(missingKeyApi, missingKeyApi.context);
assert.equal(missingKey, false);
assert.equal(missingKeyApi.terminals.length, 0);

console.log("arXiv Daily VS Code pipeline commands OK");

function createMockVscodeApi({
  apiKey,
  cliPath = "arxiv-daily",
  inputValue = "",
  pluginData = samplePluginData(),
}) {
  const terminals = [];
  const fs = createMemoryFs(pluginData);
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

function createMemoryFs(pluginData) {
  const files = new Map();
  const directories = new Set(["/"]);
  addDirectory("/workspace/vault");
  addDirectory("/workspace/vault/arxiv-daily");
  if (pluginData) {
    addFile(
      `/workspace/vault/${OBSIDIAN_PLUGIN_DATA_PATH}`,
      JSON.stringify(pluginData, null, 2),
    );
  }

  return {
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
      ensureParentDirectories(path);
      files.set(path, new Uint8Array(content));
    },
    async createDirectory(target) {
      addDirectory(target.path);
    },
  };

  function addDirectory(path) {
    ensureParentDirectories(path);
    directories.add(normalizeUriPath(path));
  }

  function addFile(path, content) {
    const normalized = normalizeUriPath(path);
    ensureParentDirectories(parentDir(normalized));
    files.set(normalized, new TextEncoder().encode(content));
  }

  function ensureParentDirectories(path) {
    const parts = normalizeUriPath(path).split("/").filter(Boolean);
    let cur = "";
    for (const part of parts) {
      cur += `/${part}`;
      directories.add(cur);
    }
  }
}

function samplePluginData() {
  return {
    settings: {
      llm: {
        apiKey: "obsidian-key",
        provider: "deepseek",
        baseUrl: "https://api.deepseek.com/v1",
        model: "deepseek-v4-pro",
        temperature: 0.3,
        timeoutMs: 300000,
        thinkingMode: true,
        reasoningEffort: "high",
      },
      arxiv: {
        category: "astro-ph",
        categories: ["astro-ph"],
        topics: [
          {
            id: "topic-1",
            name: "Photo-z",
            tag: "photo-z",
            description: "photometric redshift calibration",
            detail: true,
          },
        ],
        timezone: "Asia/Shanghai",
      },
      output: {
        dailyDir: "arxiv-daily/daily",
        papersDir: "arxiv-daily/papers",
        linkStyle: "wikilink",
      },
      schedule: {
        enabled: false,
        runAtLocal: "09:30",
        tickIntervalMin: 20,
        lookbackDays: 5,
      },
      advanced: {
        requestDelayMs: 3000,
        cacheExpiryDays: 7,
        sectionCharLimit: 8000,
        paperCharLimit: 50000,
        dailyCharLimit: 400000,
        skipSections: [],
        prioritySections: ["abstract", "conclusion"],
        logLevel: "info",
      },
    },
  };
}

function normalizeUriPath(path) {
  return `/${String(path).split("/").filter(Boolean).join("/")}`;
}

function parentDir(path) {
  const normalized = normalizeUriPath(path);
  const idx = normalized.lastIndexOf("/");
  return idx <= 0 ? "/" : normalized.slice(0, idx);
}

function fileNotFound(path) {
  return Object.assign(new Error(`File not found: ${path}`), {
    code: "FileNotFound",
  });
}
