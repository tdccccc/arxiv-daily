const { DEFAULT_API_KEY_SECRET, createSecretProvider } = require("./secrets");
const { findArxivDailyVault } = require("./workspace");

const DEFAULT_CLI_PATH = "arxiv-daily";
const OBSIDIAN_PLUGIN_DATA_PATH = ".obsidian/plugins/arxiv-daily/data.json";
const VSCODE_CLI_CONFIG_PATH = "arxiv-daily/.index/vscode-cli.config.json";

async function runForToday(vscodeApi, context, now = () => new Date()) {
  const date = formatDate(now());
  return await runCliCommand(vscodeApi, context, ["run", "--date", date]);
}

async function runPending(vscodeApi, context) {
  return await runCliCommand(vscodeApi, context, ["run-pending"]);
}

async function summarizeById(vscodeApi, context) {
  const input = await vscodeApi.window.showInputBox({
    title: "Summarize arXiv ID",
    prompt: "Enter an arXiv ID or URL.",
    placeHolder: "2606.12345",
    ignoreFocusOut: true,
  });
  if (input === undefined) return false;
  const arxivId = normalizeArxivId(input);
  if (!arxivId) {
    void vscodeApi.window.showWarningMessage("arXiv Daily: invalid arXiv ID.");
    return false;
  }
  return await runCliCommand(vscodeApi, context, ["summarize", "--id", arxivId]);
}

async function runCliCommand(vscodeApi, context, args) {
  const vault = await findArxivDailyVault(vscodeApi);
  if (!vault) {
    void vscodeApi.window.showWarningMessage(
      "arXiv Daily: no workspace folder contains arxiv-daily/.",
    );
    return false;
  }

  const apiKey = await createSecretProvider(context).getSecret(DEFAULT_API_KEY_SECRET);
  if (!apiKey) {
    void vscodeApi.window.showWarningMessage(
      "arXiv Daily: configure an API key before running the pipeline.",
    );
    return false;
  }

  const cliPath = cliPathFromSettings(vscodeApi);
  const vaultRoot = uriToFsPath(vault.vaultRootUri);
  let configPath = null;
  try {
    configPath = await prepareCliConfig(vscodeApi, vault);
  } catch (error) {
    void vscodeApi.window.showErrorMessage(
      `arXiv Daily: failed to prepare CLI config: ${error.message}`,
    );
    return false;
  }
  const terminal = vscodeApi.window.createTerminal({
    name: "arXiv Daily",
    cwd: vaultRoot,
    env: {
      ARXIV_DAILY_API_KEY: apiKey,
      ARXIV_DAILY_LINK_STYLE: "relative",
    },
  });
  terminal.show();
  const cliArgs = [...args];
  if (configPath) cliArgs.push("--config", configPath);
  cliArgs.push("--vault-root", vaultRoot);
  terminal.sendText(buildCliCommand(cliPath, cliArgs));
  return true;
}

async function prepareCliConfig(vscodeApi, vault) {
  const storage = vault.storage;
  if (!(await storage.exists(OBSIDIAN_PLUGIN_DATA_PATH))) return null;

  const raw = JSON.parse(await storage.readText(OBSIDIAN_PLUGIN_DATA_PATH));
  const settings = sanitizeSettingsForCli(extractPluginSettings(raw));
  await storage.mkdir("arxiv-daily/.index");
  await storage.writeText(
    VSCODE_CLI_CONFIG_PATH,
    JSON.stringify({ settings, linkStyle: "relative" }, null, 2),
  );
  return uriToFsPath(
    vscodeApi.Uri.joinPath(
      vault.vaultRootUri,
      ...VSCODE_CLI_CONFIG_PATH.split("/"),
    ),
  );
}

function extractPluginSettings(data) {
  if (!isRecord(data)) {
    throw new Error("Obsidian plugin data must be a JSON object");
  }
  const settings = isRecord(data.settings) ? data.settings : data;
  if (!isRecord(settings)) {
    throw new Error("Obsidian plugin settings must be a JSON object");
  }
  return settings;
}

function sanitizeSettingsForCli(settings) {
  const next = JSON.parse(JSON.stringify(settings));
  if (!isRecord(next.llm)) next.llm = {};
  next.llm.apiKey = "";
  if (!isRecord(next.output)) next.output = {};
  next.output.linkStyle = "relative";
  return next;
}

function cliPathFromSettings(vscodeApi) {
  const configured = vscodeApi.workspace
    .getConfiguration("arxivDaily")
    .get("cliPath", DEFAULT_CLI_PATH);
  return String(configured || DEFAULT_CLI_PATH).trim() || DEFAULT_CLI_PATH;
}

function buildCliCommand(cliPath, args) {
  return [cliPath, ...args].map(shellQuote).join(" ");
}

function shellQuote(value) {
  const text = String(value);
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(text)) return text;
  return `'${text.replace(/'/g, "'\\''")}'`;
}

function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function normalizeArxivId(input) {
  const trimmed = String(input).trim();
  const match = trimmed.match(/(?:arxiv\.org\/(?:abs|pdf)\/)?([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?/i);
  return match?.[1] ?? "";
}

function uriToFsPath(uri) {
  return uri.fsPath || uri.path;
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

module.exports = {
  OBSIDIAN_PLUGIN_DATA_PATH,
  VSCODE_CLI_CONFIG_PATH,
  buildCliCommand,
  formatDate,
  normalizeArxivId,
  prepareCliConfig,
  runForToday,
  runPending,
  summarizeById,
};
