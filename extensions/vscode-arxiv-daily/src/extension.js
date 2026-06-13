const vscode = require("vscode");
const { openDashboard } = require("./dashboard");
const { runForToday, runPending, summarizeById } = require("./pipeline-commands");
const { promptAndStoreApiKey } = require("./secrets");

function activate(context) {
  const commands = [
    vscode.commands.registerCommand("arxivDaily.openDashboard", async () => {
      await openDashboard(vscode, context);
    }),
    vscode.commands.registerCommand("arxivDaily.run", async () => {
      await runForToday(vscode, context);
    }),
    vscode.commands.registerCommand("arxivDaily.runPending", async () => {
      await runPending(vscode, context);
    }),
    vscode.commands.registerCommand("arxivDaily.summarizeById", async () => {
      await summarizeById(vscode, context);
    }),
    vscode.commands.registerCommand("arxivDaily.configureApiKey", async () => {
      await promptAndStoreApiKey(vscode, context);
    }),
  ];

  context.subscriptions.push(...commands);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
