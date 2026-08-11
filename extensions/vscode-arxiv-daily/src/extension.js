const vscode = require("vscode");
const { openDashboard } = require("./dashboard");
const { runForToday, summarizeById } = require("./pipeline-commands");

function activate(context) {
  const commands = [
    vscode.commands.registerCommand("arxivDaily.openDashboard", async () => {
      await openDashboard(vscode, context);
    }),
    vscode.commands.registerCommand("arxivDaily.run", async () => {
      await runForToday(vscode);
    }),
    vscode.commands.registerCommand("arxivDaily.summarizeById", async () => {
      await summarizeById(vscode);
    }),
  ];

  context.subscriptions.push(...commands);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
