const vscode = require("vscode");
const dashboard = require("./dashboard");
const { openDashboard } = dashboard;
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
  // Not used by VS Code. The tests load the built bundle (the shipped
  // artifact) and drive these directly with a mock `vscode` API.
  internals: {
    ...dashboard,
    runForToday,
    summarizeById,
  },
};
