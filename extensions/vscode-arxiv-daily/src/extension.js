const vscode = require("vscode");
const { openDashboard } = require("./dashboard");
const { promptAndStoreApiKey } = require("./secrets");

function activate(context) {
  const commands = [
    vscode.commands.registerCommand("arxivDaily.openDashboard", async () => {
      await openDashboard(vscode, context);
    }),
    vscode.commands.registerCommand("arxivDaily.run", () => {
      void vscode.window.showInformationMessage(
        "arXiv Daily: pipeline command wiring is planned next.",
      );
    }),
    vscode.commands.registerCommand("arxivDaily.runPending", () => {
      void vscode.window.showInformationMessage(
        "arXiv Daily: pending-run command wiring is planned next.",
      );
    }),
    vscode.commands.registerCommand("arxivDaily.summarizeById", () => {
      void vscode.window.showInformationMessage(
        "arXiv Daily: summarize-by-ID command wiring is planned next.",
      );
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
