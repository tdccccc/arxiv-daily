const vscode = require("vscode");

function activate(context) {
  const commands = [
    vscode.commands.registerCommand("arxivDaily.openDashboard", () => {
      void vscode.window.showInformationMessage(
        "arXiv Daily: VS Code Dashboard wiring is planned next.",
      );
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
    vscode.commands.registerCommand("arxivDaily.configureApiKey", () => {
      void vscode.window.showInformationMessage(
        "arXiv Daily: SecretStorage wiring is planned next.",
      );
    }),
  ];

  context.subscriptions.push(...commands);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
