const vscode = require("vscode");
const { promptAndStoreApiKey } = require("./secrets");
const { findArxivDailyVault } = require("./workspace");

function activate(context) {
  const commands = [
    vscode.commands.registerCommand("arxivDaily.openDashboard", async () => {
      const vault = await findArxivDailyVault(vscode);
      if (!vault) {
        void vscode.window.showWarningMessage(
          "arXiv Daily: no workspace folder contains arxiv-daily/.",
        );
        return;
      }
      void vscode.window.showInformationMessage(
        `arXiv Daily: found vault in ${vault.workspaceFolder.name}; Dashboard wiring is planned next.`,
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
