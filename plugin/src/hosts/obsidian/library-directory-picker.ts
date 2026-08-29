export interface DirectoryDialogResult {
  canceled: boolean;
  filePaths: string[];
}

export interface DirectoryDialog {
  showOpenDialog(options: {
    title: string;
    properties: string[];
  }): Promise<DirectoryDialogResult>;
}

export type LibraryDirectorySelection =
  | { kind: "selected"; path: string }
  | { kind: "cancelled" }
  | { kind: "unsupported" };

export class ObsidianLibraryDirectoryPicker {
  constructor(private readonly dialog: DirectoryDialog | null = hostDirectoryDialog()) {}

  async select(): Promise<LibraryDirectorySelection> {
    if (!this.dialog) return { kind: "unsupported" };
    const result = await this.dialog.showOpenDialog({
      title: "Choose personal literature library",
      properties: ["openDirectory", "dontAddToRecent"],
    });
    if (result.canceled || result.filePaths.length === 0) {
      return { kind: "cancelled" };
    }
    if (result.filePaths.length !== 1 || !result.filePaths[0]) {
      throw new Error("Expected exactly one selected library directory");
    }
    return { kind: "selected", path: result.filePaths[0] };
  }
}

function hostDirectoryDialog(): DirectoryDialog | null {
  const hostWindow = window as Window & {
    electron?: {
      remote?: {
        dialog?: DirectoryDialog;
      };
    };
  };
  return hostWindow.electron?.remote?.dialog ?? null;
}
