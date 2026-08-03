import { describe, expect, it, vi } from "vitest";
import {
  ObsidianLibraryDirectoryPicker,
  type DirectoryDialog,
} from "../src/hosts/obsidian/library-directory-picker";

describe("ObsidianLibraryDirectoryPicker", () => {
  it("selects exactly one directory through the injected desktop dialog", async () => {
    const showOpenDialog = vi.fn(async () => ({
      canceled: false,
      filePaths: ["/research/library"],
    }));
    const picker = new ObsidianLibraryDirectoryPicker({ showOpenDialog });

    await expect(picker.select()).resolves.toEqual({
      kind: "selected",
      path: "/research/library",
    });
    expect(showOpenDialog).toHaveBeenCalledWith({
      title: "Choose personal literature library",
      properties: ["openDirectory", "dontAddToRecent"],
    });
  });

  it("reports cancellation without retaining a path", async () => {
    const dialog: DirectoryDialog = {
      showOpenDialog: vi.fn(async () => ({ canceled: true, filePaths: [] })),
    };

    await expect(new ObsidianLibraryDirectoryPicker(dialog).select())
      .resolves.toEqual({ kind: "cancelled" });
  });

  it("reports an unsupported host when no desktop dialog is available", async () => {
    await expect(new ObsidianLibraryDirectoryPicker(null).select())
      .resolves.toEqual({ kind: "unsupported" });
  });

  it("rejects an unexpected multi-directory result", async () => {
    const dialog: DirectoryDialog = {
      showOpenDialog: vi.fn(async () => ({
        canceled: false,
        filePaths: ["/first", "/second"],
      })),
    };

    await expect(new ObsidianLibraryDirectoryPicker(dialog).select())
      .rejects.toThrow("Expected exactly one selected library directory");
  });
});
