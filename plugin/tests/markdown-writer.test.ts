import { describe, it, expect } from "vitest";
import { MarkdownWriter } from "../src/pipeline/markdown-writer";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

function makeVault(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  return {
    files,
    vault: {
      adapter: {
        async write(path: string, content: string) {
          files[path] = content;
        },
        async exists(path: string) {
          return Object.prototype.hasOwnProperty.call(files, path);
        },
        async mkdir(_path: string) {},
        async rename(from: string, to: string) {
          files[to] = files[from];
          delete files[from];
        },
        async remove(path: string) {
          delete files[path];
        },
      },
    } as any,
  };
}

function makeWriter(initialFiles: Record<string, string> = {}) {
  const { files, vault } = makeVault(initialFiles);
  const writer = new MarkdownWriter({
    vault,
    logger: new Logger("error"),
    arxiv: DEFAULT_SETTINGS.arxiv,
    output: DEFAULT_SETTINGS.output,
  });
  return { files, writer };
}

describe("MarkdownWriter existence checks", () => {
  it("dailyExists returns false when daily missing", async () => {
    const { writer } = makeWriter();
    expect(await writer.dailyExists("2026-05-11")).toBe(false);
  });

  it("dailyExists returns true when daily present", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    expect(await writer.dailyExists("2026-05-11")).toBe(true);
  });

  it("paperDetailExists returns false when paper missing", async () => {
    const { writer } = makeWriter();
    expect(await writer.paperDetailExists("2605.06587")).toBe(false);
  });

  it("paperDetailExists returns true when paper present", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/papers/2605.06587.md": "x",
    });
    expect(await writer.paperDetailExists("2605.06587")).toBe(true);
  });
});
