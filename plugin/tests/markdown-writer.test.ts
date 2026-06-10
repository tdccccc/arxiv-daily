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

describe("MarkdownWriter strictness on existing files", () => {
  it("writeDaily throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    await expect(writer.writeDaily("2026-05-11", "new")).rejects.toThrow(
      /already exists/,
    );
  });

  it("writePaperDetail throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/papers/2605.06587.md": "x",
    });
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
    };
    await expect(writer.writePaperDetail(paper as any, "2026-05-11", "x"))
      .rejects.toThrow(/already exists/);
  });

  it("writeEmptyDaily throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    await expect(writer.writeEmptyDaily("2026-05-11")).rejects.toThrow(
      /already exists/,
    );
  });

  it("writeDaily writes content (no bak file produced)", async () => {
    const { files, writer } = makeWriter();
    await writer.writeDaily("2026-05-11", "body");
    const written = files["arxiv-daily/daily/2026-05-11.md"];
    expect(written).toContain("date: 2026-05-11");
    expect(written).toContain("weekday: Monday");
    expect(written).toContain("body");
    expect(files["arxiv-daily/daily/2026-05-11.bak.md"]).toBeUndefined();
  });

  it("writePaperDetail writes date and weekday properties", async () => {
    const { files, writer } = makeWriter();
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
    };
    await writer.writePaperDetail(paper as any, "2026-06-10", "detail");
    const written = files["arxiv-daily/papers/2605.06587.md"];
    expect(written).toContain("date: 2026-06-10");
    expect(written).toContain("weekday: Wednesday");
    expect(written).toContain("detail");
  });
});
