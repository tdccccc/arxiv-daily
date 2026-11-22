import { describe, expect, it, vi } from "vitest";
import { PdfService } from "../src/services/pdf";
import { PaperIndexStore } from "../src/services/paper-index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { Logger } from "../src/services/logger";
import type { StorageAdapter } from "../src/core/adapters";

function makeStorage(withBinary = true) {
  const files: Record<string, string> = {};
  const binaries: Record<string, ArrayBuffer> = {};
  const dirs = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    async readText(path: string) {
      return files[path];
    },
    async writeText(path: string, content: string) {
      files[path] = content;
    },
    async exists(path: string) {
      return path in files || path in binaries || dirs.has(path);
    },
    async mkdir(path: string) {
      dirs.add(path);
    },
    async rename(from: string, to: string) {
      files[to] = files[from];
      delete files[from];
    },
    async remove(path: string) {
      delete files[path];
      delete binaries[path];
      dirs.delete(path);
    },
  };
  if (withBinary) {
    storage.writeBinary = async (path, content) => {
      binaries[path] = content;
    };
  }
  return { files, binaries, dirs, storage };
}

describe("PdfService", () => {
  const pdfBytes = new TextEncoder().encode("%PDF-1.7\n%%EOF\n").buffer;

  it("downloads PDFs to vault storage and records pdfPath", async () => {
    const { files, binaries, storage } = makeStorage();
    const store = new PaperIndexStore(
      storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-13T00:00:00.000Z"),
    );
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A",
      date: "2026-06-13",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });
    const service = new PdfService({
      fetcher: {
        fetchPdf: vi.fn().mockResolvedValue(pdfBytes),
      },
      storage,
      paperIndex: store,
      output: DEFAULT_SETTINGS.output,
      logger: new Logger("error"),
    });

    const result = await service.downloadById("https://arxiv.org/abs/2606.12345v2");

    expect(result).toMatchObject({
      kind: "done",
      arxivId: "2606.12345",
      path: "arxiv-daily/pdfs/2606.12345.pdf",
      bytes: pdfBytes.byteLength,
      entryUpdated: true,
    });
    expect(Array.from(new Uint8Array(binaries["arxiv-daily/pdfs/2606.12345.pdf"]))).toEqual(
      Array.from(new Uint8Array(pdfBytes)),
    );
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.papers["arxiv:2606.12345"].pdfPath).toBe(
      "arxiv-daily/pdfs/2606.12345.pdf",
    );
  });

  it("rejects an HTML error page without writing a .pdf or updating pdfPath", async () => {
    const { files, binaries, storage } = makeStorage();
    const setPdfPath = vi.fn();
    const service = new PdfService({
      fetcher: {
        fetchPdf: vi.fn().mockResolvedValue(
          new TextEncoder().encode("<!DOCTYPE html><html>captcha</html>").buffer,
        ),
      },
      storage,
      paperIndex: { setPdfPath } as any,
      output: DEFAULT_SETTINGS.output,
      logger: new Logger("error"),
    });

    await expect(service.downloadById("2606.12345")).resolves.toMatchObject({
      kind: "fetch_error",
    });
    expect(binaries).toEqual({});
    expect(setPdfPath).not.toHaveBeenCalled();
    expect(files).toEqual({});
  });

  it("reports missing binary storage support", async () => {
    const { storage } = makeStorage(false);
    const service = new PdfService({
      fetcher: {
        fetchPdf: vi.fn(),
      },
      storage,
      output: DEFAULT_SETTINGS.output,
      logger: new Logger("error"),
    });

    await expect(service.downloadById("2606.12345")).resolves.toMatchObject({
      kind: "missing_storage",
    });
  });
});
