import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  authorizeLibraryConnection,
  createLibraryConnection,
  type EmbeddingSettings,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import {
  describeDiagnosticsError,
  formatFullTextRuntimeDiagnostics,
  summarizeFullTextRuntimeDiagnostics,
  type FullTextRuntimeDiagnostics,
} from "../src/services/fulltext-runtime-diagnostics";

const transformersRuntime = vi.hoisted(() => ({ pipeline: vi.fn() }));
vi.mock("@huggingface/transformers", () => transformersRuntime);

beforeEach(() => {
  transformersRuntime.pipeline.mockReset();
});

function passReport(): FullTextRuntimeDiagnostics {
  return {
    library: { connected: true, scopeFingerprint: "scope-hex", paperCount: 5 },
    pdfJs: {
      status: "pass",
      loadPdfJsResolved: true,
      loaderReturnedLib: true,
      windowPdfJsLibPresent: true,
      windowPdfJsLibVersion: "4.2.189",
      smoke: { status: "pass", paperKey: "arXiv:1706.03762", pages: 15, chars: 12345 },
    },
    embedding: {
      status: "pass",
      mode: "local",
      modelId: "multilingual-e5-small-q8",
      dimension: 384,
      remoteHost: "https://huggingface.co/",
      wasmPaths: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/",
      runtimeProbe: "process.release.name=electron (electron 33.4.11)",
      loadMs: 1234,
    },
  };
}

describe("formatFullTextRuntimeDiagnostics", () => {
  it("renders a deterministic full report for a pass run", () => {
    expect(formatFullTextRuntimeDiagnostics(passReport())).toBe(
      [
        "arXiv Daily full-text runtime diagnostics",
        "",
        "library:",
        "  connection: connected",
        "  scope fingerprint: scope-hex",
        "  catalog papers: 5",
        "",
        "pdf.js:",
        "  loadPdfJs(): resolved",
        "  loadPdfJs() return value: present",
        "  window.pdfjsLib: present (version 4.2.189)",
        "  smoke extraction: pass",
        "    paper: arXiv:1706.03762",
        "    pages: 15",
        "    chars: 12345",
        "  status: pass",
        "",
        "embeddings:",
        "  mode: local",
        "  model: multilingual-e5-small-q8",
        "  dimension: 384",
        "  remoteHost: https://huggingface.co/",
        "  wasmPaths: https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/",
        "  runtime probe: process.release.name=electron (electron 33.4.11)",
        "  model load + probe: 1234 ms",
        "  status: pass",
      ].join("\n"),
    );
  });

  it("reports a failing embedding part with its error and keeps pdf.js lines", () => {
    const report = passReport();
    report.embedding = {
      status: "fail",
      mode: "local",
      modelId: "multilingual-e5-small-q8",
      dimension: 384,
      error: "fetch failed",
    };
    const text = formatFullTextRuntimeDiagnostics(report);
    expect(text).toContain("embeddings:");
    expect(text).toContain("  error: fetch failed");
    expect(text).toContain("  status: fail");
    expect(text).toContain("  smoke extraction: pass");
  });

  it("marks pdf.js skipped when there is no library smoke, with the reason", () => {
    const report = passReport();
    report.library = { connected: false };
    report.pdfJs = {
      status: "skipped",
      loadPdfJsResolved: true,
      loaderReturnedLib: true,
      windowPdfJsLibPresent: true,
      windowPdfJsLibVersion: "4.2.189",
      smoke: {
        status: "skipped",
        error: "no library connection — smoke extraction skipped",
      },
    };
    const text = formatFullTextRuntimeDiagnostics(report);
    expect(text).toContain("  connection: none");
    expect(text).toContain("  smoke extraction: skipped");
    expect(text).toContain("    error: no library connection — smoke extraction skipped");
    expect(text).toContain("  status: skipped");
  });
});

describe("summarizeFullTextRuntimeDiagnostics", () => {
  it("summarizes a pass run in one line", () => {
    const summary = summarizeFullTextRuntimeDiagnostics(passReport());
    expect(summary).toBe(
      "pdf.js PASS; window.pdfjsLib v4.2.189; smoke 15 pages / 12345 chars | "
        + "local embeddings PASS; 384 dim; 1234 ms",
    );
  });

  it("carries fail reasons into the summary", () => {
    const report = passReport();
    report.pdfJs = {
      status: "fail",
      loadPdfJsResolved: true,
      loaderReturnedLib: false,
      windowPdfJsLibPresent: false,
      smoke: { status: "fail", error: "extraction failed" },
    };
    report.embedding = {
      status: "fail",
      mode: "local",
      modelId: "multilingual-e5-small-q8",
      dimension: 384,
      error: "dimension mismatch",
    };
    const summary = summarizeFullTextRuntimeDiagnostics(report);
    expect(summary).toContain("pdf.js FAIL");
    expect(summary).toContain("extraction failed");
    expect(summary).toContain("embeddings FAIL");
    expect(summary).toContain("dimension mismatch");
  });

  it("labels a skipped pdf.js part without a library", () => {
    const report = passReport();
    report.library = { connected: false };
    report.pdfJs = {
      status: "skipped",
      loadPdfJsResolved: true,
      loaderReturnedLib: true,
      windowPdfJsLibPresent: true,
      smoke: { status: "skipped" },
    };
    expect(summarizeFullTextRuntimeDiagnostics(report)).toContain("pdf.js skipped (no library)");
  });
});

describe("describeDiagnosticsError", () => {
  it("uses the Error message and falls back to String()", () => {
    expect(describeDiagnosticsError(new Error("boom"))).toBe("boom");
    expect(describeDiagnosticsError("plain string")).toBe("plain string");
    expect(describeDiagnosticsError(42)).toBe("42");
  });
});

describe("diagnoseEmbedding backend selection (ADR 0008)", () => {
  function remotePlugin(embedding: Partial<EmbeddingSettings>) {
    const plugin = Object.create(ArxivDailyPlugin.prototype) as any;
    const request = vi.fn(async () => ({
      status: 200,
      bodyText: JSON.stringify({ data: [{ embedding: [0.1, 0.2] }] }),
    }));
    Object.assign(plugin, {
      settings: {
        ...structuredClone(DEFAULT_SETTINGS),
        embedding: {
          ...DEFAULT_SETTINGS.embedding,
          mode: "remote" as const,
          baseUrl: "https://api.openai.com/v1",
          apiKey: "sk-embed",
          model: "text-embedding-3-small",
          dimension: 2,
          ...embedding,
        },
      },
      logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
      host: { http: { request }, storage: { normalizePath: (path: string) => path } },
      libraryConnection: undefined,
    });
    return { plugin, request };
  }

  function authorize(plugin: any): void {
    plugin.libraryConnection = authorizeLibraryConnection(
      createLibraryConnection("/papers", "1:2"),
      {
        llmBaseUrl: plugin.settings.llm.baseUrl,
        embeddingEndpoint: { baseUrl: plugin.settings.embedding.baseUrl },
      },
    );
  }

  it("reports an incomplete remote configuration without touching either backend", async () => {
    const { plugin, request } = remotePlugin({ apiKey: "" });

    const diagnostics = await plugin.diagnoseEmbedding();

    expect(diagnostics).toMatchObject({
      status: "fail",
      mode: "remote",
      authorization: "disconnected",
    });
    expect(diagnostics.error).toMatch(/configuration incomplete/i);
    expect(request).not.toHaveBeenCalled();
    expect(transformersRuntime.pipeline).not.toHaveBeenCalled();
  });

  it("skips the remote probe when full-text processing is not authorized", async () => {
    const { plugin, request } = remotePlugin({});

    const diagnostics = await plugin.diagnoseEmbedding();

    expect(diagnostics).toMatchObject({
      status: "skipped",
      mode: "remote",
      authorization: "disconnected",
    });
    expect(diagnostics.error).toMatch(/authorizing full-text processing/i);
    expect(request).not.toHaveBeenCalled();
    expect(transformersRuntime.pipeline).not.toHaveBeenCalled();
  });

  it("probes the configured remote endpoint and never loads the local model", async () => {
    const { plugin, request } = remotePlugin({});
    authorize(plugin);

    const diagnostics = await plugin.diagnoseEmbedding();

    expect(diagnostics).toMatchObject({
      status: "pass",
      mode: "remote",
      authorization: "authorized",
      modelId: "remote:text-embedding-3-small:2",
      dimension: 2,
    });
    expect(request).toHaveBeenCalledOnce();
    expect(request.mock.calls[0]![0].url).toBe("https://api.openai.com/v1/embeddings");
    expect(transformersRuntime.pipeline).not.toHaveBeenCalled();
  });
});
