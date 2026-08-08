import { describe, expect, it } from "vitest";
import {
  describeDiagnosticsError,
  formatFullTextRuntimeDiagnostics,
  summarizeFullTextRuntimeDiagnostics,
  type FullTextRuntimeDiagnostics,
} from "../src/services/fulltext-runtime-diagnostics";

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
        + "embeddings PASS; 384 dim; 1234 ms",
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
