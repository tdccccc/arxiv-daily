/**
 * Report types and presenters for the `diagnose-fulltext-runtime` command.
 *
 * Probing happens in the plugin class (main.ts) — the hosts it touches
 * (Obsidian's pdf.js, transformers.js) need the plugin context; this file
 * only defines the report shape and its deterministic text renderings
 * (summary + full report), kept pure so tests can lock the output.
 */

export type DiagnosticsStatus = "pass" | "fail" | "skipped";

export interface PdfJsSmokeDiagnostics {
  status: DiagnosticsStatus;
  /** paperKey of the paper whose PDF was used for the real extraction. */
  paperKey?: string;
  /** Number of pages extracted by the real smoke run. */
  pages?: number;
  /** Total characters across all extracted pages. */
  chars?: number;
  error?: string;
}

export interface PdfJsDiagnostics {
  status: DiagnosticsStatus;
  /** `loadPdfJs()` from the obsidian module resolved without throwing. */
  loadPdfJsResolved: boolean;
  /** The promise resolved to a pdf.js library object. */
  loaderReturnedLib: boolean;
  /** `window.pdfjsLib` is present after `loadPdfJs()` (production path). */
  windowPdfJsLibPresent: boolean;
  windowPdfJsLibVersion?: string;
  /** Real extraction smoke on the first library PDF (absent when none). */
  smoke?: PdfJsSmokeDiagnostics;
  error?: string;
}

export interface EmbeddingDiagnostics {
  status: DiagnosticsStatus;
  modelId: string;
  dimension: number;
  /** transformers.js `env.remoteHost` after a load attempt. */
  remoteHost?: string;
  /** transformers.js ONNX wasm paths (URL string, or JSON for a map). */
  wasmPaths?: string;
  /** Wall-clock ms for the first embed (includes model download on cold cache). */
  loadMs?: number;
  /** Runtime branch probe (process.release.name / electron marker). */
  runtimeProbe?: string;
  error?: string;
}

export interface LibraryDiagnostics {
  connected: boolean;
  scopeFingerprint?: string;
  paperCount?: number;
}

export interface FullTextRuntimeDiagnostics {
  library: LibraryDiagnostics;
  pdfJs: PdfJsDiagnostics;
  embedding: EmbeddingDiagnostics;
}

/** Stable single-line rendering of an unknown error for report fields. */
export function describeDiagnosticsError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
}

/**
 * Deterministic full report for the diagnostics modal and console capture.
 * One fact per line, no timestamps, so repeated runs diff cleanly.
 */
export function formatFullTextRuntimeDiagnostics(
  report: FullTextRuntimeDiagnostics,
): string {
  const { library, pdfJs, embedding } = report;
  const lines: string[] = ["arXiv Daily full-text runtime diagnostics"];
  lines.push("");
  lines.push("library:");
  lines.push(`  connection: ${library.connected ? "connected" : "none"}`);
  if (library.scopeFingerprint) {
    lines.push(`  scope fingerprint: ${library.scopeFingerprint}`);
  }
  if (library.paperCount !== undefined) {
    lines.push(`  catalog papers: ${library.paperCount}`);
  }
  lines.push("");
  lines.push("pdf.js:");
  lines.push(`  loadPdfJs(): ${pdfJs.loadPdfJsResolved ? "resolved" : "failed"}`);
  lines.push(
    `  loadPdfJs() return value: ${pdfJs.loaderReturnedLib ? "present" : "absent"}`,
  );
  const windowLine = pdfJs.windowPdfJsLibPresent
    ? `present${pdfJs.windowPdfJsLibVersion ? ` (version ${pdfJs.windowPdfJsLibVersion})` : ""}`
    : "absent";
  lines.push(`  window.pdfjsLib: ${windowLine}`);
  if (pdfJs.smoke) {
    lines.push(`  smoke extraction: ${pdfJs.smoke.status}`);
    if (pdfJs.smoke.paperKey) lines.push(`    paper: ${pdfJs.smoke.paperKey}`);
    if (pdfJs.smoke.pages !== undefined) lines.push(`    pages: ${pdfJs.smoke.pages}`);
    if (pdfJs.smoke.chars !== undefined) lines.push(`    chars: ${pdfJs.smoke.chars}`);
    if (pdfJs.smoke.error) lines.push(`    error: ${pdfJs.smoke.error}`);
  }
  if (pdfJs.error) lines.push(`  error: ${pdfJs.error}`);
  lines.push(`  status: ${pdfJs.status}`);
  lines.push("");
  lines.push("embeddings:");
  lines.push(`  model: ${embedding.modelId}`);
  lines.push(`  dimension: ${embedding.dimension}`);
  if (embedding.remoteHost) lines.push(`  remoteHost: ${embedding.remoteHost}`);
  if (embedding.wasmPaths) lines.push(`  wasmPaths: ${embedding.wasmPaths}`);
  if (embedding.runtimeProbe) lines.push(`  runtime probe: ${embedding.runtimeProbe}`);
  if (embedding.loadMs !== undefined) {
    lines.push(`  model load + probe: ${embedding.loadMs} ms`);
  }
  if (embedding.error) lines.push(`  error: ${embedding.error}`);
  lines.push(`  status: ${embedding.status}`);
  return lines.join("\n");
}

/** One-line summary for the completion Notice. */
export function summarizeFullTextRuntimeDiagnostics(
  report: FullTextRuntimeDiagnostics,
): string {
  const { pdfJs, embedding } = report;
  const pdfJsBits: string[] = [];
  if (pdfJs.status === "pass") {
    pdfJsBits.push("pdf.js PASS");
    if (pdfJs.windowPdfJsLibVersion) {
      pdfJsBits.push(`window.pdfjsLib v${pdfJs.windowPdfJsLibVersion}`);
    }
    if (pdfJs.smoke?.status === "pass" && pdfJs.smoke.pages !== undefined && pdfJs.smoke.chars !== undefined) {
      pdfJsBits.push(`smoke ${pdfJs.smoke.pages} pages / ${pdfJs.smoke.chars} chars`);
    }
  } else if (pdfJs.status === "fail") {
    pdfJsBits.push("pdf.js FAIL");
    if (pdfJs.error) pdfJsBits.push(pdfJs.error);
    if (pdfJs.smoke?.error) pdfJsBits.push(pdfJs.smoke.error);
  } else {
    pdfJsBits.push("pdf.js skipped (no library)");
  }
  const embeddingBits: string[] = [];
  if (embedding.status === "pass") {
    embeddingBits.push("embeddings PASS");
    embeddingBits.push(`${embedding.dimension} dim`);
    if (embedding.loadMs !== undefined) embeddingBits.push(`${embedding.loadMs} ms`);
  } else if (embedding.status === "fail") {
    embeddingBits.push("embeddings FAIL");
    if (embedding.error) embeddingBits.push(embedding.error);
  } else {
    embeddingBits.push("embeddings skipped");
  }
  return `${pdfJsBits.join("; ")} | ${embeddingBits.join("; ")}`;
}
