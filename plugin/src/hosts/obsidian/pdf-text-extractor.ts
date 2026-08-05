/**
 * Obsidian host implementation of the core `PdfTextExtractor` port, backed by
 * Obsidian's built-in PDF.js (exposed as `window.pdfjsLib`).
 *
 * Access mechanism: the official `obsidian` package (v1.13.1) declares
 * `loadPdfJs(): Promise<any>` — "Load PDF.js and return a promise to the
 * global pdfjsLib object. Can also use `window.pdfjsLib` after this promise
 * resolves to get the same reference." This host therefore reads
 * `window.pdfjsLib` (injectable for tests); callers should let
 * `loadPdfJs()` (from the "obsidian" module) resolve before extracting.
 * NOTE: actual presence/version of `window.pdfjsLib` inside the plugin
 * runtime must be confirmed in Obsidian (see plan phase 01 risk table).
 *
 * Cancellation: PDF.js `getDocument` has no `signal` parameter (verified
 * against pdfjs-dist 6.2.108 types and build); the supported cancel path is
 * `PDFDocumentLoadingTask.destroy()`, which is wired to the caller's
 * AbortSignal. PDF.js abandons in-flight page promises on destroy (they
 * never settle), so every await is raced against the signal and converted to
 * an `AbortError`-named error that core's `isCancellationError` recognizes.
 *
 * Only type imports from core — no runtime coupling — keeping this file a
 * pure host boundary.
 */

import type {
  PdfExtractionOptions,
  PdfExtractionResult,
  PdfTextExtractor,
} from "@arxiv-daily/core";

/**
 * Minimal, version-stable subset of the PDF.js API this host relies on.
 * Mirrors the surface every pdf.js >= 2.x exposes: getDocument ->
 * loading task -> document proxy -> page -> text content -> items[].str.
 * The injected library may be Obsidian's bundled build (any version) or a
 * pdfjs-dist build in tests.
 */
export interface PdfJsLib {
  getDocument(src: { data: Uint8Array }): PdfJsLoadingTask;
}

export interface PdfJsLoadingTask {
  /** Resolves to the document proxy once the PDF is parsed. */
  promise: Promise<PdfJsDocument>;
  /** Cancels loading/parsing; the `promise` rejects with an AbortException. */
  destroy(): Promise<void>;
}

export interface PdfJsDocument {
  numPages: number;
  getPage(pageNumber: number): Promise<PdfJsPage>;
  /** Optional resource release; return type varies across pdf.js versions. */
  cleanup?(): unknown;
}

export interface PdfJsPage {
  getTextContent(): Promise<PdfJsTextContent>;
  /** Best-effort release of page resources; present in pdf.js >= 2.x. */
  cleanup?(): void;
}

export interface PdfJsTextContent {
  items: readonly PdfJsTextItem[];
}

export interface PdfJsTextItem {
  /** Present on real text items; marked-content items have no `str`. */
  str?: string;
  /** True when the item ends a line (pdf.js >= 2.x). */
  hasEOL?: boolean;
}

export class ObsidianPdfTextExtractor implements PdfTextExtractor {
  private readonly pdfjsLib: PdfJsLib | undefined;

  /**
   * @param pdfjsLib Injected PDF.js library object. Defaults to
   *   `window.pdfjsLib` (Obsidian's built-in build); pass explicitly in
   *   tests.
   */
  constructor(pdfjsLib?: PdfJsLib) {
    this.pdfjsLib = pdfjsLib ?? defaultPdfJsLib();
  }

  async extractPdfText(
    bytes: Uint8Array,
    options?: PdfExtractionOptions,
  ): Promise<PdfExtractionResult> {
    const lib = this.requirePdfJs();
    const signal = options?.signal;
    if (signal?.aborted) throwAbortError(signal);

    let loadingTask: PdfJsLoadingTask;
    try {
      loadingTask = lib.getDocument({ data: bytes });
    } catch (error) {
      throw new Error(
        `PDF extraction failed to start: ${describeError(error)}`,
        { cause: error },
      );
    }

    // PDF.js cancels through the loading task's destroy(), not a signal
    // option on getDocument (not supported by the API).
    const abortHandler = () => {
      void loadingTask.destroy().catch(() => undefined);
    };
    signal?.addEventListener("abort", abortHandler, { once: true });

    try {
      const doc = await this.openDocument(loadingTask, signal);
      try {
        return { pages: await this.extractAllPages(doc, signal) };
      } finally {
        // Document teardown belongs to the loading task (PDFDocumentProxy
        // has no destroy()); idempotent, and safe if the abort handler
        // already ran.
        await loadingTask.destroy().catch(() => undefined);
      }
    } finally {
      signal?.removeEventListener("abort", abortHandler);
    }
  }

  private requirePdfJs(): PdfJsLib {
    if (this.pdfjsLib) return this.pdfjsLib;
    throw new Error(
      "Obsidian's built-in pdf.js is not available. Call `loadPdfJs()` " +
        "(from the \"obsidian\" module) and wait for it to resolve before " +
        "extracting, or inject a pdf.js library into ObsidianPdfTextExtractor.",
    );
  }

  private async openDocument(
    loadingTask: PdfJsLoadingTask,
    signal?: AbortSignal,
  ): Promise<PdfJsDocument> {
    try {
      return await raceWithAbort(loadingTask.promise, signal);
    } catch (error) {
      if (signal?.aborted) throwAbortError(signal);
      throw new Error(
        `Failed to open PDF: ${describePdfOpenError(error)}`,
        { cause: error },
      );
    }
  }

  private async extractAllPages(
    doc: PdfJsDocument,
    signal?: AbortSignal,
  ): Promise<string[]> {
    const pages: string[] = [];
    for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber++) {
      try {
        const page = await raceWithAbort(doc.getPage(pageNumber), signal);
        try {
          const text = await raceWithAbort(page.getTextContent(), signal);
          pages.push(joinPageText(text.items));
        } finally {
          page.cleanup?.();
        }
      } catch {
        if (signal?.aborted) throwAbortError(signal);
        // Core contract: malformed pages degrade to empty strings; only
        // document-level failures throw.
        pages.push("");
      }
    }
    return pages;
  }
}

/**
 * Race a PDF.js promise against the abort signal. PDF.js abandons in-flight
 * page promises when the loading task is destroyed (they never settle), so
 * without this every post-abort await would hang forever.
 */
function raceWithAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    );
  });
}

function defaultPdfJsLib(): PdfJsLib | undefined {
  if (typeof window === "undefined") return undefined;
  return (window as unknown as { pdfjsLib?: PdfJsLib }).pdfjsLib;
}

/** Join text items in their given (reading) order into one page string. */
function joinPageText(items: readonly PdfJsTextItem[]): string {
  let text = "";
  for (const item of items) {
    const str = item.str;
    if (!str) continue;
    if (text.length === 0) {
      text = str;
      continue;
    }
    if (text.endsWith("-")) {
      // Rejoin LaTeX hyphenation ("inter-" + "pret" -> "interpret").
      text += str;
    } else if (item.hasEOL) {
      text += `\n${str}`;
    } else {
      text += ` ${str}`;
    }
  }
  return text;
}

/** Map pdf.js open failures to a readable reason. */
function describePdfOpenError(error: unknown): string {
  if (error instanceof Error) {
    switch (error.name) {
      case "InvalidPDFException":
        return "invalid or corrupted PDF structure";
      case "PasswordException":
        return "PDF is password protected";
      case "MissingPDFException":
        return "PDF data is missing or unreadable";
      case "UnexpectedResponseException":
        return "PDF data could not be fetched";
      case "AbortException":
        return "PDF loading was aborted";
      default:
        return error.message ? `${error.name}: ${error.message}` : error.name;
    }
  }
  return String(error);
}

function describeError(error: unknown): string {
  return error instanceof Error && error.message
    ? error.message
    : String(error);
}

/**
 * Mirror core's cancellation convention (`throwIfCancelled` /
 * `isCancellationError`): an Error named "AbortError" with the signal reason
 * as message, so downstream core code treats this as cancellation.
 */
function abortError(signal: AbortSignal): Error {
  const reason = (signal as { reason?: unknown }).reason;
  const message =
    typeof reason === "string" && reason ? reason : "cancelled by user";
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

function throwAbortError(signal: AbortSignal): never {
  throw abortError(signal);
}
