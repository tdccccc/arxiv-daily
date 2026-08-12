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
  PdfLayoutLine,
  PdfTextExtractor,
} from "@arxiv-daily/core";

/**
 * Obsidian serves its bundled PDF.js assets under these absolute paths in the
 * renderer process (same origin as the plugin; see the pdf.js options in
 * Obsidian's app.js: `cMapUrl: "/lib/pdfjs/cmaps/"`,
 * `standardFontDataUrl: "/lib/pdfjs/standard_fonts/"`). PDFs whose fonts are
 * not embedded (the standard 14 fonts) or that use CID/CMap encodings require
 * them — without these parameters pdf.js's text extraction throws
 * UnknownErrorException ("Ensure that the `standardFontDataUrl` API parameter
 * is provided"), failing the whole extraction for such files.
 */
const PDFJS_CMAP_URL = "/lib/pdfjs/cmaps/";
const PDFJS_STANDARD_FONTS_URL = "/lib/pdfjs/standard_fonts/";

/**
 * Minimal, version-stable subset of the PDF.js API this host relies on.
 * Mirrors the surface every pdf.js >= 2.x exposes: getDocument ->
 * loading task -> document proxy -> page -> text content -> items[].str.
 * The injected library may be Obsidian's bundled build (any version) or a
 * pdfjs-dist build in tests.
 */
export interface PdfJsLib {
  getDocument(src: {
    data: Uint8Array;
    /** URL prefix of the pdf.js CMap assets (Obsidian renderer path). */
    cMapUrl?: string;
    /** CMaps are packed .bcmap files (Obsidian's `cMapPacked: true`). */
    cMapPacked?: boolean;
    /** URL prefix of the pdf.js standard-font assets (Obsidian renderer path). */
    standardFontDataUrl?: string;
  }): PdfJsLoadingTask;
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
  /** Document metadata (info.Title is the machine-readable title). */
  getMetadata?(): Promise<{ info?: { Title?: string } }>;
}

export interface PdfJsPage {
  getTextContent(): Promise<PdfJsTextContent>;
  /** Best-effort release of page resources; present in pdf.js >= 2.x. */
  cleanup?(): void;
  /** Page box [x1, y1, x2, y2] in PDF units; used for vertical layout positions. */
  view?: readonly [number, number, number, number];
}

export interface PdfJsTextContent {
  items: readonly PdfJsTextItem[];
}

export interface PdfJsTextItem {
  /** Present on real text items; marked-content items have no `str`. */
  str?: string;
  /** True when the item ends a line (pdf.js >= 2.x). */
  hasEOL?: boolean;
  /** Text-space transform matrix [a, b, c, d, e, f]; the font size and the
   * baseline position are derived from it. */
  transform?: readonly number[];
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
      // Obsidian's pdf.js asset URLs are required for PDFs with non-embedded
      // (standard) or CID/CMap fonts; see the constants above. The bytes are
      // copied because pdf.js transfers (detaches) the buffer it receives to
      // its worker — the original bytes stay usable for the metadata-title
      // fallback (`rawInfoTitle` reads the file head after extraction).
      loadingTask = lib.getDocument({
        data: new Uint8Array(bytes),
        cMapUrl: PDFJS_CMAP_URL,
        cMapPacked: true,
        standardFontDataUrl: PDFJS_STANDARD_FONTS_URL,
      });
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
        return await this.extractAllPages(doc, bytes, signal);
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

  private async metadataTitle(
    doc: PdfJsDocument,
    bytes: Uint8Array,
  ): Promise<string | undefined> {
    let title: string | undefined;
    if (doc.getMetadata) {
      try {
        const metadata = await doc.getMetadata();
        title = metadata?.info?.Title?.trim() || undefined;
      } catch {
        // Unreadable metadata must not fail the extraction (core validates
        // the title anyway).
      }
    }
    // Obsidian's bundled pdf.js can resolve a duplicate /Title key (a
    // literal plus an indirect reference) to the wrong/empty entry, while
    // the Info dict's first literal holds the real title. Fall back to a
    // byte-level parse of the head metadata region.
    return title ?? rawInfoTitle(bytes);
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
    bytes: Uint8Array,
    signal?: AbortSignal,
  ): Promise<PdfExtractionResult> {
    const pages: string[] = [];
    const layout: PdfLayoutLine[][] = [];
    for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber++) {
      try {
        const page = await raceWithAbort(doc.getPage(pageNumber), signal);
        try {
          const text = await raceWithAbort(page.getTextContent(), signal);
          const built = buildPageLayout(text.items, page.view);
          pages.push(built.text);
          layout.push(built.lines);
        } catch {
          if (signal?.aborted) throwAbortError(signal);
          // Core contract: malformed pages degrade to empty strings; only
          // document-level failures throw.
          pages.push("");
          layout.push([]);
        } finally {
          try {
            page.cleanup?.();
          } catch {
            // Resource release must never fail the extraction: a cleanup
            // error in a finally block would swallow the extracted text.
          }
        }
      } catch {
        if (signal?.aborted) throwAbortError(signal);
        pages.push("");
        layout.push([]);
      }
    }
    return { pages, layout, metadataTitle: await this.metadataTitle(doc, bytes) };
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

/**
 * Byte-level /Title parse of the head metadata region, used when the host's
 * pdf.js resolved the Info dict's /Title to an empty or indirect entry. Reads
 * the first literal or UTF-16 hex /Title in the first 256 KiB (Info dicts
 * live at the file head) and decodes PDF string escapes.
 */
function rawInfoTitle(bytes: Uint8Array): string | undefined {
  const latin = new TextDecoder("iso-8859-1").decode(bytes.subarray(0, 256 * 1024));
  const literal = latin.match(/\/Title\s*\(((?:\\.|[^()\\]){1,400})\)/i);
  if (literal) {
    return decodePdfStringLiteral(literal[1]!);
  }
  const hex = latin.match(/\/Title\s*<([0-9A-Fa-f]{2,})>/i);
  if (hex) {
    const content = hex[1]!;
    const utf16 = /^(?:feff|fffe)/i.test(content);
    const out: number[] = [];
    for (let index = 0; index + 1 < content.length; index += 2) {
      out.push(parseInt(content.slice(index, index + 2), 16));
    }
    if (utf16) {
      const littleEndian = content.startsWith("fffe");
      const codeUnits: number[] = [];
      for (let index = 2; index + 1 < out.length; index += 2) {
        codeUnits.push(littleEndian
          ? out[index]! | (out[index + 1]! << 8)
          : (out[index]! << 8) | out[index + 1]!);
      }
      return String.fromCharCode(...codeUnits);
    }
    return String.fromCharCode(...out);
  }
  return undefined;
}

/** Decode PDF string literal escapes (`\(` -> `(`, `\\` -> `\`, `\n` etc.). */
function decodePdfStringLiteral(value: string): string {
  return value
    .replace(/\\([nrtbf])/g, (_, code: string) => (
      { n: "\n", r: "\r", t: "\t", b: "\b", f: "\f" } as Record<string, string>
    )[code]!)
    .replace(/\\(.)/g, "$1");
}

/**
 * Join text items in their given (reading) order into one page string, and
 * derive the typographic line layout (text + font size + vertical position)
 * that core uses for fallback title extraction. The two share the same line
 * grouping, so the layout's line texts are exactly the page's lines.
 */
function buildPageLayout(
  items: readonly PdfJsTextItem[],
  view?: readonly [number, number, number, number],
): { text: string; lines: PdfLayoutLine[] } {
  const height = view ? view[3] - view[1] : undefined;
  const lines: Array<{ text: string; fontSize: number; y: number }> = [];
  let current: { text: string; fontSize: number; y: number } | null = null;
  let lineEnded = false;
  for (const item of items) {
    const str = item.str;
    if (str) {
      if (!current) {
        current = { text: "", fontSize: 0, y: item.transform?.[5] ?? 0 };
      }
      if (current.text.length === 0) {
        current.text = str;
      } else if (current.text.endsWith("-")) {
        // Rejoin LaTeX hyphenation ("inter-" + "pret" -> "interpret").
        current.text += str;
      } else if (lineEnded) {
        lines.push(current);
        current = { text: str, fontSize: 0, y: item.transform?.[5] ?? 0 };
      } else {
        current.text += ` ${str}`;
      }
      const transform = item.transform;
      if (transform) {
        // Font size = scale of the text-space matrix; baseline y = e/f entry.
        current.fontSize = Math.max(current.fontSize, Math.hypot(transform[2]!, transform[3]!));
      }
      lineEnded = false;
    }
    // pdf.js commonly represents a line ending as an empty text item. The
    // marker belongs to the current item and must be remembered for the next
    // non-empty item rather than discarded with the empty string.
    if (item.hasEOL) lineEnded = true;
  }
  if (current) lines.push(current);
  return {
    text: lines.map((line) => line.text).join("\n"),
    lines: lines.map((line) => ({
      text: line.text,
      fontSize: line.fontSize,
      topFraction: height && height > 0 ? (height - line.y) / height : 0,
    })),
  };
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
