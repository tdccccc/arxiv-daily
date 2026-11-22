/** The ASCII signature every PDF file starts with. */
const PDF_HEADER = [0x25, 0x50, 0x44, 0x46, 0x2d]; // "%PDF-"

/**
 * The PDF specification allows leading bytes before the header, and readers
 * only look for it in the first 1024 bytes.
 */
const PDF_HEADER_SEARCH_LIMIT = 1024;

/**
 * Whether `buffer` carries a PDF header in the region readers inspect. arXiv
 * answers some requests (rate limits, captcha pages, not-yet-generated PDFs)
 * with an HTTP 200 HTML page; this rejects that body before it is saved as a
 * `.pdf` or recorded in the paper index.
 */
export function hasPdfMagic(buffer: ArrayBuffer): boolean {
  const bytes = new Uint8Array(buffer);
  const end = Math.min(bytes.length, PDF_HEADER_SEARCH_LIMIT);
  for (let start = 0; start + PDF_HEADER.length <= end; start += 1) {
    let matches = true;
    for (let offset = 0; offset < PDF_HEADER.length; offset += 1) {
      if (bytes[start + offset] !== PDF_HEADER[offset]) {
        matches = false;
        break;
      }
    }
    if (matches) return true;
  }
  return false;
}
