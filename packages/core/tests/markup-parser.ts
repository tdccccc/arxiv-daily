import type { MarkupParser } from "../src/core/adapters";
import { DOMParser } from "linkedom";

/**
 * Core tests parse markup with linkedom, the same library the production Node
 * host adapter (`LinkedomMarkupParser`) uses.
 *
 * This helper used to build a fresh happy-dom `Window` per call. happy-dom
 * retains every parsed `Document` until the owning window is closed
 * asynchronously, which a synchronous `MarkupParser` cannot do without
 * destroying the document it just returned. Suites that parse the large arXiv
 * listing fixture repeatedly therefore accumulated ~130 MB per parse and
 * exhausted the vitest worker heap. linkedom holds no such per-parse state.
 */
export const markupParser: MarkupParser = {
  parseFromString(markup, mimeType) {
    const normalized =
      mimeType === "text/xml" ? markup.replace(/^<\?xml[^?]*\?>\s*/i, "") : markup;
    return new DOMParser().parseFromString(normalized, mimeType) as unknown as Document;
  },
};
