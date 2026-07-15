import type { MarkupParser } from "../src/core/adapters";
import { Window } from "happy-dom";

export const markupParser: MarkupParser = {
  parseFromString(markup, mimeType) {
    const window = new Window();
    const normalized =
      mimeType === "text/xml" ? markup.replace(/^<\?xml[^?]*\?>\s*/i, "") : markup;
    return new window.DOMParser().parseFromString(normalized, mimeType);
  },
};
