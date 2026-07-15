import type { MarkupParser } from "@arxiv-daily/core";

export class ObsidianMarkupParser implements MarkupParser {
  parseFromString(
    markup: string,
    mimeType: "text/html" | "text/xml",
  ): Document {
    return new DOMParser().parseFromString(markup, mimeType);
  }
}
