import type { MarkupParser } from "@arxiv-daily/core";
import { DOMParser } from "linkedom";

export class LinkedomMarkupParser implements MarkupParser {
  parseFromString(
    markup: string,
    mimeType: "text/html" | "text/xml",
  ): Document {
    return new DOMParser().parseFromString(markup, mimeType) as unknown as Document;
  }
}
