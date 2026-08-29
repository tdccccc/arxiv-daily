import type {
  ParsedDocument,
  ParserCapability,
} from "../../documents/parsed-document";
import type { PdfExtractionResult, PdfLayoutLine } from "./ports";

export function parsedDocumentToPdfExtractionResult(
  document: ParsedDocument,
  capabilities: readonly ParserCapability[],
): PdfExtractionResult {
  const pages: string[] = [];
  const layout: PdfLayoutLine[][] = [];
  const providesLayout = capabilities.includes("text-layout");

  for (let index = 0; index < document.blocks.length; index++) {
    const block = document.blocks[index]!;
    const expectedPage = index + 1;
    if (block.kind !== "page") {
      throw new Error(
        "Legacy PDF text projection requires top-level page blocks",
      );
    }
    if (block.locator.page !== expectedPage) {
      throw new Error(
        `Legacy PDF text projection expected page ${expectedPage}`,
      );
    }
    if (block.locator.block !== index) {
      throw new Error(
        `Legacy PDF text projection expected block ${index}`,
      );
    }

    pages.push(block.text);
    if (providesLayout) layout.push([...(block.layout ?? [])]);
  }

  return {
    pages,
    ...(providesLayout ? { layout } : {}),
    ...(capabilities.includes("document-metadata") && document.metadata?.title !== undefined
      ? { metadataTitle: document.metadata.title }
      : {}),
  };
}
