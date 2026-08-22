import { describe, expect, it } from "vitest";
import {
  SIDECAR_PARSER_PROTOCOL_VERSION,
  decodeSidecarParseResponse,
  decodeSidecarParserCapabilities,
  requireLoopbackSidecarUrl,
} from "../src/documents/sidecar-document-parser";

const parser = { id: "docling", version: "2.0" };

describe("sidecar document parser protocol", () => {
  it("accepts a bounded loopback capability document and structured parse response", () => {
    expect(requireLoopbackSidecarUrl("http://127.0.0.1:5001/v1/parse").toString())
      .toBe("http://127.0.0.1:5001/v1/parse");
    expect(decodeSidecarParserCapabilities({
      protocolVersion: SIDECAR_PARSER_PROTOCOL_VERSION,
      parser,
      capabilities: ["page-text", "document-structure", "document-metadata"],
      maxRequestBytes: 5_000_000,
      maxResponseBytes: 2_000_000,
    })).toEqual({
      protocolVersion: 1,
      parser,
      capabilities: ["page-text", "document-structure", "document-metadata"],
      maxRequestBytes: 5_000_000,
      maxResponseBytes: 2_000_000,
    });
    expect(decodeSidecarParseResponse({
      protocolVersion: 1,
      parser,
      document: {
        mediaType: "application/pdf",
        metadata: { title: "Structured paper" },
        blocks: [{
          kind: "heading",
          text: "Methods",
          headingLevel: 1,
          locator: { page: 2, block: 7, charStart: 0, charEnd: 7 },
        }, {
          kind: "paragraph",
          text: "A bounded parser response.",
          locator: { page: 2, block: 8 },
        }],
      },
    })).toMatchObject({
      parser,
      document: {
        mediaType: "application/pdf",
        metadata: { title: "Structured paper" },
        blocks: [{ kind: "heading", headingLevel: 1 }, { kind: "paragraph" }],
      },
    });
  });

  it.each([
    "https://127.0.0.1:5001/v1/parse",
    "http://localhost:5001/v1/parse",
    "http://192.168.1.2:5001/v1/parse",
    "http://127.0.0.1:5001/v1/parse?redirect=https://example.test",
  ])("rejects a sidecar endpoint outside the strict loopback contract: %s", (url) => {
    expect(() => requireLoopbackSidecarUrl(url)).toThrow(/loopback/);
  });

  it("fails closed on paths, unknown fields, unsupported capabilities, and malformed locators", () => {
    expect(() => decodeSidecarParseResponse({
      protocolVersion: 1,
      parser,
      path: "/private/library/paper.pdf",
      document: { mediaType: "application/pdf", blocks: [] },
    })).toThrow(/unknown field/);
    expect(() => decodeSidecarParserCapabilities({
      protocolVersion: 1,
      parser,
      capabilities: ["directory-scan"],
      maxRequestBytes: 5_000_000,
      maxResponseBytes: 2_000_000,
    })).toThrow(/capability/);
    expect(() => decodeSidecarParseResponse({
      protocolVersion: 1,
      parser,
      document: {
        mediaType: "application/pdf",
        blocks: [{ kind: "paragraph", text: "bad", locator: { page: 0 } }],
      },
    })).toThrow(/locator\.page/);
  });
});
