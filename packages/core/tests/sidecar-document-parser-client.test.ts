import { describe, expect, it, vi } from "vitest";
import type { HttpClient, HttpRequest, HttpResponse } from "../src/core/adapters";
import {
  SidecarDocumentParserError,
  probeLoopbackSidecarParser,
} from "../src/documents/sidecar-document-parser-client";

const parser = { id: "docling", version: "2.0" };

function http(responses: HttpResponse[]): { client: HttpClient; requests: HttpRequest[] } {
  const requests: HttpRequest[] = [];
  return {
    client: {
      request: vi.fn(async (request) => {
        requests.push(request);
        return responses.shift()!;
      }),
    },
    requests,
  };
}

describe("loopback sidecar document parser", () => {
  it("probes capabilities then posts one PDF byte buffer without a path", async () => {
    const fixture = http([
      {
        status: 200,
        headers: {},
        bodyText: JSON.stringify({
          protocolVersion: 1,
          parser,
          capabilities: ["page-text", "document-structure", "document-metadata"],
          maxRequestBytes: 1_000,
          maxResponseBytes: 1_000,
        }),
      },
      {
        status: 200,
        headers: {},
        bodyText: JSON.stringify({
          protocolVersion: 1,
          parser,
          document: {
            mediaType: "application/pdf",
            metadata: { title: "Structured paper" },
            blocks: [{ kind: "heading", text: "Methods", headingLevel: 1, locator: { page: 2 } }],
          },
        }),
      },
    ]);

    const sidecar = await probeLoopbackSidecarParser({
      http: fixture.client,
      capabilitiesUrl: "http://127.0.0.1:5001/v1/capabilities",
      parseUrl: "http://127.0.0.1:5001/v1/parse",
    });
    const document = await sidecar.parse(new Uint8Array([0x25, 0x50, 0x44, 0x46]));

    expect(document).toMatchObject({ metadata: { title: "Structured paper" }, blocks: [{ kind: "heading" }] });
    expect(fixture.requests[0]).toMatchObject({
      method: "GET",
      url: "http://127.0.0.1:5001/v1/capabilities",
      headers: { Accept: "application/json" },
    });
    expect(fixture.requests[0]?.body).toBeUndefined();
    expect(fixture.requests[1]).toMatchObject({
      method: "POST",
      url: "http://127.0.0.1:5001/v1/parse",
      headers: { Accept: "application/json", "Content-Type": "application/pdf" },
    });
    expect(new Uint8Array(fixture.requests[1]?.body as ArrayBuffer)).toEqual(new Uint8Array([0x25, 0x50, 0x44, 0x46]));
  });

  it("rejects cross-origin capability/parse URLs and a response from another parser", async () => {
    const fixture = http([]);
    await expect(probeLoopbackSidecarParser({
      http: fixture.client,
      capabilitiesUrl: "http://127.0.0.1:5001/v1/capabilities",
      parseUrl: "http://127.0.0.1:5002/v1/parse",
    })).rejects.toBeInstanceOf(SidecarDocumentParserError);

    const changing = http([
      {
        status: 200,
        headers: {},
        bodyText: JSON.stringify({
          protocolVersion: 1,
          parser,
          capabilities: ["page-text"],
          maxRequestBytes: 1_000,
          maxResponseBytes: 1_000,
        }),
      },
      {
        status: 200,
        headers: {},
        bodyText: JSON.stringify({
          protocolVersion: 1,
          parser: { id: "other", version: "1" },
          document: { mediaType: "application/pdf", blocks: [] },
        }),
      },
    ]);
    const sidecar = await probeLoopbackSidecarParser({
      http: changing.client,
      capabilitiesUrl: "http://127.0.0.1:5001/v1/capabilities",
      parseUrl: "http://127.0.0.1:5001/v1/parse",
    });
    await expect(sidecar.parse(new Uint8Array([1]))).rejects.toMatchObject({ kind: "invalid-response" });
  });
});
