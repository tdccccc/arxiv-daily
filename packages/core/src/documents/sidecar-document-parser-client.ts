import type { HttpClient } from "../core/adapters";
import type { DocumentParser, ParseDocumentOptions, ParsedDocument } from "./parsed-document";
import {
  MAX_SIDECAR_RESPONSE_BYTES,
  decodeSidecarParseResponse,
  decodeSidecarParserCapabilities,
  requireLoopbackSidecarUrl,
  type SidecarParserCapabilities,
} from "./sidecar-document-parser";

const DEFAULT_TIMEOUT_MS = 15_000;
const MAX_CAPABILITY_RESPONSE_BYTES = 64 * 1024;

export type SidecarDocumentParserErrorKind =
  | "invalid-endpoint"
  | "unavailable"
  | "input-too-large"
  | "invalid-response";

export class SidecarDocumentParserError extends Error {
  constructor(
    readonly kind: SidecarDocumentParserErrorKind,
    message: string,
    options: ErrorOptions = {},
  ) {
    super(message, options);
    this.name = "SidecarDocumentParserError";
  }
}

export interface LoopbackSidecarParserProbeInput {
  readonly http: HttpClient;
  readonly capabilitiesUrl: string;
  readonly parseUrl: string;
  readonly timeoutMs?: number;
}

/** Probe an explicit local sidecar without sending a PDF or any library path. */
export async function probeLoopbackSidecarParser(
  input: LoopbackSidecarParserProbeInput,
): Promise<LoopbackSidecarDocumentParser> {
  let capabilitiesUrl: URL;
  let parseUrl: URL;
  try {
    capabilitiesUrl = requireLoopbackSidecarUrl(input.capabilitiesUrl);
    parseUrl = requireLoopbackSidecarUrl(input.parseUrl);
    if (capabilitiesUrl.origin !== parseUrl.origin) {
      throw new TypeError("sidecar capability and parse endpoints must share one loopback origin");
    }
  } catch (caught) {
    throw new SidecarDocumentParserError("invalid-endpoint", "sidecar endpoint configuration is invalid", { cause: caught });
  }
  const timeoutMs = timeout(input.timeoutMs);
  let response: Awaited<ReturnType<HttpClient["request"]>>;
  try {
    response = await input.http.request({
      url: capabilitiesUrl.toString(),
      method: "GET",
      headers: { Accept: "application/json" },
      timeoutMs,
    });
  } catch (caught) {
    throw new SidecarDocumentParserError("unavailable", "sidecar capability probe failed", { cause: caught });
  }
  if (response.status !== 200) {
    throw new SidecarDocumentParserError("unavailable", `sidecar capability probe returned HTTP ${response.status}`);
  }
  const capabilities = decodeCapabilityBody(response.bodyText);
  return new LoopbackSidecarDocumentParser(input.http, parseUrl, capabilities, timeoutMs);
}

export class LoopbackSidecarDocumentParser implements DocumentParser {
  readonly capabilities: readonly DocumentParser["capabilities"][number][];
  readonly provenance: DocumentParser["provenance"];

  constructor(
    private readonly http: HttpClient,
    private readonly parseUrl: URL,
    private readonly sidecar: SidecarParserCapabilities,
    private readonly timeoutMs: number,
  ) {
    this.capabilities = [...sidecar.capabilities];
    this.provenance = { ...sidecar.parser };
  }

  async parse(bytes: Uint8Array, options?: ParseDocumentOptions): Promise<ParsedDocument> {
    options?.signal?.throwIfAborted();
    if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0 || bytes.byteLength > this.sidecar.maxRequestBytes) {
      throw new SidecarDocumentParserError("input-too-large", "PDF bytes exceed the sidecar request limit");
    }
    let response: Awaited<ReturnType<HttpClient["request"]>>;
    try {
      response = await this.http.request({
        url: this.parseUrl.toString(),
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/pdf",
        },
        body: bytes.slice().buffer,
        timeoutMs: this.timeoutMs,
        signal: options?.signal,
      });
    } catch (caught) {
      options?.signal?.throwIfAborted();
      throw new SidecarDocumentParserError("unavailable", "sidecar parse request failed", { cause: caught });
    }
    if (response.status !== 200) {
      throw new SidecarDocumentParserError("unavailable", `sidecar parse request returned HTTP ${response.status}`);
    }
    const parsed = decodeParseBody(response.bodyText, this.sidecar.maxResponseBytes);
    if (parsed.parser.id !== this.provenance.id || parsed.parser.version !== this.provenance.version) {
      throw new SidecarDocumentParserError("invalid-response", "sidecar parser provenance changed after capability probe");
    }
    try {
      assertCapabilities(parsed.document, this.capabilities);
    } catch (caught) {
      throw new SidecarDocumentParserError("invalid-response", "sidecar response exceeds its declared capabilities", { cause: caught });
    }
    return parsed.document;
  }
}

function decodeCapabilityBody(body: string): SidecarParserCapabilities {
  const value = decodeJson(body, MAX_CAPABILITY_RESPONSE_BYTES, "sidecar capability response");
  try {
    return decodeSidecarParserCapabilities(value);
  } catch (caught) {
    throw new SidecarDocumentParserError("invalid-response", "sidecar capability response is invalid", { cause: caught });
  }
}

function decodeParseBody(body: string, maxBytes: number) {
  const value = decodeJson(body, Math.min(maxBytes, MAX_SIDECAR_RESPONSE_BYTES), "sidecar parse response");
  try {
    return decodeSidecarParseResponse(value);
  } catch (caught) {
    throw new SidecarDocumentParserError("invalid-response", "sidecar parse response is invalid", { cause: caught });
  }
}

function decodeJson(body: string, maxBytes: number, label: string): unknown {
  if (typeof body !== "string" || new TextEncoder().encode(body).byteLength > maxBytes) {
    throw new SidecarDocumentParserError("invalid-response", `${label} exceeds the allowed size`);
  }
  try {
    return JSON.parse(body) as unknown;
  } catch (caught) {
    throw new SidecarDocumentParserError("invalid-response", `${label} is not valid JSON`, { cause: caught });
  }
}

function timeout(value: number | undefined): number {
  if (value === undefined) return DEFAULT_TIMEOUT_MS;
  if (!Number.isSafeInteger(value) || value < 1 || value > 120_000) {
    throw new SidecarDocumentParserError("invalid-endpoint", "sidecar timeout must be between 1 and 120000 milliseconds");
  }
  return value;
}

function assertCapabilities(
  document: ParsedDocument,
  capabilities: readonly DocumentParser["capabilities"][number][],
): void {
  const declared = new Set(capabilities);
  if (document.metadata && !declared.has("document-metadata")) {
    throw new Error("document metadata was not declared");
  }
  for (const block of document.blocks) {
    if (block.kind !== "page" && !declared.has("document-structure")) {
      throw new Error("structured blocks were not declared");
    }
    if (block.layout && !declared.has("text-layout")) {
      throw new Error("text layout was not declared");
    }
  }
}
