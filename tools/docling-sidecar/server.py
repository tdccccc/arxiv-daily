from __future__ import annotations

from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
from io import BytesIO
import json
from socket import AF_INET6
from typing import Protocol, cast
from urllib.parse import urlsplit


PROTOCOL_VERSION = 1
CAPABILITIES_PATH = "/v1/capabilities"
PARSE_PATH = "/v1/parse"
MAX_REQUEST_BYTES = 25 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_BLOCK_TEXT_LENGTH = 256 * 1024
MAX_BLOCKS = 100_000
LOOPBACK_HOSTS = ("127.0.0.1", "::1")


class PdfByteParser(Protocol):
    provenance: dict[str, str]
    capabilities: list[str]
    max_request_bytes: int
    max_response_bytes: int

    def parse_pdf_bytes(self, payload: bytes) -> dict[str, object]: ...


class SidecarHTTPServer(HTTPServer):
    def __init__(self, address: tuple[str, int], parser: PdfByteParser):
        super().__init__(address, SidecarRequestHandler)
        self.parser = parser

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(30)
        return request, client_address


class IPv6SidecarHTTPServer(SidecarHTTPServer):
    address_family = AF_INET6


class SidecarRequestHandler(BaseHTTPRequestHandler):
    server: SidecarHTTPServer

    def do_GET(self) -> None:
        if not self._route_is(CAPABILITIES_PATH):
            self._error(404, "not-found")
            return
        parser = self.server.parser
        self._json(200, {
            "protocolVersion": PROTOCOL_VERSION,
            "parser": parser.provenance,
            "capabilities": parser.capabilities,
            "maxRequestBytes": parser.max_request_bytes,
            "maxResponseBytes": parser.max_response_bytes,
        })

    def do_POST(self) -> None:
        if not self._route_is(PARSE_PATH):
            self._error(404, "not-found")
            return
        if self.headers.get("Content-Type", "").lower() != "application/pdf":
            self._error(415, "unsupported-media-type")
            return
        if self.headers.get("Transfer-Encoding"):
            self._error(400, "transfer-encoding-not-supported")
            return
        body = self._read_pdf_body()
        if body is None:
            return
        parser = self.server.parser
        try:
            document = parser.parse_pdf_bytes(body)
        except Exception:
            # Conversion details can include provider internals; the protocol
            # intentionally returns only a typed HTTP failure to its client.
            self._error(422, "parse-failed")
            return
        payload = {
            "protocolVersion": PROTOCOL_VERSION,
            "parser": parser.provenance,
            "document": document,
        }
        serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(serialized) > parser.max_response_bytes:
            self._error(413, "response-too-large")
            return
        self._bytes(200, serialized)

    def _route_is(self, expected_path: str) -> bool:
        parsed = urlsplit(self.path)
        return parsed.path == expected_path and not parsed.query and not parsed.fragment

    def _read_pdf_body(self) -> bytes | None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdecimal():
            self._error(400, "content-length-required")
            return None
        length = int(raw_length)
        if length < 1:
            self._error(400, "empty-pdf")
            return None
        if length > self.server.parser.max_request_bytes:
            self._error(413, "request-too-large")
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._error(400, "incomplete-body")
            return None
        return body

    def _error(self, status: int, code: str) -> None:
        self._json(status, {"error": code})

    def _json(self, status: int, payload: dict[str, object]) -> None:
        self._bytes(status, json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))

    def _bytes(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        # The server never logs request paths or bodies.
        return


class DoclingByteParser:
    capabilities = ["page-text", "document-structure", "document-metadata"]
    max_request_bytes = MAX_REQUEST_BYTES
    max_response_bytes = MAX_RESPONSE_BYTES

    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter

        self.provenance = {"id": "docling", "version": version("docling")}
        self._converter = DocumentConverter()

    def parse_pdf_bytes(self, payload: bytes) -> dict[str, object]:
        from docling.datamodel.base_models import DocumentStream

        result = self._converter.convert(
            DocumentStream(name="document.pdf", stream=BytesIO(payload)),
            max_file_size=self.max_request_bytes,
        )
        return document_to_protocol(result.document)


def document_to_protocol(document: object) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    title: str | None = None
    iterate_items = getattr(document, "iterate_items", None)
    if not callable(iterate_items):
        raise ValueError("Docling document does not expose reading-order items")
    for item, _depth in iterate_items():
        label = str(getattr(item, "label", "text"))
        if label in {"picture", "chart"}:
            # Figure captions are body text items and retain their own locator.
            continue
        text = table_text(item) if label == "table" else getattr(item, "text", None)
        page = first_page(item)
        if not isinstance(text, str) or not text or page is None:
            continue
        text_length = utf16_length(text)
        if text_length > MAX_BLOCK_TEXT_LENGTH:
            raise ValueError("text block exceeds sidecar response limit")
        kind = text_kind(label)
        if kind == "heading" and title is None and title_candidate(text):
            title = text
        block: dict[str, object] = {
            "kind": kind,
            "text": text,
            "locator": {
                "page": page,
                "block": len(blocks),
                "charStart": 0,
                "charEnd": text_length,
            },
        }
        if kind == "heading":
            block["headingLevel"] = heading_level(text, getattr(item, "level", None))
        blocks.append(block)
        if len(blocks) > MAX_BLOCKS:
            raise ValueError("document exceeds sidecar block limit")

    response: dict[str, object] = {
        "mediaType": "application/pdf",
        "blocks": blocks,
    }
    if title:
        response["metadata"] = {"title": title}
    return response


def first_page(item: object) -> int | None:
    values = [getattr(provenance, "page_no", None) for provenance in getattr(item, "prov", [])]
    pages = [value for value in values if isinstance(value, int) and 1 <= value <= 100_000]
    return min(pages) if pages else None


def text_kind(label: str) -> str:
    if label == "section_header":
        return "heading"
    if label == "caption":
        return "caption"
    if label == "formula":
        return "equation"
    if label == "list_item":
        return "list-item"
    if label == "code":
        return "code"
    if label == "table":
        return "table"
    return "paragraph"


def title_candidate(text: str) -> bool:
    return text.strip().lower() not in {
        "abstract",
        "keywords",
        "ccs concepts",
        "acm reference format",
    }


def heading_level(text: str, documented_level: object) -> int:
    if isinstance(documented_level, int) and 1 <= documented_level <= 16:
        return documented_level
    prefix = text.strip().split(maxsplit=1)[0].rstrip(".")
    if prefix and all(piece.isdigit() for piece in prefix.split(".")):
        return min(16, len(prefix.split(".")))
    return 1


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def table_text(table: object) -> str:
    cells = getattr(getattr(table, "data", None), "table_cells", [])
    rows: dict[int, dict[int, str]] = {}
    for cell in cells:
        row = getattr(cell, "start_row_offset_idx", None)
        column = getattr(cell, "start_col_offset_idx", None)
        text = getattr(cell, "text", None)
        if not isinstance(row, int) or not isinstance(column, int) or not isinstance(text, str):
            continue
        rows.setdefault(row, {})[column] = text.strip()
    if not rows:
        return ""
    width = max(max(row) for row in rows.values()) + 1
    return "\n".join(
        " | ".join(rows[row].get(column, "") for column in range(width))
        for row in sorted(rows)
    )


def parse_args() -> tuple[str, int]:
    parser = ArgumentParser(description="Docling loopback PDF-byte sidecar")
    parser.add_argument("--host", choices=LOOPBACK_HOSTS, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("port must be between 1 and 65535")
    return cast(str, args.host), cast(int, args.port)


def main() -> None:
    host, port = parse_args()
    parser = DoclingByteParser()
    server_class = IPv6SidecarHTTPServer if host == "::1" else SidecarHTTPServer
    server = server_class((host, port), parser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
