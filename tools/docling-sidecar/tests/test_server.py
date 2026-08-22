from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
import sys
from threading import Thread
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import PROTOCOL_VERSION, SidecarHTTPServer, document_to_protocol  # noqa: E402


class FakeParser:
    provenance = {"id": "docling", "version": "fixture"}
    capabilities = ["page-text", "document-structure", "document-metadata"]
    max_request_bytes = 32
    max_response_bytes = 4_096

    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def parse_pdf_bytes(self, payload: bytes) -> dict[str, object]:
        self.payloads.append(payload)
        return {
            "mediaType": "application/pdf",
            "metadata": {"title": "Structured fixture"},
            "blocks": [{
                "kind": "heading",
                "text": "Methods",
                "headingLevel": 1,
                "locator": {"page": 2, "block": 0, "charStart": 0, "charEnd": 7},
            }],
        }


class SidecarServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FakeParser()
        self.server = SidecarHTTPServer(("127.0.0.1", 0), self.parser)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=2)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, payload

    def test_capabilities_are_loopback_protocol_metadata_without_a_request_body(self) -> None:
        status, payload = self.request("GET", "/v1/capabilities")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload), {
            "protocolVersion": PROTOCOL_VERSION,
            "parser": self.parser.provenance,
            "capabilities": self.parser.capabilities,
            "maxRequestBytes": self.parser.max_request_bytes,
            "maxResponseBytes": self.parser.max_response_bytes,
        })
        self.assertEqual(self.parser.payloads, [])

    def test_docling_body_items_keep_reading_order_and_page_locators(self) -> None:
        provenance = [SimpleNamespace(page_no=3)]
        document = SimpleNamespace(iterate_items=lambda: iter([
            (SimpleNamespace(
                label="section_header",
                text="3 Methods",
                level=2,
                prov=provenance,
            ), 1),
            (SimpleNamespace(
                label="table",
                data=SimpleNamespace(table_cells=[
                    SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=0, text="Method"),
                    SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=1, text="Score"),
                    SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=0, text="Docling"),
                    SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=1, text="1.0"),
                ]),
                prov=provenance,
            ), 1),
            (SimpleNamespace(
                label="caption",
                text="Table 1: Fixture results",
                prov=provenance,
            ), 1),
        ]))

        parsed = document_to_protocol(document)

        self.assertEqual(parsed, {
            "mediaType": "application/pdf",
            "metadata": {"title": "3 Methods"},
            "blocks": [
                {
                    "kind": "heading",
                    "text": "3 Methods",
                    "headingLevel": 2,
                    "locator": {"page": 3, "block": 0, "charStart": 0, "charEnd": 9},
                },
                {
                    "kind": "table",
                    "text": "Method | Score\nDocling | 1.0",
                    "locator": {"page": 3, "block": 1, "charStart": 0, "charEnd": 28},
                },
                {
                    "kind": "caption",
                    "text": "Table 1: Fixture results",
                    "locator": {"page": 3, "block": 2, "charStart": 0, "charEnd": 24},
                },
            ],
        })

    def test_docling_locator_offsets_use_utf16_code_units(self) -> None:
        document = SimpleNamespace(iterate_items=lambda: iter([(
            SimpleNamespace(
                label="caption",
                text="Note 😀",
                prov=[SimpleNamespace(page_no=1)],
            ),
            1,
        )]))

        parsed = document_to_protocol(document)

        self.assertEqual(parsed["blocks"][0]["locator"]["charEnd"], 7)

    def test_parse_accepts_one_pdf_byte_body_and_never_receives_a_path(self) -> None:
        status, payload = self.request(
            "POST",
            "/v1/parse",
            body=b"%PDF-fixture",
            headers={"Content-Type": "application/pdf"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(self.parser.payloads, [b"%PDF-fixture"])
        self.assertEqual(json.loads(payload), {
            "protocolVersion": PROTOCOL_VERSION,
            "parser": self.parser.provenance,
            "document": {
                "mediaType": "application/pdf",
                "metadata": {"title": "Structured fixture"},
                "blocks": [{
                    "kind": "heading",
                    "text": "Methods",
                    "headingLevel": 1,
                    "locator": {"page": 2, "block": 0, "charStart": 0, "charEnd": 7},
                }],
            },
        })

    def test_rejects_non_protocol_routes_queries_content_types_and_oversized_bodies(self) -> None:
        for method, path, body, headers in [
            ("GET", "/v1/capabilities?path=/private/library", None, {}),
            ("POST", "/v1/parse", b"not-pdf", {"Content-Type": "text/plain"}),
            ("POST", "/v1/parse", b"x" * 33, {"Content-Type": "application/pdf"}),
        ]:
            with self.subTest(path=path, headers=headers):
                status, _ = self.request(method, path, body=body, headers=headers)
                self.assertIn(status, {400, 404, 413, 415})
        self.assertEqual(self.parser.payloads, [])
