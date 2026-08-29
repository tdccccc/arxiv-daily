# Docling Loopback Sidecar

This optional local process implements the arXiv Daily sidecar protocol at
`http://127.0.0.1:5001`. It accepts only one `application/pdf` byte body per
request. It has no route for a library root, a file path, a directory listing,
or an inventory.

Install it into an isolated Python environment. For CPU-only hosts, pin the
CPU PyTorch wheels so the installer does not download CUDA runtimes:

```sh
python3 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  docling torch==2.7.1+cpu torchvision==0.22.1+cpu
```

Run the process on literal loopback only. The cache paths below are examples;
keep them outside a personal library and outside a shared Vault when possible.

```sh
DOCLING_CACHE_DIR=/tmp/docling-cache \
HF_HOME=/tmp/docling-hf \
.venv/bin/python server.py --host 127.0.0.1 --port 5001
```

The Plugin remains on PDF.js unless the user explicitly enables the local
parser sidecar and configures these same-origin loopback endpoints:

```text
http://127.0.0.1:5001/v1/capabilities
http://127.0.0.1:5001/v1/parse
```

Use `python3 -m unittest tests/test_server.py` to run the protocol and
reading-order mapping tests. The tests bind a temporary loopback port.
