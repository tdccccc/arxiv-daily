# Optional Node library runtimes

These lockfiles are build resources for `packages/node-runtime/src/library-runtime.ts`, not additional user products or root npm workspaces. The ordinary CLI embeds the manifests and lockfiles; `arxiv-daily library prepare` materializes them under the configured product cache and runs `npm ci --ignore-scripts` there.

- `pdf/` pins PDF.js 5.4.624, whose Node requirement includes the product's Node 20.19 baseline. Its worker, CMaps and standard-font assets remain inside the installed package.
- `embedding/` pins Transformers 4.2.0 with its real CPU ONNX dependency. This keeps it separate from the Obsidian workspace's browser/WASM overrides. Remote embedding does not prepare this component.
- The local model remains `Xenova/multilingual-e5-small`, q8, 384 dimensions, mean pooling and normalized vectors. Model weights are downloaded on first use into `<cache_dir>/models/`; runtime packages live in `<cache_dir>/runtimes/`.

To change a dependency, update the corresponding manifest and regenerate its lockfile from that directory with `npm install --package-lock-only --ignore-scripts --workspaces=false`. Update the versioned runtime path and run real PDF/model smoke checks in supported Node versions. Do not copy root overrides into these isolated runtime projects.

This implementation was exercised with real PDF parsing and CPU inference on Linux Node 22 and, offline from the same cache, Node 20.19. Native dependency behavior on Windows and macOS still needs platform smoke checks before a general release.
