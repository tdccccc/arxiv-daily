# Private storage native capability

This first-party Node-API v8 component exposes only directory and private-file
capabilities. The delivery protocol stays in TypeScript. It uses no V8 API,
third-party native runtime, or user-install-time compilation.

## Local feasibility build

Use an installed C++17 compiler, CMake 3.20+, and Node-API headers:

```sh
cmake -S packages/node-runtime/native -B packages/node-runtime/native/build \
  -DCMAKE_BUILD_TYPE=Release -DNODE_INCLUDE_DIR=/usr/include/node
cmake --build packages/node-runtime/native/build --config Release
node --test packages/node-runtime/native/tests/storage.test.cjs
```

On Windows, also supply `NODE_LIBRARY` pointing to the matching official
`node.lib`; MSVC and the Windows SDK are required. The delay-load hook binds to
the hosting Node/Electron process, not a separately installed node.exe. Build
inputs and package integration are owned by Helm P5; end users do not run CMake.

## Safety boundary

- Directory names are validated in native code even if callers already did so.
- POSIX children are opened relative to held directory descriptors, without
  following symlinks. Namespace checks compare current and held identities.
- Windows pins every traversed directory without delete sharing, rejects reparse
  points, and requires a local filesystem with persistent ACLs. New files get a
  protected current-user DACL at creation, not after content has been written.
- A created file is private before the caller writes data. Existing files are
  read-only capabilities; `restrict()` tightens their permissions without writing
  content. A writable capability comes only from exclusive creation.
- The caller checks `assertCurrent()` around asynchronous work and immediately
  before an external action. Cleanup can unlink through the original capability
  even when the logical POSIX path has moved.
- `close()` is idempotent; all other operations on a closed capability reject.
  Native finalizers close forgotten handles, but correctness does not rely on GC.
- Flushes cover file data and POSIX directory updates; Windows atomic replacement
  requests write-through. This is not a network-filesystem/distributed protocol.

The implementation is exercised on Linux during P5. Actual Windows/macOS and
Electron runtime results remain P6 obligations, not inferred successes.
