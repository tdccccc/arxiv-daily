# Private storage native capability

This first-party Node-API v8 component exposes only directory and private-file
capabilities. The delivery protocol stays in TypeScript. It uses no V8 API,
third-party native runtime, or user-install-time compilation.

## Developer build and verification

Use an installed C++17 compiler and CMake 3.20+. Supply Node-API headers through
`NODE_INCLUDE_DIR` (and `NODE_LIBRARY` on Windows), or prepare the exact running
Node version's official SDK first:

```sh
node scripts/native-sdk.mjs
node scripts/native-build.mjs
node --test packages/node-runtime/native/tests/storage.test.cjs
npm run build
node scripts/native-package-smoke.mjs
npm run smoke:install
```

SDK downloads are build-time headers/import libraries checked against the exact
version's official SHASUMS256.txt, with cross-origin redirects refused. No package
installation script is used. Linux development can use `/usr/include/node`
instead; `native-build.mjs` also checks the active Node installation's headers.
Windows builds require MSVC and the Windows SDK. The delay-load hook resolves
Node-API from the hosting Node/Electron process, not a separate node.exe.

To exercise default host selection with real native bytes in Vitest, set
`ARXIV_DAILY_TEST_NATIVE=1`. The ordinary suites retain coverage of the safe Linux
compatibility path when native assets are absent.

## Distribution

`native-assets.mjs export` writes only the current native target's binary and
source/digest/API metadata into ignored `prebuilds/`. Development product builds
embed that one target. Release jobs obtain all six x64/arm64 OS targets from the
same workflow run and set `ARXIV_DAILY_NATIVE_RELEASE=1`; missing, corrupt or
source-mismatched targets fail assembly rather than silently producing a partial
release. The plugin's existing three-file install surface is unchanged.

Bundles contain compressed native bytes and their digests. Runtime loads only a
matching bundled target, validates its bytes, and uses a content-addressed,
machine-local code cache. It never downloads executable code. The offline package
smoke disables HTTP and clears PATH for the CLI, and installed-package smoke uses
`--offline --ignore-scripts`, so end users need neither SDK nor compiler.

The CI target set is Linux x64/arm64, macOS Intel/Apple Silicon and Windows
x64/arm64. CMake targets macOS 11+; other runtime/OS compatibility beyond the
actual tested runners is not inferred. Native macOS/Windows and real Electron
loading remain explicit P6 acceptance tasks.

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
