# ADR 0005: Scoped desktop access to an external personal library

Status: Accepted (2026-08-03 Helm P1 research)

Related: ADR 0001 (one TypeScript core, two hosts); ADR 0004 (personal-library-guided discovery).

## Context

ADR 0004 allows a researcher to select a personal literature library inside or outside the Obsidian Vault. Obsidian's public plugin API exposes Vault-relative storage and Vault folder suggesters, but no supported operating-system directory picker or Vault-external storage adapter. Obsidian Desktop does expose an Electron dialog capability, and desktop plugins can use Node filesystem APIs when declared desktop-only.

The existing `StorageAdapter` is intentionally writable and Vault-oriented. Extending it with absolute paths or using it for an external library would grant business code write, remove, and rename operations over researcher-owned files. Importing Node filesystem modules throughout plugin source would also weaken the repository's host boundaries.

We considered:

1. Require the personal library to live inside the Vault.
2. Let plugin business code use arbitrary Node filesystem APIs and absolute paths.
3. Add a separate, root-bound, read-only desktop capability with a narrow package boundary.

The first option contradicts ADR 0004. The second creates an unnecessarily broad mutation and path-escape surface.

## Decision

### 1. Use a dedicated read-only capability

Personal-library access uses a contract separate from `StorageAdapter`. The contract exposes only bounded listing and reading under one already-selected root. It exposes no write, append, mkdir, remove, rename, copy, arbitrary-root, HTTP, or model operation.

Core owns the host-neutral contract and logical relative-path semantics. Core does not receive Electron objects, Node modules, or an unrestricted absolute filesystem path.

### 2. Bind each capability instance to one canonical root

The desktop implementation canonicalizes the selected root with `realpath` when opening the capability. Every listed or read target is resolved from a validated logical relative path and checked against the canonical root.

- Absolute paths, drive-qualified paths, UNC paths, `.` and `..` traversal segments are rejected.
- Symbolic links and equivalent filesystem redirects encountered below the selected root are not followed in the first implementation.
- Directory traversal is bounded by explicit entry, depth, and byte limits and supports cancellation.
- Errors and diagnostics avoid exposing file contents or unnecessary absolute paths.

The boundary protects against accidental traversal and static link escape. It does not claim to resist a privileged local process that races filesystem mutations between validation and read.

### 3. Keep Node access behind a narrow runtime export

The Node implementation lives in `@arxiv-daily/node-runtime` and is exported through a dedicated subpath. The Obsidian plugin may import only that approved subpath; it remains forbidden from importing Node built-ins directly or importing the broad node-runtime root.

This is a deliberate exception to the prior plugin dependency direction: the plugin consumes one audited desktop host capability, not Node business services. `@arxiv-daily/core` remains free of Node dependencies.

### 4. Treat directory selection as an Obsidian desktop host seam

The plugin invokes the currently available Obsidian Desktop Electron dialog through a minimal, runtime-checked interface. The dialog implementation is injected behind a small picker seam so tests and future host changes do not spread Electron globals through settings or domain code.

The plugin remains `isDesktopOnly: true`. If the dialog capability is unavailable, the product reports the feature as unsupported rather than accepting a manually typed unrestricted path as a fallback.

### 5. Separate connection from model authorization

Selecting and locally inventorying a root does not authorize sending its contents to a model. Model-processing consent is a later P1 application/UI concern and remains bound to selected scope, eligible content types, processing depth, and effective endpoint identity as required by ADR 0004.

## Consequences

- Existing Vault storage and daily behavior remain unchanged.
- The plugin gains a narrowly allowlisted dependency on one node-runtime subpath; repository boundary checks must enforce that exact import.
- The selected absolute path remains host-local permission data and must not leak into portable research records or ordinary logs.
- P1 requires adversarial tests for traversal, sibling-prefix confusion, symlinks, limits, cancellation, and the absence of mutation methods.
- Accessing files outside the Vault must be disclosed in user-facing plugin documentation before the feature is shipped.
- If Obsidian removes or changes its desktop Electron dialog bridge, only the picker seam should require replacement.

## Non-decisions

- Catalog schema, paper identification, metadata providers, or abstract resolution.
- Settings and UI representation of processing consent.
- Full-text parsing, RAG, Agent execution, or daily discovery semantics.
- Supporting multiple library roots or mobile hosts.
