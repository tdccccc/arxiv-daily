# ADR 0009: Bundled native storage for cross-platform automatic delivery

Status: Proposed implementation (native-component direction accepted by the user on 2026-09-29)

Related: ADR 0001 (TypeScript core and hosts); ADR 0003 (two products and portable Vault data); Helm 2026-09-28-runtime-reliability-fixes P5/P6.

## Context

Automatic email must durably claim a date/recipient before contacting its provider, and must not send after its claim directory has been replaced. Existing Linux hosts use descriptor-relative filesystem operations; ordinary exclusive path creation alone cannot preserve the parent-directory boundary. Windows additionally needs an ACL: Node chmod does not implement Unix owner/group/others privacy there.

The user explicitly accepted adding a platform-specific support component rather than relaxing these guarantees. The current plugin release installs only main.js, manifest.json and styles.css, while the CLI is a self-contained bundle. Making an end user install a compiler, download executable code on first send, or move delivery state outside portable Vault data would add different costs.

## Decision

1. Keep all claim/decision/result semantics and provider logic in TypeScript Core. Add only a small, first-party Node-API v8 storage backend in node-runtime. Do not use V8 or Electron-specific APIs.
2. POSIX operations are relative to opened directory descriptors. Windows pins each traversed directory against rename/delete while open, rejects reparse points, and creates private files with a protected current-user DACL. A synchronous namespace check remains immediately before the HTTP invocation.
3. Separate namespace/file capabilities from the TypeScript create/replace/recover orchestration, so existing deterministic race and crash tests can observe actual native handles. Closed, malformed, unavailable and unsupported capabilities fail closed.
4. Build the backend in controlled native CI jobs. Embed the supported platform/architecture binaries with content digests in product bundles and extract only the matching binary to machine-local code storage. No runtime download or user build tool is required. Development builds may contain only their native platform; release builds must verify the full declared platform set and source identity.
5. Retain the existing Linux backend as a compatibility fallback only when native assets are absent, not when a supplied native asset fails integrity or compatibility validation. Never introduce a less-protected path-based fallback for automatic delivery.

## Consequences

- Packaging grows and CI must produce/test multiple OS/architecture assets. Node-API avoids per-Electron-version binaries, but actual Electron loading remains an acceptance obligation.
- Native source and loader errors are part of the trusted computing base; neither a generic FFI surface nor arbitrary path loading is exposed to Core.
- Delivery state, claims, and export/import remain in the Vault with their existing schema. Extracted code is machine-local, replaceable, and outside research data.
- The native I/O surface is synchronous and limited to small bookkeeping files; it does not replace bulk PDF/Markdown storage.
- Linux evidence cannot close macOS/Windows acceptance. P6 retains real platform/runtime verification, and this ADR becomes accepted only after the implementation's recorded feasibility checkpoint.
