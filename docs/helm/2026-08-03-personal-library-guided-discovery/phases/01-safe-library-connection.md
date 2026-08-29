# P1 — safe library connection

goal_ref: ../goal.md
updated: 2026-08-03

## Outcome

On Obsidian desktop, a researcher can select one Vault-internal or external library root, obtain a local read-only inventory preview, and separately review, grant, invalidate, or revoke a scope- and endpoint-bound metadata/abstract processing authorization through a host capability that cannot escape or mutate the selected root; the existing daily workflow remains unchanged.

## Assumptions

- Obsidian's supported desktop runtime can provide or safely host a directory-selection and scoped filesystem capability without importing Node built-ins into Core.
- One library root is sufficient for the first initiative; multiple roots and source-manager presets are not needed to prove the product path.
- P1 may classify inventory entries by extension and basic paper-likeness, but does not identify logical papers, resolve metadata, call an LLM, or persist a library catalog.
- Consent metadata may live in the plugin settings store during P1 because it is product authorization, not the authoritative research record.
- Mobile may report the feature as unsupported; mobile library access is not required.

## Approach

First prove the narrow host boundary: investigate the supported Obsidian desktop selection/runtime APIs, then introduce a dedicated scoped read-only library capability rather than widening the existing writable Vault `StorageAdapter`. Keep absolute host paths and permission details out of Core domain objects and logs where possible. Add a small plugin flow that can list a bounded local inventory after directory selection, then separately discloses root, eligible content types, processing depth, and effective endpoint identity before model-processing authorization. Verify containment, symlink behavior, revocation, endpoint invalidation, cancellation, and unchanged daily behavior before accepting the phase.

## Tasks

- [x] Verify the supported Obsidian desktop directory-selection and external-filesystem mechanisms, plugin review/bundle constraints, mobile behavior, and cross-platform path risks; record the selected host boundary and rejected alternatives in this phase or an ADR if the decision meets ADR criteria.
- [x] Define a minimal scoped, read-only library-source contract and threat model covering root containment, path representation, symlinks, listing/stat/read limits, cancellation, unsupported hosts, and redacted diagnostics without widening `StorageAdapter`.
- [x] Implement the desktop host capability and directory-selection composition so only the selected root can be listed/read and no write, remove, rename, arbitrary-root, or escape operation is exposed.
- [x] Add plugin settings/status entry and a focused connection flow that shows selected scope, eligible paper-file types, metadata/abstract depth, and configured endpoint identity before granting authorization.
- [x] Persist authorization identity, support explicit revocation, and invalidate authorization when the selected root, filesystem identity, eligible content, processing depth, or effective endpoint changes; keep local inventory separate from model processing.
- [x] Produce a bounded, cancellable inventory preview for Vault-internal and external roots, visibly separating eligible candidates, ignored files, unsupported host, permission errors, and unsafe paths without resolving paper metadata or invoking the LLM.
- [x] Add adversarial and regression coverage for containment, symlink/path traversal, read-only surface, revocation/invalidation, redaction, cancellation, mobile/unsupported behavior, and unchanged manual daily runs; run the focused suites and repository quality checks required by the touched workspaces.

## Verification

- Automated tests demonstrate that `..`, absolute-path substitution, sibling roots, and escaping symlinks cannot be listed or read through the scoped capability.
- Contract/type inspection confirms that the library capability exposes no write, remove, rename, or arbitrary HTTP/LLM operation.
- Plugin and runtime tests demonstrate that Vault-internal and external absolute roots reach the same bounded local inventory-preview path without requiring model-processing authorization.
- Changing the selected root, filesystem identity, configured effective endpoint, eligible file types, or processing depth invalidates authorization; revocation prevents later model processing while local preview remains available.
- Logs and persisted consent state do not contain API keys, raw endpoint credentials, file contents, or unredacted external paths outside the minimum host-local permission record.
- Instrumented tests confirm P1 performs no LLM call and creates no library catalog or daily-report mutation.
- Existing manual settings, pipeline, scheduling, Dashboard, and daily rendering tests remain green; workspace typecheck, lint, boundary checks, and the relevant full repository suite pass.

## Abort / reshape triggers

- If supported Obsidian desktop APIs cannot safely select and read a Vault-external root under plugin distribution constraints, stop and reshape P1 rather than importing unrestricted Node filesystem access into plugin business code.
- If symlink/realpath containment cannot be made cross-platform and testable, restrict the capability to a safer path policy and surface the limitation before proceeding.
- If endpoint-bound consent cannot be represented without leaking sensitive endpoint or path data into synced Vault artifacts, separate host-local authorization from portable research data before implementing catalog work.
- If the capability requires widening Core or the existing `StorageAdapter` to arbitrary writable filesystem access, reject that approach and redesign the host boundary.
- If P1 starts resolving metadata, generating directions, or changing daily output, split that work into P2 or later instead of expanding the phase outcome.
