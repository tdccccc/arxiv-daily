# Journal

## 2026-10-02T14:56:36+08:00 — Begin DSH integration

The user chose DSH as the first desktop plugin foundation, with later Claude Mod adaptation. P4 paper workspace remains done. The new work preserves its implementation and data, adds the DSH host boundary, and keeps the existing Claude Code plugin usable. Installed DSH is 0.1.7-alpha.1. Source research confirms Connection RPC intercept/call and the composer dock / right-sidebar Browser interfaces. All acceptance work will use isolated profiles; no Computer Use per user instruction.


## 2026-10-02T15:40:01+08:00 — P1 accepted and initiative closed

Delivered the first DSH integration on the existing isolated worktree. The accepted workbench/P4 remains intact. CLI embedding is an explicit, exact-loopback option; DSH startup runs through its authenticated Connection RPC and opens the existing Browser tab from a native composer button. One managed child service serves the plugin's sessions and is stopped on unload. Business data and generation stay owned by the original CLI/core; the DSH-specific adapter is separate from the reusable process boundary.

All five success criteria are accepted through the recorded process, registry, React DOM, packaging and actual installed DSH Host checks. The no-Computer-Use exception was planned before implementation and is explicit in the README: no Electron click-through or visual acceptance, no claim for other DSH versions or other OS artifacts. No remaining feature work is required for this first local plugin foundation. GUI first-run setup and Claude Mod are outside this phase.

Final verification: 13 DSH tests (including actual 0.1.7-alpha.1 Host, none skipped), 209 CLI tests, seven existing integration checks, seventeen governance checks, CLI typecheck and boundaries. The experimental Host and workbench were stopped; the user's live profile/config/Vault were untouched. A linux/x64 local install archive is ready, unpublished. Next product validation is installing that package through Desktop's plugin UI and reviewing its physical placement.

## 2026-10-02T19:45:14+08:00 — Revisit P1 via P2 after an activation collision

The user's screenshot shows DSH refusing the plugin because /api already has an interceptor. Installed Connection source confirms that interceptor is a single slot; installed API Gateway claims it. P1's isolated smoke test accepted our response without asserting Gateway health, so it could miss the opposite activation order failing the built-in gateway. That integration acceptance is superseded. Retain the process controller, UI, workbench, data and security boundary; replace only the transport registration and strengthen tests. P2 is the only active phase.

## 2026-10-02T20:03:50+08:00 — P2 accepted, activation fix packaged as 0.1.1

The fault was our use of Connection's single /api interceptor, which belongs to the built-in Gateway. Exact authenticated Fetch-route registration fixes both load orders and keeps the existing UI/RPC contract. A separate channel was evaluated and rejected after the installed Host exposed a webServer injection failure. No host patch or permission weakening was introduced.

Sixteen DSH checks pass, including the screenshot reproduction now Green, explicit Gateway/plugin activation state, actual Plugin Manager disable/re-enable, persisted marks and lifecycle cleanup. Inventory and boundaries pass. All original success criteria are re-accepted with this stronger coexistence evidence; P2 and the initiative are done. New local 0.1.1 archive is ready for the user's installation. Their live DSH profile and research files were not changed.

## 2026-10-02T21:33:39+08:00 — Add P3 for fixed literature navigation

The user confirmed the composer button exists inside a conversation but requested the dsh-context pattern: left footer above Settings and a right-Sidebar guide entry. Verified dsh-context client/index.ts, components/overviewButton.tsx, components/overviewPanel.tsx and sidebar.ts. P2 remains accepted. P3 adds root navigation and a dedicated tab, sharing the existing workbench through an isolated frame; the composer-only entry is removed.
