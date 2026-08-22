## 2026-08-23 — L1 adjust

- evidence: PR #20 Node 20.11.0 compatibility job failed during `smoke:build`; Vite 7.3.6 requires Node 20.19.0 and the bundled CLI hit missing `util.styleText`. Node 20.19.0, Node 22.17.0, Relay, Companion, CodeQL, and root verification then passed.
- change: revised the P2 compatibility matrix and phase record from Node 20.11.0 to Node 20.19.0; kept the implementation and tests, and recorded the separate public engine metadata follow-up.
- disposition: keep dependency remediation, audit gates, CodeQL, Dependabot, install smoke, and DOM renderer regression tests; discard the unsupported 20.11 matrix entry.
- next: leave PR #20 ready for maintainer merge; the first follow-up after merge is a release metadata update that aligns package engine declarations with the Node 20.19 floor.
