# P3 — audit, verify, and deploy

goal_ref: ../goal.md
status: superseded

## Outcome

The complete scientific-math validation change is independently audited, fully verified, and only the three approved plugin assets are deployed to the isolated test Vault.

## Assumptions

- Repository-wide verification can run without modifying tracked source or Vault data.
- The isolated test Vault location and approved asset deployment procedure are already available from the active initiative context or repository configuration.
- An independent audit can inspect the full initiative diff without changing the accepted P1/P2 contract.

## Approach

Audit the complete change against the goal and non-goals, address only confirmed defects, run repository-wide verification, then deploy exactly `main.js`, `manifest.json`, and `styles.css` to the isolated test Vault and verify those assets without reading or overwriting Vault data.

## Audit stopping criterion

The scanner separates currency, prose, and mathematics with conservative heuristics, so an adversarial audit can always surface another ambiguous edge case. Bound the audit with a severity rubric instead of open-ended edge-case hunting:

- **Blocker (must fix before deploy):** any input that corrupts output bytes or mis-canonicalizes an accepted value — i.e. `ok: true` with a `value` whose meaning changed, or a `\(...\)`/`\[...\]` rewrite that drops or alters content. The load-bearing invariant is that `$...$` spans re-emit identical bytes and only explicitly delimited spans change bytes.
- **Acceptable (record only, do not fix):** conservative over-rejection that sends a single paper to the existing fallback (e.g. an isolated prose `$`, a `$5-$10` range read as math). These do not corrupt reports and stay inside the per-paper fallback boundary.

Close P3 when no Blocker-severity finding survives — not when no edge case exists.

## Tasks

- [ ] Run an independent audit of the full scientific-math validation diff against the goal, scanner contract, retry boundary, and persistence path, judged by the severity rubric above.
- [x] Resolve any confirmed Blocker-severity findings without expanding scope or adding downstream canonicalization; record Acceptable over-rejections rather than fixing them.
- [ ] Add an idempotence property test over accepted values (`canonicalize(canonicalize(x).value).value === canonicalize(x).value` when `ok`) to close the class the journal has rediscovered twice.
- [ ] Record the intentional decision that within this structured-field boundary `\(...\)`/`\[...\]` are treated as math intent, not CommonMark escaped parens, so the audit does not reopen it.
- [ ] Confirm the MULTILINE rejection is intended for these single-line fields; if legitimate newlines occur, narrow it rather than forcing fallback.
- [ ] Run full repository tests, typechecks, build, and diff checks required by the repository.
- [ ] After build, confirm the bundled `main.js` contains the updated math-contract prompt wording, since prompts are imported as modules and asset hashing alone will not catch a stale bundle.
- [ ] Identify the isolated test Vault plugin destination without reading Vault `data.json`, `.cache/`, indexes, notes, or reports.
- [ ] Deploy only `main.js`, `manifest.json`, and `styles.css` to that destination.
- [ ] Verify the three deployed assets match the built artifacts and record the final checkpoint.

## Verification

- Full repository test, typecheck, and build commands pass.
- `git diff --check` passes and the final diff stays within the initiative constraints.
- Idempotence holds for accepted values, and no Blocker-severity finding (byte corruption or mis-canonicalization) survives the audit.
- Independent audit reports no unresolved correctness, safety, retry-classification, or persistence findings.
- The bundled `main.js` includes the new math-contract wording, and exactly the three approved plugin assets are deployed and match the verified build outputs.

## Abort / reshape triggers

- If audit evidence shows the P1 scanner or P2 acceptance boundary is unsound, stop deployment and reshape the relevant implementation before proceeding.
- If the isolated Vault destination cannot be identified without accessing forbidden Vault data, stop and report the blocker rather than guessing.
- If deployment would modify anything beyond the three approved assets, stop before copying files.

## Superseded reason

2026-07-23 L3 steer: destination changed from thick scanner-primary quality + audit/deploy to prompt-primary generation with a thin script net. Remaining verify/deploy work moves to P6 after prompt and thin-fallback phases; do not continue open-ended scanner edge-case hunting under this phase.
