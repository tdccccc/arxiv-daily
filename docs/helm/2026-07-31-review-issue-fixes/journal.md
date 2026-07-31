## 2026-07-31 — L1 adjust

- evidence: deleting the `dateWindowNote` option from
  `MarkdownWriter.writeDaily` broke markdown-writer.test.ts:351
  ("writeDaily includes submitted-date fallback notes"); initial grep had
  excluded tests, so the consumer was hidden
- change: reverted markdown-writer.ts to original; T1 now only removes the
  dead `const dateWindowNote = undefined` binding in pipeline.ts and the
  `dateWindowNote` property from the writeDaily call
- disposition: writer option + render hook + test kept (tested dormant
  capability; consider wiring it up or removing it in a later phase)
- next: re-run checks, commit P1

## 2026-07-31 — note

- evidence: P1 completed (PR #3, commits 60bccfe..a125cf6, all checks green)
- change: phase 01-cleanup → done; P2 → blocked
- disposition: P2 waits for the bug-fix branch to merge into main (conflict
  surface: pipeline.ts, view.ts); P3 waits for a product decision
- next: when the bug-fix PR merges, rebase P2 worktree and start view.ts split

## 2026-07-31 — note (P2 dependencies + P3 evidence)

- evidence: user named the bug-fix branches — `fix/arxiv-failure-recovery`
  and `fix/arxiv-request-resilience`; also closed the two stale active goals
  (email-delivery-beta done; cli-product-config done with waived criterion)
- change: goal.md constraint now names both fix branches as the P2 gate;
  while closing cli-product-config, its 07-28 journal revealed the P3
  decision already exists: TOML deliberately omits detail_selection/profile
  (balanced + per-topic `detail`)
- disposition: P3 likely closes as documented-as-fixed pending user
  confirmation
- next: P2 starts after the fix branches merge; P3 confirm with user

## 2026-07-31 — note (P3 closed; paper-identity closed)

- evidence: user confirmed P3 close; cli-toml-schema.md already documents
  the decision in six places (59–60, 178, 186, 348, 455, 465) — no doc
  change needed. Also closed paper-identity-source-boundary as done (all
  criteria checked, merged via 0.3.3)
- change: goal P3 → done, resolved section added
- disposition: config.ts hard-coding balanced matches the schema; keep
- next: P2 blocked on the two fix branches; nothing else pending

## 2026-07-31 — note (P2 unblocked; all merge work done)

- evidence: fix/arxiv-failure-recovery reviewed (helm 2026-07-30 goal done,
  journal verification reproduced: 1139 tests + typecheck/lint/boundaries
  green), committed as 1657069, merged into main be4c705; request-resilience
  and cli-smoke-build contained the same commits and merged implicitly
- change: P2 → pending (gate cleared); PR #3 merged 1d24600, PR #2 merged 52c74ae
- disposition: keep — remaining unmerged branches (personalized-literature-agent,
  ui-polish, rust-standalone) are unrelated/stale, left untouched
- next: start P2 (split dashboard view.ts; settings getSettingDefinitions
  migration) when the user confirms

## 2026-07-31 — note (P2a started; dead branches cleaned)

- evidence: goal re-cut into P2a (view.ts split) / P2b (settings API);
  deleted abandoned Rust-rewrite branches feat/ui-polish (⊂
  refactor/rust-standalone) local + origin — no main traces, dead since
  2026-07-15; remaining unmerged: feat/personalized-literature-agent
- change: P2a active on refactor/dashboard-view-modules; phase
  02-a-split-dashboard-view.md written (7 tasks, extraction order T1–T7)
- disposition: view.ts class body untouched this phase; HubModal extracted
- next: T1 constants.ts extraction

## 2026-07-31 — note (P2a done)

- evidence: view.ts 3360 → 2443 lines; 6 new modules (constants, types,
  calendar, files, pagination, log-format, detail-refs, actions,
  hub-modal); two source-level test suites retargeted to the new homes
  (detail-refs.ts, hub-modal.ts); full workspace 1139 tests green,
  typecheck green, lint back at the 53-warning baseline
- change: P2a closed; outcome corrected (L1): class body stays intact per
  scope, so ≤1700 was unreachable; class-internal rendering splits noted
  for a future phase
- disposition: keep all modules; no behavior change (byte-identical output)
- next: P2b — settings tab getSettingDefinitions migration with 1.4.0
  fallback

## 2026-07-31 — note (T6 done: tab wired to the declarative API)

- evidence: the three overrides land on ArxivDailySettingTab —
  getSettingDefinitions binds every host callback (declarative-rows
  renderers + the tab's mutation methods), getControlValue /
  setControlValue route flat keys through the dotted-path resolvers and
  persist via saveSettings, with display()-parity trimming for email
  to/fromEmail (fromName stays raw, resolving T5's noted deviation);
  +10 structure tests in a new settings-declarative-tab.test.ts (key
  rows present, every control key resolves through getControlValue,
  list mutations route to tab methods, round-trip setControlValue);
  the settings-tab source-inspection suite (33 tests) passes unchanged —
  the three overrides sit at the class top and disturb none of its
  regexes
- change: T6 committed; phase 02-b → done; full workspace 1163 tests
  green (plugin 272), typecheck green, lint 0 errors / 50 warnings (the
  obsidianmd/settings-tab/prefer-setting-definitions warning is gone; the
  2 sentence-case copies of imperative copy remain), boundaries OK,
  esbuild build OK
- disposition: display() untouched (<1.13 fallback unchanged); runtime
  1.13+ behavior still unverifiable locally (structure tests + types are
  the safety net, as assumed)
- next: PR review + merge; P2b goal criterion stays unchecked until the
  merge lands (mirrors how P2a was handled)

## 2026-07-31 — note (P2b started)

- evidence: user confirmed migration despite limited user-facing value
  (Settings search + native rendering); API researched from obsidian
  1.13.1 types — declarative items (action/control/render/group/page/
  list), resolver hooks getControlValue/setControlValue default to
  plugin.settings (ours is nested → need path mapping)
- change: P2b active; phase 02-b-settings-declarative-api.md written
  (6 tasks, T1–T6); display() stays as <1.13 fallback
- disposition: new settings/definitions.ts module; reuse existing tab.ts
  helpers; runtime 1.13 behavior not locally testable (documented
  assumption)
- next: T1 key constants + resolver mapping

## 2026-07-31 — note (branch re-cut)

- evidence: PR #4 merged (pure P2a); P2b T1/T2 were on the same branch
  and were split out to keep main clean of un-wired declarative code
- change: P2b commits cherry-picked onto new branch
  refactor/settings-declarative-api (from merged main); journal conflict
  resolution also restored the P2a-done record that PR #4 had dropped
- disposition: keep working on refactor/settings-declarative-api
- next: T3 complex rows (API key sentinel, get models, onboarding guide)

## 2026-07-31 — note (T4 done: arXiv section declarative)

- evidence: categories + topics lists, quick start, detail notes, timezone
  all expressible; the `list` type cannot nest inside a `group`, so the
  arXiv section items sit top-level between the AI model and Output &
  schedule groups; two settings-tab source regressions forced keeping
  `renderTopicCard` private (public `renderTopicRow` wrapper instead) and
  restoring the `${topicName}` delete message shape — tests untouched
- change: T4 committed 35c6140; new shared tab mutations (addTopic,
  deleteTopic, addCategory, deleteCategory, reorderCategories,
  reorderTopics, applyTopicTemplate) with a `refreshSettings` dispatcher
  (update() on 1.13+ via requireApiVersion gate, display() otherwise);
  TIMEZONE_OPTIONS + addCategoryOptions exported for reuse
- disposition: topics delete stays in the card (confirm + expanded-state
  cleanup) so the topics list has no framework delete button; drag reorder
  is a new 1.13+ capability (report section order); detail-notes "custom"
  option mirrors display()'s conditional row; list descs render as first
  item with empty name (L2 if the layout looks off)
- next: T5 email + schedule blocks

## 2026-07-31 — note (T5 done: schedule + email blocks declarative)

- evidence: T2's plain Enable toggle bypassed setScheduleEnabled
  (validation modal + scheduler start), so T5 re-renders it as a
  render row with a dynamic "Enable · Running/Paused" name; tick
  interval likewise re-rendered (sanitize + restartScheduler); email
  rows are build-time mode conditionals (self: api key + from;
  hosted: verify + code) refreshed via refreshSettings on mode change
- change: T5 committed 5d66b15; tab.ts exposes saveRunWindowTime,
  renderRunWindowTimeSelect, emailGuideContent (display() refactored
  to use the content builder — no behavior change); +3 structure
  tests (enable row, schedule group rows, mode-dependent email rows)
- disposition: email text controls (to/fromEmail/fromName) write raw
  values via the resolver — the imperative trims; acceptable deviation
  noted for T6 setControlValue if trimming is wanted; 2 new lint
  warnings are sentence-case copies of existing imperative copy
- next: T6 wire tab.ts (getSettingDefinitions + getControlValue +
  setControlValue overrides) + final structure tests + verification

## 2026-07-31 — note (P2b manual-test UI follow-up)

- evidence: 1.13+ manual testing found completed onboarding and Quick start
  redundant, LLM controls crowded, category rows noisy, and topic cards
  unable to collapse because the scoped grid rule overrode the collapsed
  display rule; list description pseudo-items also offset framework reorder /
  delete indices
- change: L1 adjustment within P2b — completed setup and Quick start removed
  only from declarative definitions; LLM reordered as URL / key / model /
  reasoning with a masked revealable blur-or-Enter key input and None folded
  into reasoning; categories use numbered fixed dropdowns; topic copy and
  spacing reduced, responsive grid added, collapse precedence fixed; new
  installs now default reasoning effort to medium
- disposition: <1.13 display() remains unchanged; existing saved reasoning
  effort is retained; existing non-preset category values remain selectable
  until changed; pipeline rendering code is untouched
- validation: plugin typecheck green; plugin 278 tests green; lint 0 errors /
  52 warnings; boundaries OK; production build OK
- next: copy manifest/main/styles to the test vault and wait for another
  1.13+ manual pass before push / PR

## 2026-07-31 — note (P2b second manual-test UI follow-up)

- evidence: the framework's declarative list reorder API renders drag handles
  outside custom topic cards, producing a second visual layer; email action
  definitions make the whole setting row clickable, and the email guide's
  empty framework name/control columns indent its card
- change: L1 adjustment — remove reorder from both category and topic lists;
  keep topic tags immediately after shrinkable titles; make the declarative
  email guide full width; move verification/test sends into right-side control
  buttons; give hosted verification codes the same masked, revealable,
  blur-or-Enter save flow as the LLM key, including normalization and rollback
- disposition: no brittle styling of private framework drag DOM; <1.13
  display() and its sentinel controls stay unchanged; inline test sending
  first persists the current email/token draft to avoid stale credentials
- validation: plugin typecheck green; plugin 286 tests green; lint 0 errors /
  52 warnings; boundaries OK; production build OK
- next: deploy three plugin files, then another 1.13+ manual pass before
  push / PR

## 2026-08-01 — steer (P2b output criterion clarified)

- evidence: P2b changes `DEFAULT_SETTINGS.llm.reasoningEffort` from high to
  medium at the user's request; reasoning effort is sent to the provider, so
  live LLM prose for new or incomplete configurations cannot be guaranteed
  byte-identical even though output assemblers, writers, fixtures, and their
  tests are unchanged from main
- change: L3 success-criterion steer approved by the user — retain Medium and
  narrow the output criterion to deterministic rendering with identical fetched
  data and LLM responses; record live-response differences from the intentional
  default change as an explicit waiver, not as byte-identity evidence
- disposition: deterministic report/note rendering is source-identical and the
  full plugin regression suite is green; proceed to push, PR, CI, and merge
- next: merge P2b; check `P2b landed` only after the merge reaches main
