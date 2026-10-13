/**
 * Mirrors exploration-browser.mjs for the Obsidian plugin surface (P2 of
 * docs/helm/2026-10-10-model-driven-release-exploration). Chunk 1 ships only
 * the plugin-scoped observation and restriction policy: a self-contained,
 * stringifiable observation function (run inside Obsidian's renderer via CDP
 * `Runtime.evaluate`, see obsidian-ui.mjs/desktop-acceptance/cdp.mjs) plus a
 * pure restriction classifier. CDP action execution and host wiring land in
 * Chunks 2-3; this file does not drive a real session yet.
 *
 * Root-selector evidence (file:line), see also the chunk-1 report:
 * - Dashboard view: `.arxiv-daily-dashboard` on the ItemView's own contentEl
 *   (plugin/src/dashboard/view.ts:524,535) — leaf-scoped, so other panes are
 *   naturally excluded.
 * - Settings tab: Obsidian 1.13+ renders our tab via `getSettingDefinitions()`
 *   (tab.ts:238), which is Obsidian's OWN declarative-settings engine — no
 *   plugin CSS class on `containerEl` is guaranteed there. The reliable
 *   signal is Obsidian's own object graph: `app.setting.activeTab.id ===
 *   "arxiv-daily"` (same id already used by `app.setting.openTabById(
 *   "arxiv-daily")` in obsidian-ui.mjs:97). The legacy `<1.13` fallback
 *   (`display()`) additionally adds a `.arxiv-daily-settings` class
 *   (tab.ts:1394), kept as a secondary signal.
 * - Marked plugin modals (Chunk 2, plugin/src/plugin-modal.ts): every modal
 *   this plugin opens — `Modal` subclass or a one-off `new Modal(app)` — now
 *   carries a shared `.arxiv-daily-modal` class on `modalEl`, added by
 *   `ArxivDailyModal`/`markArxivDailyModal()`. Existing per-modal classes
 *   (HubModal `.arxiv-daily-hub-modal`, SimilarPapersModal
 *   `.arxiv-daily-similar-modal` on contentEl, PersonalLibraryInterestProfileModal
 *   `.arxiv-daily-interest-review-modal`, the library authorization modal's
 *   `.arxiv-daily-library-authorization-modal`) are kept but no longer needed
 *   for scoping — the single shared marker is sufficient and is the only
 *   thing this file looks for. Chunk 1's "single open non-settings modal"
 *   fallback is removed: an unmarked modal is, by construction, not ours, and
 *   is now invisible rather than guessed at. See the "known gap closed" test
 *   in acceptance-exploration-obsidian.test.mjs.
 * - Obsidian's `Menu` (dashboard "More" toolbar menu, view.ts:1874
 *   `showMoreMenu`) renders into a body-level `.menu` element outside any
 *   plugin-scoped root with no plugin-identifying class in Obsidian's public
 *   API (confirmed by reading view.ts:1874-2040 and node_modules/obsidian/
 *   obsidian.d.ts). P3 Chunk 1 adds a tiny product marker instead of leaving
 *   it out of scope: `showMoreMenu` now calls `markArxivDailyMenu(menu)`
 *   (plugin/src/plugin-menu.ts) right after `new Menu()`, adding a shared
 *   `.arxiv-daily-menu` class to the menu's own `dom` — a property not in
 *   `obsidian.d.ts` but confirmed, by extracting and reading (never
 *   modifying) the installed Obsidian's own `resources/obsidian.asar`, to be
 *   set in the `Menu` constructor itself (so it exists immediately, before
 *   the menu is even shown) and to be the exact element later appended to
 *   `document.body`. This mirrors the existing `.arxiv-daily-modal` marker
 *   and was chosen over harness-side "last plugin click" provenance tracking
 *   because it is unambiguous by construction: an unmarked `.menu`
 *   (Obsidian's own context menus, other plugins' menus) is simply never
 *   matched, with no separate "ambiguous menu" case to get wrong. Each menu
 *   item is a plain `<div class="menu-item tappable">` with no
 *   `role`/`tabindex` (also confirmed via the asar read), so `CONTROL_
 *   SELECTOR` below adds `.menu-item`, scoped to this one surface by
 *   `resolveSurface()`; `MenuItem.setDisabled(true)` toggles an `is-disabled`
 *   class on the item's own dom rather than a native `disabled` property,
 *   handled in `describeControl`. See `classifyControlRestriction` for which
 *   of the menu's items are restricted (the scheduler toggle, "Clear run
 *   state…", and the four external-link items) versus merely disabled (the
 *   "Scheduler: Enabled/Disabled" status item) versus left open ("Run for
 *   date…", "Force run for date…", "Retry failed dates", "Cancel active
 *   tasks", "Run pending", "Show logs & history").
 *
 * Chunk 3 (CDP action execution with live re-validation) adds
 * `createObsidianActionExecutor`. Decisions, each justified where used below:
 * - click/fill reuse `waitForStableTarget` (obsidian-ui.mjs) against a new
 *   per-ref hit-test expression, then send real `Input.dispatchMouseEvent`
 *   moved/pressed/released — the same real-input approach `runDate` already
 *   uses for the date picker's submit button.
 * - fill selects existing content with a real Ctrl+A key event (a native
 *   Chromium/Electron edit command on a focused text input) rather than
 *   setting `selectionStart`/`selectionEnd` in-page, to stay consistent with
 *   "real events, not DOM shortcuts" for anything that reaches product code;
 *   it commits with a trailing Tab (blur) after verifying the live value,
 *   because some Setting-built fields only save on the native `change` event
 *   (fired on blur), mirroring obsidian-ui.mjs's own `fill()` helper, which
 *   also blurs after dispatching `input`/`change`.
 * - select sets `.value` in-page and dispatches `input`/`change` instead of
 *   driving Chromium's native `<select>` popup with CDP mouse events, which
 *   is not reliably drivable in Electron (per the phase doc's own
 *   assumption); it never produces an `Input.*` call.
 * - scroll sends one `Input.dispatchMouseEvent` `mouseWheel` at the center of
 *   the currently open plugin surface (dashboard/settings/modal), not at a
 *   specific ref — there is no hover region to target, unlike the workbench.
 * - press (the shared, host-agnostic keyboard action from exploration.mjs)
 *   is rejected without sending anything when `document.activeElement` is
 *   outside the current plugin surface. P1 Chunk 1 (review Finding 1): the
 *   scope check alone used to be the only guard — it never classified the
 *   focused element itself, so Tab-focusing a restricted/sensitive control
 *   (an external-link settings button, the scheduler toggle, "Clear run
 *   state…", an API-key reveal button) and pressing Enter/Space activated it
 *   with zero interception. `press` now also describes
 *   `document.activeElement` with the same `describeControl` the observation
 *   uses (`obsidianFocusedControlExpression`) and runs it through the same
 *   `classifyControlRestriction`/sensitive/disabled policy `revalidateRef()`
 *   applies to click/fill/select — for every key except Tab/Shift+Tab/Escape
 *   (`PRESS_KEYS_ALWAYS_ALLOWED_ON_FOCUS`), which never activate or mutate
 *   the focused control and so stay available even on a restricted control,
 *   keeping a keyboard-only persona from being stranded on one. Only then is
 *   a real `Input.dispatchKeyEvent` keyDown/keyUp pair sent. Native `<form>`
 *   submission on Enter is not reachable here: the plugin's settings/modals
 *   are built with Obsidian's own `Setting` API (plain divs), which never
 *   renders an actual `<form>` element (confirmed by reading
 *   plugin/src/settings/tab.ts and plugin/src/plugin-modal.ts) — unlike the
 *   workbench host, which does use real `<form>`s.
 * - back presses Escape only when a plugin modal (`.arxiv-daily-modal`) or
 *   the plugin's own menu (`.arxiv-daily-menu`, P3 Chunk 1) is open, closing
 *   whichever is open — Obsidian's own `Menu` registers Escape on its
 *   keyboard Scope to close itself, the same mechanism a modal's Escape
 *   handling uses (confirmed via the asar read above), so one real Escape
 *   suffices either way; otherwise back is rejected. reload is always
 *   rejected: Obsidian has no per-page reload, and resetting state between
 *   tasks is the driver's job (Chunk 4), not an exploration action's.
 * - Live re-validation (every ref-bearing action) re-runs the same in-page
 *   surface/visibility logic the observation uses — `resolveSurface`,
 *   `controlVisible`, `describeControl` below are shared, not duplicated,
 *   between `obsidianObservation` and the new `obsidianLiveControl` —  and
 *   the exported `classifyControlRestriction` is reused unchanged, so a ref
 *   that has gone stale, left plugin scope, or become
 *   restricted/sensitive/disabled/read-only is rejected before any
 *   `Input.*` call is sent.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { blockersFromError } from "../desktop-acceptance/app-state.mjs";
import { preflight } from "../desktop-acceptance/preflight.mjs";
import { runDesktopSession } from "../desktop-acceptance/session.mjs";
import { installSettingsFixture } from "../desktop-acceptance/settings-fixture.mjs";
import { collapseOptionSets, ExplorationActionError, PRESS_KEYS } from "./exploration.mjs";
import { classifyObsidianDiagnostics, createObsidianSettings } from "./obsidian.mjs";
import { createObsidianDriver, PLUGIN, pollJsonFile, waitForStableTarget } from "./obsidian-ui.mjs";

/**
 * Embed a value in JavaScript sent to the renderer. U+2028 and U+2029 end a
 * statement in source text, so they are escaped like the other desktop helpers.
 */
const asCode = value => JSON.stringify(value)
  .replace(/\u2028/g, "\\u2028")
  .replace(/\u2029/g, "\\u2029");

/**
 * The in-page observation. Self-contained (no closures over outer
 * variables) so it can be stringified and sent as a CDP `Runtime.evaluate`
 * expression, exactly like `browserObservation` is passed to Playwright's
 * `page.evaluate`. `testMode` lets unit tests run this against happy-dom,
 * whose `getBoundingClientRect()` always returns zeroes: when true, a
 * zero-size rect no longer excludes an element, but `display:none`,
 * `visibility:hidden`, `opacity:0`, and `[hidden]`/`[inert]` ancestors still
 * do. Tests that need real `inViewport`/`rect` assertions can still
 * override `getBoundingClientRect` on individual nodes, same as the
 * existing `acceptance-obsidian.test.mjs` fixtures do.
 *
 * `resolveName`, `controlVisible`, `resolveSurface`, and `describeControl`
 * are plain top-level functions (not nested) precisely so Chunk 3's live
 * re-validation (`obsidianLiveControl`) can reuse them too: every in-page
 * expression builder below concatenates the `.toString()` source of the
 * helpers it needs into one self-evaluating block (function declarations
 * hoist, so concatenation order does not matter), so the surface/visibility/
 * restriction policy is written once.
 */
function resolveName(node) {
  // Obsidian's own `.setting-item`/`.setting-item-name` convention (stable,
  // part of its public Setting API) is the label fallback, since Obsidian's
  // Setting-built inputs carry no `name="..."` attribute (confirmed by
  // reading tab.ts addText() call sites).
  const own = () => (node.textContent || "").trim();
  if (node.getAttribute("aria-label")) return node.getAttribute("aria-label");
  const labelledBy = (node.getAttribute("aria-labelledby") || "").split(/\s+/).filter(Boolean)
    .map(labelId => document.getElementById(labelId)?.textContent || "").join(" ").trim();
  if (labelledBy) return labelledBy;
  const labelled = Array.from(node.labels || []).map(item => item.textContent || "").join(" ").trim();
  if (labelled) return labelled;
  if (node.getAttribute("title")) return node.getAttribute("title");
  if (/^(BUTTON|A|SUMMARY)$/.test(node.tagName) && own()) return own();
  const settingName = node.closest(".setting-item")?.querySelector(".setting-item-name")?.textContent?.trim();
  if (settingName) return settingName;
  return (node.getAttribute("name") || own() || "").trim();
}

function controlVisible(node, testMode) {
  const rect = node.getBoundingClientRect();
  const style = getComputedStyle(node);
  const sized = testMode ? true : rect.width > 0 && rect.height > 0;
  return sized && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0" && !node.closest("[hidden],[inert]");
}

/**
 * Resolves the one surface (menu/modal/settings/dashboard) the model may
 * see, in priority order. The dashboard "More" toolbar menu (an Obsidian
 * `Menu`, view.ts:1874 `showMoreMenu`) is a body-level `.menu` with no
 * plugin-identifying class of its own in Obsidian's public API — the
 * plugin now marks its root with `.arxiv-daily-menu` right after
 * construction (plugin/src/plugin-menu.ts, mirroring `.arxiv-daily-modal`
 * for modals), so an unmarked `.menu` (Obsidian's own context menus, other
 * plugins' menus) is, by construction, never ambiguous with ours: it is
 * simply not found by this selector and stays fully invisible, the same
 * "no fallback" policy Chunk 2 already applies to unmarked modals. Checked
 * before modals/settings/dashboard because it is the most recent overlay
 * when open (opened from a toolbar button on the dashboard, never nested
 * inside a modal in this plugin).
 */
function resolveSurface() {
  let surface, view = "none", modalOpen = false, menuOpen = false;
  const markedMenu = document.querySelector(".arxiv-daily-menu");
  if (markedMenu) { surface = markedMenu; view = "menu"; menuOpen = true; }
  if (!surface) {
    const markedModal = document.querySelector(".arxiv-daily-modal");
    if (markedModal) { surface = markedModal.closest(".modal") || markedModal; view = "modal"; modalOpen = true; }
  }
  if (!surface) {
    const activeTab = typeof app !== "undefined" ? app?.setting?.activeTab : undefined;
    const settingsOpen = (typeof app !== "undefined" && app?.setting?.isOpen && activeTab?.id === "arxiv-daily") || Boolean(document.querySelector(".arxiv-daily-settings"));
    if (settingsOpen) {
      surface = (activeTab?.id === "arxiv-daily" && activeTab.containerEl) || document.querySelector(".arxiv-daily-settings");
      if (surface) view = "settings";
    }
  }
  if (!surface) {
    const dashboard = document.querySelector(".arxiv-daily-dashboard");
    if (dashboard) { surface = dashboard; view = "dashboard"; }
  }
  return { surface, view, modalOpen, menuOpen };
}

/** Builds the observed/live control shape for one node; `ref` is already known (generated or re-found by attribute). */
function describeControl(node, ref) {
  const rect = node.getBoundingClientRect();
  const name = resolveName(node).slice(0, 180);
  const field = node.id || node.getAttribute("name") || "";
  const hasRevealSibling = Boolean(node.parentElement?.querySelector(".arxiv-daily-settings__reveal-key"));
  const sensitive = node.type === "password" || hasRevealSibling || /api.?key|password|secret|token|verification code/i.test(name);
  return {
    ref, tag: node.tagName.toLowerCase(), type: node.getAttribute("type") || "", role: node.getAttribute("role") || "",
    name, field,
    // Obsidian's own menu items are plain, non-focusable `<div class="menu-item">`s: `MenuItem.
    // setDisabled(true)` (resources/obsidian.asar, read-only inspection) toggles an `is-disabled`
    // class on the item's own dom rather than a native `disabled` property or `aria-disabled`.
    disabled: node.disabled === true || node.getAttribute("aria-disabled") === "true" || node.classList.contains("is-disabled"), readOnly: node.readOnly === true,
    ...(sensitive ? { sensitive: true } : {}),
    ...(typeof node.value === "string" ? { value: sensitive ? "" : node.value.slice(0, 1000) } : {}),
    ...(typeof node.checked === "boolean" ? { checked: node.checked } : {}),
    ...(node.hasAttribute("href") ? { href: node.getAttribute("href").slice(0, 1000) } : {}),
    ...(node.tagName === "SELECT" ? { options: Array.from(node.options).slice(0, 60).map(option => ({ value: option.value.slice(0, 256), label: option.text.slice(0, 96), disabled: option.disabled })) } : {}),
    dataAction: node.getAttribute("data-settings-action") || node.getAttribute("data-settings") || node.getAttribute("data-action") || "",
    paperKey: node.getAttribute("data-arxiv-id") || node.getAttribute("data-paper") || "",
    purpose: "",
    rect: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) },
    inViewport: rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth,
  };
}

// `.menu-item` is added for the dashboard's "More" menu (resources/obsidian.asar, read-only
// inspection confirms Obsidian's own MenuItem never sets role/tabindex on it, unlike every other
// interactive element this selector already covers) — scoped to the single `.arxiv-daily-menu`
// surface by resolveSurface(), so it never matches anything outside an open plugin menu.
const CONTROL_SELECTOR = 'button,summary,a[href],input:not([type="hidden"]),textarea,select,[role="button"],[role="tab"],[role="checkbox"],[role="combobox"],[tabindex],.menu-item';

export function obsidianObservation(id, testMode) {
  const { surface, view, modalOpen, menuOpen } = resolveSurface();
  const controls = [];
  if (surface) {
    for (const node of surface.querySelectorAll(CONTROL_SELECTOR)) {
      if (!controlVisible(node, testMode) || controls.length >= 220) continue;
      const ref = `${id}-${controls.length + 1}`;
      node.setAttribute("data-acceptance-ref", ref);
      controls.push(describeControl(node, ref));
    }
  }

  // Obsidian's `.notice` is a generic class (no plugin marker); surfaced as
  // text-only context per the chunk-1 contract, never as a control.
  const notices = Array.from(document.querySelectorAll(".notice")).map(node => (node.textContent || "").trim()).filter(Boolean).slice(0, 20);

  // P4 Chunk 2 Fix 2: with zero controls and (before this) no explanation, a model that ends up here could
  // only send `done` (the real P4 Chunk 1 run's first-time-user persona dead-ended exactly this way).
  // `home` is always advertised (it is always safe: in Obsidian it reopens the Reading Dashboard, see
  // createObsidianExplorationDriver's execute() below). The guidance triggers on "zero controls", not
  // only "no surface at all": a real-host run surfaced a second, narrower shape of the same dead-end —
  // after opening a note from the dashboard, Obsidian kept the dashboard leaf's own `.arxiv-daily-
  // dashboard` element mounted (so resolveSurface() still reports `view: "dashboard"`, not "none"), but
  // every control inside it had become invisible because that leaf itself was backgrounded — `controls`
  // is still empty, and a model there is exactly as stuck as with no surface at all. Any real surface text
  // is kept (never discarded), with the guidance appended, so no information is lost either way.
  const noControls = controls.length === 0;
  const recoveryText = "There are no interactive controls visible right now (you may be outside the arXiv Daily plugin, e.g. viewing a plain note or another pane, or on a plugin view with nothing currently visible). Use the \"home\" action to return to the Reading Dashboard.";
  const baseText = surface ? surface.innerText ?? surface.textContent ?? "" : "";
  return {
    id, title: document.title,
    text: (noControls ? [baseText, recoveryText].filter(Boolean).join("\n\n") : baseText).slice(0, 16000),
    controls,
    // Chunk 3: which browser-shaped actions this host currently allows, so the model does not have
    // to guess at back/reload support that differs by host and by whether a modal or the plugin's
    // own menu is open. `reload` is never listed — Obsidian has no per-page reload, and resetting
    // state between tasks is the driver's job, not an action a model can request. `home` (P4 Chunk 2
    // Fix 2) is always listed: it is a safe, idempotent recovery action from any state.
    capabilities: ["click", "fill", "select", "scroll", "wait", "press", ...((modalOpen || menuOpen) ? ["back"] : []), "home"],
    layout: { rootMounted: Boolean(surface), view, dialogOpen: modalOpen || menuOpen, viewport: { width: innerWidth, height: innerHeight }, notices },
  };
}

/** Builds the CDP `Runtime.evaluate` expression string for `obsidianObservation`, matching obsidian-ui.mjs's `read(expression)` convention (the caller wraps with `JSON.stringify(...)`). */
export function obsidianObservationExpression(id, testMode = false) {
  const shared = [resolveName, controlVisible, resolveSurface, describeControl].map(fn => fn.toString()).join("\n");
  return `(() => { ${shared}\n  const CONTROL_SELECTOR = ${asCode(CONTROL_SELECTOR)};\n  return (${obsidianObservation})(${asCode(id)}, ${asCode(Boolean(testMode))});\n})()`;
}

/**
 * Chunk 3 live re-validation: re-finds a ref by its `data-acceptance-ref`
 * attribute (set during a prior observation and otherwise untouched, exactly
 * like the workbench driver's `[data-acceptance-ref="..."]` lookup) and
 * reuses the same surface/visibility logic the observation uses, so a ref
 * that no longer exists, has left the plugin's current surface, or is no
 * longer visible is reported as not-ok without any separate "stale
 * observation id" bookkeeping — this executor is stateless between calls.
 */
function obsidianLiveControl(ref, testMode) {
  const node = document.querySelector(`[data-acceptance-ref="${ref}"]`);
  if (!node || !node.isConnected) return { ok: false, reason: "gone" };
  const { surface } = resolveSurface();
  if (!surface || !(surface === node || surface.contains(node))) return { ok: false, reason: "scope" };
  if (!controlVisible(node, testMode)) return { ok: false, reason: "hidden" };
  return { ok: true, control: describeControl(node, ref) };
}

export function obsidianLiveControlExpression(ref, testMode = false) {
  const shared = [resolveName, controlVisible, resolveSurface, describeControl].map(fn => fn.toString()).join("\n");
  return `(() => { ${shared}\n  return (${obsidianLiveControl})(${asCode(ref)}, ${asCode(Boolean(testMode))});\n})()`;
}

/**
 * A single node's click/fill target geometry and hit-test, in the shape
 * `waitForStableTarget` (obsidian-ui.mjs) expects — the same shape
 * `dateSubmitTargetExpression` already produces for the date picker's submit
 * button, generalized to any ref. Tests stub `getBoundingClientRect` and
 * `document.elementFromPoint` per node/case, matching the existing
 * `acceptance-obsidian.test.mjs` convention; there is no `testMode` bypass
 * here because, unlike the many-controls observation scan, a single-node hit
 * test is cheap for a test to stub directly.
 */
export function obsidianHitTestExpression(ref) {
  const selector = asCode(`[data-acceptance-ref="${ref}"]`);
  return `(() => {
    const node = document.querySelector(${selector});
    if (!node || !node.isConnected) return { ready: false, reason: "gone" };
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    const visible = rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
    const x = rect.x + rect.width / 2, y = rect.y + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    const hitTarget = hit === node || node.contains(hit);
    return {
      ready: visible && hitTarget && node.disabled !== true,
      x, y, width: rect.width, height: rect.height, hitTarget,
      hit: hit ? { tag: hit.tagName, cls: hit.className, text: (hit.textContent ?? "").trim().slice(0, 100) } : null,
    };
  })()`;
}

function obsidianSurfaceCenter() {
  const { surface } = resolveSurface();
  if (!surface) return null;
  const rect = surface.getBoundingClientRect();
  return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
}

/** The center of the current plugin surface, for `scroll` — there is no hover region to target in Obsidian. */
export function obsidianSurfaceCenterExpression() {
  const shared = [resolveSurface].map(fn => fn.toString()).join("\n");
  return `(() => { ${shared}\n  return (${obsidianSurfaceCenter})();\n})()`;
}

function obsidianFocusInScope() {
  const { surface } = resolveSurface();
  const active = document.activeElement;
  if (!surface || !active) return false;
  return active === surface || surface.contains(active);
}

/** Whether the currently focused element is inside the plugin's current surface, for `press`. */
export function obsidianFocusInScopeExpression() {
  const shared = [resolveSurface].map(fn => fn.toString()).join("\n");
  return `(() => { ${shared}\n  return (${obsidianFocusInScope})();\n})()`;
}

/**
 * P1 Chunk 1 (review Finding 1 fix): `obsidianFocusInScope` above only answers "is focus
 * somewhere inside the plugin surface" — it never describes/classifies the focused element
 * itself, unlike every ref-bearing action's `revalidateRef()`. That gap let a Tab-focused
 * restricted or sensitive control (an external-link settings button, the scheduler toggle,
 * "Clear run state…", an API-key reveal button) be activated by `press Enter`/`press Space`
 * with no interception. This reuses the SAME `describeControl` the observation and
 * `obsidianLiveControl` already build — not a second, divergent describer — so `press`'s
 * executor can run it through the identical `classifyControlRestriction`/sensitive/disabled
 * policy `revalidateRef()` applies to click/fill/select.
 */
function obsidianFocusedControl(testMode) {
  const { surface } = resolveSurface();
  const active = document.activeElement;
  if (!surface || !active) return { ok: false, reason: "scope" };
  if (!(active === surface || surface.contains(active))) return { ok: false, reason: "scope" };
  if (!controlVisible(active, testMode)) return { ok: false, reason: "hidden" };
  return { ok: true, control: describeControl(active, null) };
}

/** Describes the currently focused element (scope + the shared control descriptor), for `press`. */
export function obsidianFocusedControlExpression(testMode = false) {
  const shared = [resolveName, controlVisible, resolveSurface, describeControl].map(fn => fn.toString()).join("\n");
  return `(() => { ${shared}\n  return (${obsidianFocusedControl})(${asCode(Boolean(testMode))});\n})()`;
}

/**
 * P1 Chunk 3 (review Finding 1 "Compounding" fix): Obsidian screenshots had no secret masking at
 * all, unlike the workbench driver's `mask:` list (exploration-browser.mjs:234) — a revealed
 * ("Show ... API key") or plain password input would be captured verbatim into a PNG artifact by
 * `capture()` below. Kept byte-for-byte identical to `describeControl`'s own `sensitive`
 * computation above (same three conditions, same name source) rather than a second, divergent
 * rule; defined as its own function instead of folding into `describeControl` so the already-
 * accepted observation/press-revalidation paths and their `shared` serialization bundles (lines
 * 305, 328, 409) stay untouched. `hasRevealSibling` is what covers a *revealed* key input: the
 * reveal button's sibling class is there regardless of whether the input's own `type` is currently
 * "password" or "text".
 */
function isSecretField(node) {
  const name = resolveName(node).slice(0, 180);
  const hasRevealSibling = Boolean(node.parentElement?.querySelector(".arxiv-daily-settings__reveal-key"));
  return node.type === "password" || hasRevealSibling || /api.?key|password|secret|token|verification code/i.test(name);
}

/**
 * Covers every secret `<input>` in the whole document — not only the model's current plugin
 * surface — with an opaque, non-interactive overlay positioned over its live rect, each tagged
 * `data-exploration-mask` so `obsidianUnmaskSecretFields` can find and remove exactly these and
 * nothing else. The whole document, not just the current surface: a screenshot's pixels are
 * whatever is visually on top of the photographed region (e.g. Obsidian's own settings window),
 * regardless of which element the model's own observation happens to scope to. Returns the count
 * masked, mainly so a caller (or a test) can tell something happened.
 */
function obsidianMaskSecretFields() {
  let count = 0;
  for (const node of document.querySelectorAll("input")) {
    if (!isSecretField(node)) continue;
    const rect = node.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue; // nothing to cover if it is not laid out/visible
    const overlay = document.createElement("div");
    overlay.setAttribute("data-exploration-mask", "1");
    overlay.style.cssText = `position:fixed;left:${rect.left}px;top:${rect.top}px;width:${rect.width}px;height:${rect.height}px;background:#000;z-index:2147483647;pointer-events:none;`;
    document.body.appendChild(overlay);
    count += 1;
  }
  return count;
}

/** Builds the CDP `Runtime.evaluate` expression for `obsidianMaskSecretFields`, for use via `helper.read(...)` right before `capture()`'s screenshot. */
export function obsidianMaskSecretFieldsExpression() {
  const shared = [resolveName, isSecretField].map(fn => fn.toString()).join("\n");
  return `(() => { ${shared}\n  return (${obsidianMaskSecretFields})(); })()`;
}

/** Removes every overlay `obsidianMaskSecretFields` added, identified solely by its own marker attribute — never anything it did not add itself. */
function obsidianUnmaskSecretFields() {
  const nodes = document.querySelectorAll('[data-exploration-mask="1"]');
  nodes.forEach(node => node.remove());
  return nodes.length;
}

/** Builds the CDP `Runtime.evaluate` expression for `obsidianUnmaskSecretFields`, for use via `helper.read(...)` right after `capture()`'s screenshot (including when it throws). */
export function obsidianUnmaskSecretFieldsExpression() {
  return `(() => { return (${obsidianUnmaskSecretFields})(); })()`;
}

// Keys that only move focus or cancel, rather than activating the focused control's own
// handler or mutating its value the way native Enter/Space button-activation and native
// <select>/listbox Arrow/Home/End/Page navigation do — sent regardless of the focused
// control's own restriction, so a keyboard-only persona that lands on (merely observes) a
// restricted control can always Tab past it or Escape away, instead of being stranded with
// only "home" (which abandons its current position entirely) as an escape route.
const PRESS_KEYS_ALWAYS_ALLOWED_ON_FOCUS = new Set(["Tab", "Shift+Tab", "Escape"]);

/**
 * Obsidian capability restrictions. Pure, like the browser's
 * `classifyControlRestriction` — operates on the already-resolved control
 * (name/field/type/href), no DOM access, so it is unit-testable standalone.
 * Rules mirror the browser's policy field-for-field (see exploration-
 * browser.mjs:14-30); Obsidian's Setting-built inputs have no stable
 * `name="..."` attribute, so most rules match on the resolved label text
 * instead of a dotted field key — the chunk-1 contract explicitly allows
 * this "falling back to labels with a comment." Exact label strings are
 * cited with file:line evidence in the file header above and the chunk-1
 * report. Chunk 3 reuses this unchanged for live re-validation, so the
 * restriction policy is defined in exactly one place.
 */
export function classifyControlRestriction(control, allowedUrl) {
  const name = control.name || "";
  const field = control.field || "";
  if (control.type === "password" || control.sensitive || /api.?key|password|secret|token|verification code/i.test(name)) {
    return "credentials are excluded from exploration";
  }
  if (control.type === "file") return "file selection is outside the Obsidian capability";
  if (/choose folder|change folder|select folder/i.test(name)) return "folder selection opens a native OS dialog";
  if (/folder|directory|\bpath\b/i.test(name)) return "filesystem locations must remain owned by the fixture";
  // The dashboard "More" menu's scheduler items (view.ts:1879-1895): the disabled status item
  // ("Scheduler: Enabled/Disabled") is caught by the `disabled` check in createObsidianActionExecutor
  // instead (menu items use an is-disabled CLASS — see describeControl — not restriction-by-name); the
  // toggle itself ("Enable scheduler"/"Disable scheduler") is restricted here, same policy as the
  // settings tab's own schedule toggle just above.
  if (/^arxiv-daily-run-window-/.test(field) || /^enable\s*·/i.test(name) || /check every \(minutes\)/i.test(name) || /^(enable|disable) scheduler$/i.test(name)) return "scheduling settings are fixed by the fixture";
  if (/^api base url$|^better pdf parser$|embedding|sidecar/i.test(name)) return "external service and embedding/parser settings are fixed by the fixture";
  if (/send test|how to send|your email|send verification email|from email|from name|daily auto-send|resend api key|verification code/i.test(name)) return "email delivery is excluded from exploration";
  // "Clear run state…" (the More menu's command-backed item, commands.ts's clear-run-state): clearing
  // run state is explicitly out of scope for exploration (goal.md), independent of the generic
  // dangerous-command handling the harness already applies elsewhere.
  if (/^clear run state/i.test(name)) return "run state cannot be cleared from exploration";
  if (/^(open arxiv|open bug report|report a bug|open feature request|request a feature|open docs|open documentation|documentation|open repository|repository)$/i.test(name)) return "link leaves the plugin for an external destination";
  if (control.href) {
    // No product code currently renders a real `<a href>` for external nav
    // (confirmed by reading view.ts/tab.ts — all external opens are onClick
    // -> openUrl()); this stays as defensive parity with the browser policy
    // and the contract's explicit "http(s) not obsidian://" wording.
    try {
      const target = new URL(control.href, allowedUrl || "https://example.invalid/");
      return ["http:", "https:"].includes(target.protocol) ? "link leaves the plugin for an external destination" : undefined;
    } catch { return "link is not a permitted address"; }
  }
  return undefined;
}

/**
 * CDP `Input.dispatchKeyEvent` parameters for the shared, host-agnostic
 * `press` allowlist (exploration.mjs `PRESS_KEYS`). `modifiers` follows CDP's
 * bitmask (Alt=1, Ctrl=2, Meta=4, Shift=8); Shift+Tab is the one allowed key
 * that carries a modifier. The module-load check below keeps this mapping
 * from silently drifting out of sync with the shared allowlist.
 */
const PRESS_KEY_CODES = {
  Enter: { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 },
  Escape: { key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 },
  Tab: { key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 },
  "Shift+Tab": { key: "Tab", code: "Tab", windowsVirtualKeyCode: 9, modifiers: 8 },
  ArrowUp: { key: "ArrowUp", code: "ArrowUp", windowsVirtualKeyCode: 38 },
  ArrowDown: { key: "ArrowDown", code: "ArrowDown", windowsVirtualKeyCode: 40 },
  ArrowLeft: { key: "ArrowLeft", code: "ArrowLeft", windowsVirtualKeyCode: 37 },
  ArrowRight: { key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39 },
  Space: { key: " ", code: "Space", windowsVirtualKeyCode: 32 },
  Home: { key: "Home", code: "Home", windowsVirtualKeyCode: 36 },
  End: { key: "End", code: "End", windowsVirtualKeyCode: 35 },
  PageUp: { key: "PageUp", code: "PageUp", windowsVirtualKeyCode: 33 },
  PageDown: { key: "PageDown", code: "PageDown", windowsVirtualKeyCode: 34 },
};
for (const key of PRESS_KEYS) if (!PRESS_KEY_CODES[key]) throw new Error(`exploration-obsidian.mjs is missing a CDP key mapping for the allowed press key: ${key}`);

function refSelector(ref) { return `[data-acceptance-ref="${ref}"]`; }

/**
 * Chunk 3: the Obsidian counterpart of `exploration-browser.mjs`'s
 * `createExplorationDriver().execute`, usable standalone (Chunk 4 wires it
 * into a real CDP session/driver). `client` is a CDP client (`client.send`,
 * see desktop-acceptance/cdp.mjs); `evaluate` mirrors `session.evaluate` —
 * `expression => Promise<value>` already using `Runtime.evaluate`'s
 * `returnByValue`, exactly as `desktop-acceptance/session.mjs` constructs it
 * for `obsidian-ui.mjs`. `testMode` only affects the many-controls
 * visibility scan reused from the observation (see `controlVisible` above);
 * `stability` overrides `waitForStableTarget`'s polling for faster tests.
 */
export function createObsidianActionExecutor({ client, evaluate, redact = value => String(value ?? ""), testMode = false, stability = {} } = {}) {
  if (!client || typeof client.send !== "function") throw new Error("Obsidian action executor needs a CDP client");
  if (typeof evaluate !== "function") throw new Error("Obsidian action executor needs an evaluate(expression) function");

  // Matches obsidian-ui.mjs's own `read(expression)`: JSON round-tripping is how a value actually
  // crosses the CDP renderer boundary (see that file's comment and acceptance-exploration-obsidian's
  // buildWindow), not an arbitrary extra step.
  const read = expression => Promise.resolve(evaluate(`JSON.stringify(${expression})`)).then(raw => raw === undefined ? undefined : JSON.parse(raw));

  async function dispatchKey(name) {
    const spec = PRESS_KEY_CODES[name];
    const params = { key: spec.key, code: spec.code, windowsVirtualKeyCode: spec.windowsVirtualKeyCode, nativeVirtualKeyCode: spec.windowsVirtualKeyCode, modifiers: spec.modifiers ?? 0 };
    await client.send("Input.dispatchKeyEvent", { type: "keyDown", ...params });
    await client.send("Input.dispatchKeyEvent", { type: "keyUp", ...params });
  }

  async function dispatchSelectAll() {
    const params = { key: "a", code: "KeyA", windowsVirtualKeyCode: 65, nativeVirtualKeyCode: 65, modifiers: 2 };
    await client.send("Input.dispatchKeyEvent", { type: "keyDown", ...params });
    await client.send("Input.dispatchKeyEvent", { type: "keyUp", ...params });
  }

  async function dispatchClick(x, y) {
    await client.send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
    await client.send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
    await client.send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
  }

  /** Live re-validation (contract step 1): re-finds the ref in the current DOM before any input event. */
  async function revalidateRef(ref, actionType) {
    const live = await read(obsidianLiveControlExpression(ref, testMode));
    if (!live?.ok) throw new ExplorationActionError("Ref disappeared or is no longer in the plugin's scope; re-observe the page");
    const control = live.control;
    const restriction = classifyControlRestriction(control);
    if (restriction) throw new ExplorationActionError(`This control is restricted: ${restriction}`);
    if (control.sensitive) throw new ExplorationActionError("This control is restricted: secret");
    if (control.disabled) throw new ExplorationActionError("This control is no longer interactive (disabled)");
    if (actionType === "fill" && control.readOnly) throw new ExplorationActionError("This control is no longer interactive (read-only)");
    return control;
  }

  async function targetCenter(ref, label) {
    return waitForStableTarget(() => read(obsidianHitTestExpression(ref)), label, stability);
  }

  async function execute(action, observation, options = {}) {
    options.signal?.throwIfAborted();
    try {
      if (["click", "fill", "select"].includes(action.type)) await revalidateRef(action.ref, action.type);
      if (action.type === "click") {
        await evaluate(`(() => { document.querySelector(${asCode(refSelector(action.ref))})?.scrollIntoView({ block: "center" }); return true; })()`);
        const target = await targetCenter(action.ref, "Obsidian click target stable and hit-testable");
        await dispatchClick(target.x, target.y);
      } else if (action.type === "fill") {
        await evaluate(`(() => { document.querySelector(${asCode(refSelector(action.ref))})?.scrollIntoView({ block: "center" }); return true; })()`);
        const target = await targetCenter(action.ref, "Obsidian fill target stable and hit-testable");
        await dispatchClick(target.x, target.y);
        // P3 Chunk 3 real-host finding: the "Run for date…"/"Force run for date…" dialog's native
        // `<input type="date">` does not accept `Input.insertText` on this Obsidian/Electron build —
        // Chromium's date control is internally segmented (separate month/day/year sub-fields with
        // their own caret), and a real click + Ctrl+A + insertText sequence left the field's `.value`
        // unchanged (confirmed on a real host: it stayed ""). There is no single real keystroke
        // sequence that maps onto a YYYY-MM-DD value the way a user actually types it, so this one
        // field type is filled the same way the existing test-only `fill()` helper (obsidian-ui.mjs,
        // used by the fixed suite's own `runDate()`) already does — set `.value` directly and dispatch
        // input/change — the same "not reliably drivable with real input in Electron" exception the
        // `select` action above already documents for native `<select>` dropdowns.
        const isDateField = await read(`document.querySelector(${asCode(refSelector(action.ref))})?.type === "date"`);
        if (isDateField) {
          await evaluate(`(() => {
            const node = document.querySelector(${asCode(refSelector(action.ref))});
            if (!node) return false;
            node.value = ${asCode(action.value)};
            const EventCtor = node.ownerDocument.defaultView.Event;
            node.dispatchEvent(new EventCtor("input", { bubbles: true }));
            node.dispatchEvent(new EventCtor("change", { bubbles: true }));
            return true;
          })()`);
        } else {
          await dispatchSelectAll();
          await client.send("Input.insertText", { text: action.value });
        }
        const after = await read(`document.querySelector(${asCode(refSelector(action.ref))})?.value`);
        if (after !== action.value) throw new ExplorationActionError("Fill did not take effect; the field still shows a different value");
        await dispatchKey("Tab");
      } else if (action.type === "select") {
        const applied = await read(`(() => {
          const node = document.querySelector(${asCode(refSelector(action.ref))});
          if (!node) return false;
          const option = Array.from(node.options || []).find(item => item.value === ${asCode(action.value)});
          if (!option || option.disabled) return false;
          node.value = ${asCode(action.value)};
          const EventCtor = node.ownerDocument.defaultView.Event;
          node.dispatchEvent(new EventCtor("input", { bubbles: true }));
          node.dispatchEvent(new EventCtor("change", { bubbles: true }));
          return node.value === ${asCode(action.value)};
        })()`);
        if (!applied) throw new ExplorationActionError("Select did not take effect; the option may be disabled or absent");
      } else if (action.type === "scroll") {
        const center = await read(obsidianSurfaceCenterExpression());
        if (!center) throw new ExplorationActionError("No plugin surface is open to scroll");
        await client.send("Input.dispatchMouseEvent", { type: "mouseWheel", x: center.x, y: center.y, deltaX: 0, deltaY: action.direction === "down" ? action.pixels : -action.pixels });
      } else if (action.type === "wait") {
        await new Promise((resolve, reject) => {
          let timer;
          const onAbort = () => { clearTimeout(timer); reject(options.signal.reason instanceof Error ? options.signal.reason : new Error("Exploration cancelled during wait")); };
          timer = setTimeout(() => { options.signal?.removeEventListener("abort", onAbort); resolve(); }, action.milliseconds);
          options.signal?.addEventListener("abort", onAbort, { once: true });
        });
      } else if (action.type === "press") {
        if (!PRESS_KEYS.has(action.key)) throw new ExplorationActionError("Press is limited to a small keyboard-navigation allowlist");
        const focused = await read(obsidianFocusedControlExpression(testMode));
        if (!focused?.ok) throw new ExplorationActionError("The focused element is outside the plugin's scope; the key press was not sent");
        // P1 Chunk 1: everything except pure navigation/cancel keys (see
        // PRESS_KEYS_ALWAYS_ALLOWED_ON_FOCUS above) is checked against the same
        // restricted/sensitive/disabled policy revalidateRef() already applies to
        // click/fill/select, before any Input.* event is sent.
        if (!PRESS_KEYS_ALWAYS_ALLOWED_ON_FOCUS.has(action.key)) {
          const control = focused.control;
          const restriction = classifyControlRestriction(control);
          if (restriction) throw new ExplorationActionError(`This control is restricted: ${restriction}`);
          if (control.sensitive) throw new ExplorationActionError("This control is restricted: secret");
          if (control.disabled) throw new ExplorationActionError("This control is no longer interactive (disabled)");
        }
        await dispatchKey(action.key);
      } else if (action.type === "back") {
        // Obsidian's own Menu registers Escape on its keyboard Scope to close itself
        // (resources/obsidian.asar, read-only inspection), the same mechanism a plugin modal's own
        // Escape handling uses, so a single real Escape closes whichever is open.
        const canGoBack = await read(`Boolean(document.querySelector(".arxiv-daily-modal, .arxiv-daily-menu"))`);
        if (!canGoBack) throw new ExplorationActionError("No plugin modal is open to go back from");
        await dispatchKey("Escape");
      } else if (action.type === "reload") {
        throw new ExplorationActionError("Reload is not supported inside Obsidian; the driver resets state between tasks instead");
      } else {
        throw new ExplorationActionError(`Unsupported action for the Obsidian host: ${action.type}`);
      }
      options.signal?.throwIfAborted();
      return { ok: true };
    } catch (error) {
      return { ok: false, error: redact(error.message) };
    }
  }

  return { execute };
}

/**
 * Chunk 4 — host wiring, reset, and first Obsidian tasks. Mirrors
 * exploration-browser.mjs's `createExplorationDriver`/`createDefaultExplorationTasks`,
 * but over a real, isolated Obsidian instead of Playwright. Session startup is
 * not duplicated: `startObsidianExplorationSession` reuses the exact launch/
 * isolation/preflight/trust/build-deploy stack the fixed `obsidian` suite uses
 * (`runDesktopSession`, `preflight`, `installSettingsFixture`,
 * `createObsidianSettings`), and the driver itself builds one
 * `createObsidianDriver` (obsidian-ui.mjs) internally to get command/runDate/
 * jsonFile/exists/idle/screenshot for free — the same helpers
 * obsidian-scenarios.mjs uses — rather than re-implementing them.
 */

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");

export class ObsidianExplorationBlockedError extends Error {
  constructor(message, cause) {
    super(message, cause ? { cause } : undefined);
    this.name = "ObsidianExplorationBlockedError";
  }
}

// Exported so exploration-tasks/obsidian.mjs's goal-task verify() can read the same on-disk paths and
// lookup this driver's reset()/ensureDailyReport()/ensurePaperUnstarred() already use, without a second
// copy of these paths drifting out of sync.
export const SETTINGS_FILE = ".obsidian/plugins/arxiv-daily/data.json";
export const INDEX_FILE = "arxiv-daily/.index/papers.json";
// P3 Chunk 3: the run-state and run-history files the new run tasks (run-for-date, cancel-then-rerun,
// auth-failure-retry, completed-date-idempotent) read, matching obsidian-scenarios.mjs's own STATE/HISTORY
// constants exactly (same on-disk paths the fixed suite already exercises).
export const STATE_FILE = "arxiv-daily/.index/run-state.json";
export const HISTORY_FILE = "arxiv-daily/.index/run-history.jsonl";
export const dailyReportFile = date => `arxiv-daily/daily/${date}.md`;
// P3 Chunk 4: the vault-relative path manualFetch.fetchAndSummarize/ensurePaperNote write a Paper note to
// (papersDir + "/" + arxivId + ".md" — see packages/core/src/services/manual-fetch.ts and
// scripts/acceptance/obsidian.mjs's createObsidianSettings, output.papersDir === "arxiv-daily/papers").
export const paperNoteFile = arxivId => `arxiv-daily/papers/${arxivId}.md`;
export const entryFor = (index, arxivId) => Object.values(index?.papers ?? {}).find(entry => entry.arxivId === arxivId);

/**
 * Adapts `runDesktopSession`'s single bracketing `body` callback (launch →
 * run `body` → teardown, all in one call) into the start-now/stop-later shape
 * `runExploration` already uses for the workbench (`startWorkbenchSession`
 * returns `{ ..., stop }`), without duplicating any launch/isolation/trust/
 * diagnostics logic: `body` resolves the session to the caller and then
 * blocks on an internal promise until `stop()` releases it, at which point
 * `runDesktopSession`'s own teardown (CDP close, process-group reclaim,
 * vault-state restore, sandbox removal) runs exactly as it does for the fixed
 * suite. A failure before `body` is ever reached (preflight, trust dialog,
 * version mismatch, app not usable) rejects the "session ready" promise
 * instead, same as a direct `runDesktopSession` call would throw.
 */
export async function startObsidianExplorationSession({
  fixture, artifactDir, signal,
  obsidianPath = process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian",
  virtualDisplay = true,
  runSession = runDesktopSession,
  check = preflight,
} = {}) {
  signal?.throwIfAborted();
  if (!fixture?.vaultRoot || !fixture?.server?.url) throw new Error("Obsidian exploration needs an owned fixture with a vault and fixture server");
  const sourceDir = path.join(repoRoot, "plugin");
  let environment;
  try { environment = await check({ vaultPath: fixture.vaultRoot, obsidianPath, sourceDir, virtualDisplay }); }
  catch (error) { throw new ObsidianExplorationBlockedError(`Obsidian preflight could not complete: ${error.message}`, error); }
  if (!environment.ok) throw new ObsidianExplorationBlockedError(environment.blockers.map(({ message, remedy }) => `${message}; ${remedy}`).join("\n"));
  if (artifactDir) await fs.mkdir(artifactDir, { recursive: true });

  let resolveReady, rejectReady, settled = false;
  const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
  let releaseBody;
  const bodyReleased = new Promise(resolve => { releaseBody = resolve; });

  const overall = runSession({
    vaultPath: fixture.vaultRoot, pluginId: "arxiv-daily", sourceDir, obsidianPath, virtualDisplay,
    beforeLaunch: ({ vaultPath, pluginId, fs: storage }) => installSettingsFixture({
      vaultPath, pluginId, fs: storage, data: createObsidianSettings({ endpoint: `${fixture.server.url}/v1` }),
    }),
    body: async sessionInfo => {
      settled = true;
      resolveReady(sessionInfo);
      await bodyReleased;
    },
  });
  overall.catch(error => { if (!settled) { settled = true; rejectReady(error); } });

  const onAbort = () => releaseBody();
  signal?.addEventListener("abort", onAbort, { once: true });

  let sessionInfo;
  try { sessionInfo = await ready; }
  catch (error) {
    signal?.removeEventListener("abort", onAbort);
    const blockers = blockersFromError(error);
    throw blockers ? new ObsidianExplorationBlockedError(blockers.map(({ message }) => message).join("\n"), error) : error;
  }

  return {
    ...sessionInfo,
    async stop() {
      signal?.removeEventListener("abort", onAbort);
      releaseBody();
      return overall;
    },
  };
}

/**
 * The Obsidian counterpart of `exploration-browser.mjs`'s
 * `createExplorationDriver`. `session` is the inner CDP session object
 * (`{ evaluate, client, diagnostics, ... }`, as produced by
 * `startObsidianExplorationSession(...).session`), matching the contract's
 * `createObsidianExplorationDriver({ session, fixture, artifactDir, redact,
 * signal })` shape exactly.
 */
export async function createObsidianExplorationDriver({ session, fixture, artifactDir, redact = value => String(value ?? ""), signal } = {}) {
  if (!session?.evaluate || !session?.client) throw new Error("Obsidian exploration driver needs a CDP session");
  if (!fixture?.vaultRoot || !fixture?.server?.url) throw new Error("Obsidian exploration driver needs an owned fixture");
  await fs.mkdir(artifactDir, { recursive: true, mode: 0o700 });
  // One real createObsidianDriver (obsidian-ui.mjs) gives command/runDate/jsonFile/exists/idle/screenshot/
  // openDashboard/closeSettings for free, installs the fixture HTTP proxy, and is the exact same code path the
  // fixed `obsidian` suite exercises — no parallel implementation of any of that here.
  const helper = await createObsidianDriver({ session, fixture, artifactDir });
  const executor = createObsidianActionExecutor({ client: session.client, evaluate: session.evaluate, redact });
  let version = 0, latest, diagnosticsSeen = 0;
  // P3 Chunk 3: the auth-failure-retry goal task deliberately drives the product into the fixed suite's
  // own "controlled authentication failure" path (paper-filter.ts's `logger.error("paper-filter: LLM
  // call failed", e)` on a 401), which without a carve-out would otherwise be classified as an
  // *unexpected* console error by classifyObsidianDiagnostics and fail the task the instant it is
  // observed — before verify() even runs (exploration.mjs's loop checks `observation.errors` before
  // calling verify()). `expectAuthorizationFailure()` lets that one task's `prepare()` opt in to the
  // fixed suite's existing `authorizationFailureExpected` carve-out (obsidian.mjs's
  // classifyObsidianDiagnostics) for its own run only; `reset()` clears the flag back to false before
  // every task (including the next run of the same task across a second full pass), so every other task
  // keeps the strict "no unexpected errors" bar unchanged.
  let authFailureExpected = false;

  async function observe() {
    signal?.throwIfAborted();
    const id = `s${++version}`;
    const snapshot = await helper.read(obsidianObservationExpression(id, false));
    for (const control of snapshot.controls) {
      const restriction = classifyControlRestriction(control);
      if (restriction) { control.restricted = restriction; if (control.sensitive) control.value = ""; }
    }
    // Reuses obsidian-scenarios's own unexpected-console-error/exception classification; the generic
    // carve-out only applies when a task has opted in via expectAuthorizationFailure() (see above),
    // scoped to only the diagnostics entries collected since the previous observation.
    const entries = session.diagnostics.entries();
    const fresh = entries.slice(diagnosticsSeen);
    diagnosticsSeen = entries.length;
    const { unexpected } = classifyObsidianDiagnostics(fresh, authFailureExpected
      ? { authorizationFailureExpected: true, fixtureRequests: fixture.server.requests }
      : undefined);
    snapshot.errors = unexpected.map(entry => redact(entry.text ?? JSON.stringify(entry)));
    // P4 Chunk 2 Fix 3: dedupe repeated <select> option lists (e.g. many identical arXiv-category rows)
    // into one shared observation.optionSets map, in Node rather than inside the in-page/CDP
    // observation expression, so obsidianObservation's own DOM-scanning code stays untouched.
    snapshot.optionSets = collapseOptionSets(snapshot.controls);
    latest = snapshot;
    return snapshot;
  }

  async function execute(action, observation, options = {}) {
    signal?.throwIfAborted();
    if (latest?.id !== observation.id) return { ok: false, error: "Stale observation; capture a new page state" };
    // P4 Chunk 2 Fix 2: `home` is the Obsidian counterpart of the model's only way back to the plugin
    // surface once it has wandered outside it (obsidianObservation offers it from any state, including
    // zero controls) — it goes straight through the driver's own existing openDashboard path (the same
    // one reset() already uses), never through createObsidianActionExecutor's generic CDP click/back/fill
    // machinery, which has no ref or CDP input event to drive for this.
    if (action.type === "home") {
      try { await helper.openDashboard(); return { ok: true }; }
      catch (error) { return { ok: false, error: redact(error.message) }; }
    }
    return executor.execute(action, observation, options);
  }

  async function capture(name) {
    if (!/^[a-z0-9-]+$/.test(name)) throw new Error("Invalid exploration artifact name");
    // P1 Chunk 3 (review Finding 1 "Compounding" fix): no secret field — plain password or revealed
    // via the "Show ... API key" button — may be captured, so every one is covered right before the
    // screenshot and always restored right after, including when the capture itself throws (e.g. the
    // subject is not currently visible), so a secret field's live state is never left different from
    // how this call found it.
    await helper.read(obsidianMaskSecretFieldsExpression());
    try {
      await helper.screenshots.capture(name, { selector: ".workspace" });
    } finally {
      await helper.read(obsidianUnmaskSecretFieldsExpression());
    }
    return path.join(artifactDir, `${name}.png`);
  }

  /**
   * Deterministic start for every task (contract step 3): close any plugin modal (real Escape, reusing the
   * Chunk-3 `back` action so there is exactly one place that knows how to do this), close Obsidian's own
   * settings window, restore the plugin's settings store to the fixture baseline, and reload it the cheapest
   * reliable way found by reading plugin/main.ts: `loadSettingsAndState()` re-reads `data.json` and replaces
   * `this.settings`/run-state/library-connection in memory without the cost of a full disablePlugin/
   * enablePlugin cycle (which also tears down and re-registers the scheduler, views, ribbon and commands).
   * TypeScript `private` is erased at build time (esbuild minifies this build but never mangles property
   * names here — no `mangleProps` is configured), so the method stays callable from the renderer the same way
   * obsidian-ui.mjs's own `reloadPlugin()` already reaches into plugin internals. Vault content written by
   * earlier tasks (daily reports, star marks) is deliberately left alone: resetting it on every task would
   * cost a real Obsidian re-scan of the vault and is not needed for the two tasks in this chunk, each of which
   * reads real, independent state rather than assuming a pristine vault (see exploration-tasks/obsidian.mjs's `goalTasks`).
   * Finally the Reading Dashboard is opened so every task starts observing the same view.
   */
  async function reset({ signal: resetSignal } = {}) {
    const start = Date.now();
    resetSignal?.throwIfAborted();
    authFailureExpected = false; // P3 Chunk 3: never carries over into the next task.
    await executor.execute({ type: "back" }, null, { signal: resetSignal }).catch(() => {});
    await helper.closeSettings();
    resetSignal?.throwIfAborted();
    const settingsPath = path.join(fixture.vaultRoot, SETTINGS_FILE);
    await fs.mkdir(path.dirname(settingsPath), { recursive: true });
    await fs.writeFile(settingsPath, JSON.stringify(createObsidianSettings({ endpoint: `${fixture.server.url}/v1` }), null, 2));
    // Real-host finding: `loadSettingsAndState()` alone (re-reading data.json without a full plugin
    // reload) is NOT reliable — it replaces `plugin.settings` with a new object, but
    // `SettingsChangeService` (constructed once in `onload()`) keeps its own `deps.settings` reference
    // captured at construction time. A later settings-tab edit then persists correctly to disk (the
    // candidate it builds and saves is fresh) but "commits" onto the now-detached old settings object,
    // so the live object the settings tab reads back from (`tab.plugin.settings`) never reflects the
    // edit: confirmed on a real host by reading `app.plugins.plugins["arxiv-daily"].settings.output.
    // maxDailyPapers` immediately after a successful fill — it stayed at the reset baseline. The fixed
    // suite's own `reloadPlugin()` (disablePlugin/enablePlugin, obsidian-ui.mjs) runs the full onload()
    // again and reconstructs `SettingsChangeService` against the correct object, so it is used here
    // instead, at the cost the phase doc anticipated (slower than a bare settings re-read, see below).
    await helper.reloadPlugin();
    resetSignal?.throwIfAborted();
    await helper.openDashboard();
    const durationMs = Date.now() - start;
    if (durationMs > 30000) throw new Error(`Obsidian reset took ${durationMs}ms, exceeding the 30s per-task phase-abort budget`);
    return { durationMs };
  }

  async function dispose() {}

  return {
    observe, execute, capture, reset, dispose,
    // Task-specific extras, exactly like exploration-browser.mjs's inspect()/dailyEvidence(): real vault
    // state (or, for the two below, a program-only UI round trip) a verifier reads/drives directly, never
    // through the model's own action schema.
    jsonFile: helper.jsonFile, exists: helper.exists, file: helper.file, read: helper.read, idle: helper.idle,
    openDashboard: helper.openDashboard, reloadPlugin: helper.reloadPlugin,
    // P3 Chunk 3: the run tasks' verify() toggles the fixture's own LLM mode directly (hold/release,
    // unauthorized/normal) as a program-only side effect, exactly like the fixed suite's obsidian-
    // scenarios.mjs (`d.fixture.server.setMode(...)`) — never something the model itself can reach.
    fixture,
    // P3 Chunk 3: opts the current task's observe() into the fixed suite's existing "expected
    // authentication failure" carve-out (see observe()'s own comment above); only auth-failure-retry's
    // prepare() calls this, and reset() clears it before every task.
    expectAuthorizationFailure() { authFailureExpected = true; },
    // P3 Chunk 3: a task's prepare() calls this (never the model) to deterministically start a date fresh
    // — undoing whatever an earlier pass of the *same* task left behind (run-state entry, daily report
    // file, and that date's entries in the paper index's dailyReports) — so re-running the whole task
    // library a second time in one session still starts each of these run tasks from "not yet generated"
    // rather than silently reusing the first pass's completed state. Paper-index entries are trimmed
    // (the date's report path removed from `dailyReports`), not deleted outright, because the same
    // PAPERS fixture appears under multiple dates (other tasks' reports) and a shared index entry must
    // survive. Finishes with the same full-plugin-reload applySettingsPatch/reset() already rely on, so
    // the in-memory run state/paper index agree with what was just written; leaves no plugin surface
    // open afterward (same caveat as applySettingsPatch), so callers reopen a surface themselves next.
    async clearDateState(date) {
      await fs.rm(path.join(fixture.vaultRoot, dailyReportFile(date)), { force: true });
      const statePath = path.join(fixture.vaultRoot, STATE_FILE);
      try {
        const state = JSON.parse(await fs.readFile(statePath, "utf8"));
        if (state.runState && date in state.runState) {
          delete state.runState[date];
          await fs.writeFile(statePath, JSON.stringify(state, null, 2));
        }
      } catch (error) { if (error.code !== "ENOENT") throw error; }
      const indexPath = path.join(fixture.vaultRoot, INDEX_FILE);
      try {
        const index = JSON.parse(await fs.readFile(indexPath, "utf8"));
        let changed = false;
        for (const entry of Object.values(index.papers ?? {})) {
          if (!Array.isArray(entry.dailyReports)) continue;
          const before = entry.dailyReports.length;
          entry.dailyReports = entry.dailyReports.filter(reportPath => !reportPath.endsWith(`${date}.md`));
          if (entry.dailyReports.length !== before) changed = true;
        }
        if (changed) await fs.writeFile(indexPath, JSON.stringify(index, null, 2));
      } catch (error) { if (error.code !== "ENOENT") throw error; }
      await helper.reloadPlugin();
    },
    async closeThenReopenSettings() { await helper.closeSettings(); await helper.openSettings(); },
    // P3 Chunk 2: proof that a setting survives a full plugin reload, not only re-opening the settings
    // tab inside the same loaded instance (the two are different claims — see reset()'s own comment above
    // about SettingsChangeService staying bound to a detached settings object across a bare re-read).
    async reloadThenReopenSettings() { await helper.reloadPlugin(); await helper.openSettings(); },
    // P3 Chunk 2: a task's deterministic `prepare()` writes directly to data.json (never through the
    // model) and then reloads the plugin so the write actually takes effect in the running instance —
    // the same reload reset() already relies on, reused here instead of a second implementation of it.
    // Leaves no plugin surface open afterward (reloadPlugin tears down and re-registers the dashboard
    // view); callers that need a starting surface call openDashboard()/openSettings() themselves next.
    async applySettingsPatch(mutate) {
      const current = await helper.jsonFile(SETTINGS_FILE);
      mutate(current);
      await fs.writeFile(path.join(fixture.vaultRoot, SETTINGS_FILE), JSON.stringify(current, null, 2));
      await helper.reloadPlugin();
    },
    // P3 Chunk 4: the index counterpart of applySettingsPatch — a task's deterministic `prepare()` seeds or
    // resets paper-index entries directly (status/priority/detail/paperPath/…), never through the model's
    // own actions, then reloads the plugin so the running instance actually reflects the write (same
    // reload reset() and applySettingsPatch already rely on). Leaves no plugin surface open afterward;
    // callers reopen a surface themselves next.
    async applyIndexPatch(mutate) {
      const current = await helper.jsonFile(INDEX_FILE).catch(() => ({ schemaVersion: 5, updatedAt: new Date().toISOString(), papers: {} }));
      current.papers ??= {};
      mutate(current);
      await fs.mkdir(path.dirname(path.join(fixture.vaultRoot, INDEX_FILE)), { recursive: true });
      await fs.writeFile(path.join(fixture.vaultRoot, INDEX_FILE), JSON.stringify(current, null, 2));
      await helper.reloadPlugin();
    },
    // P3 Chunk 4: generic vault file write/remove, used by tasks that seed or clear a Paper note file
    // directly (never through the model) to make a precondition deterministic — e.g. create-paper-note's
    // prepare() clears the target's own note and seeds the fixture's other paper with one so exactly one
    // dashboard row shows the "Summarize by arXiv ID" action the task asks the model to click (row action
    // buttons carry no per-paper identifying attribute, unlike the star toggle — see obsidian.mjs's
    // CREATE_PAPER_NOTE_ID comment). Neither call reloads the plugin by itself; callers that also need an
    // index patch call applyIndexPatch afterward, so one prepare() pays for at most one reload.
    async writeVaultFile(relativePath, content) {
      const target = path.join(fixture.vaultRoot, relativePath);
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.writeFile(target, content);
    },
    async removeVaultFile(relativePath) {
      await fs.rm(path.join(fixture.vaultRoot, relativePath), { force: true });
    },
    async ensureDailyReport(date) {
      if (await helper.exists(dailyReportFile(date))) return;
      await helper.runDate(date);
      await helper.idle();
    },
    async ensurePaperUnstarred(arxivId) {
      await helper.openDashboard();
      await helper.click('.arxiv-daily-dashboard__tab[data-tab="all"]');
      await helper.wait('document.querySelectorAll(".arxiv-daily-dashboard__table tbody tr").length', value => value > 0, "dashboard paper rows");
      const starred = await helper.read(`document.querySelector('.arxiv-daily-dashboard__star[data-arxiv-id="${arxivId}"]')?.getAttribute("aria-pressed") === "true"`);
      if (!starred) return;
      await helper.click(`.arxiv-daily-dashboard__star[data-arxiv-id="${arxivId}"]`);
      await pollJsonFile(() => helper.jsonFile(INDEX_FILE), value => entryFor(value, arxivId)?.priority !== "high", "unstarred priority on disk");
    },
  };
}

