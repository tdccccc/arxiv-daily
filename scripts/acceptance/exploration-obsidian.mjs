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
 *   plugin-scoped root with no plugin-identifying class (confirmed by
 *   reading view.ts:1874-2040) — out of scope/invisible in Chunk 1 rather
 *   than guessed at.
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
 *   outside the current plugin surface, then sent as a real
 *   `Input.dispatchKeyEvent` keyDown/keyUp pair.
 * - back presses Escape only when a plugin modal (`.arxiv-daily-modal`) is
 *   open, closing the top one; otherwise it is rejected. reload is always
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

import { ExplorationActionError, PRESS_KEYS } from "./exploration.mjs";
import { waitForStableTarget } from "./obsidian-ui.mjs";

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

/** Resolves the one surface (modal/settings/dashboard) the model may see, in priority order. */
function resolveSurface() {
  let surface, view = "none", modalOpen = false;
  const markedModal = document.querySelector(".arxiv-daily-modal");
  if (markedModal) { surface = markedModal.closest(".modal") || markedModal; view = "modal"; modalOpen = true; }
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
  return { surface, view, modalOpen };
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
    disabled: node.disabled === true || node.getAttribute("aria-disabled") === "true", readOnly: node.readOnly === true,
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

const CONTROL_SELECTOR = 'button,summary,a[href],input:not([type="hidden"]),textarea,select,[role="button"],[role="tab"],[role="checkbox"],[role="combobox"],[tabindex]';

export function obsidianObservation(id, testMode) {
  const { surface, view, modalOpen } = resolveSurface();
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

  return {
    id, title: document.title,
    text: (surface ? surface.innerText ?? surface.textContent ?? "" : "").slice(0, 16000),
    controls,
    // Chunk 3: which browser-shaped actions this host currently allows, so the model does not have
    // to guess at back/reload support that differs by host and by whether a modal is open. `reload`
    // is never listed — Obsidian has no per-page reload, and resetting state between tasks is the
    // driver's job, not an action a model can request.
    capabilities: ["click", "fill", "select", "scroll", "wait", "press", ...(modalOpen ? ["back"] : [])],
    layout: { rootMounted: Boolean(surface), view, dialogOpen: modalOpen, viewport: { width: innerWidth, height: innerHeight }, notices },
  };
}

/** Builds the CDP `Runtime.evaluate` expression string for `obsidianObservation`, matching obsidian-ui.mjs's `read(expression)` convention (the caller wraps with `JSON.stringify(...)`). */
export function obsidianObservationExpression(id, testMode = false) {
  const shared = [resolveName, controlVisible, resolveSurface, describeControl].map(fn => fn.toString()).join("\n");
  return `(() => { ${shared}\n  const CONTROL_SELECTOR = ${JSON.stringify(CONTROL_SELECTOR)};\n  return (${obsidianObservation})(${JSON.stringify(id)}, ${JSON.stringify(Boolean(testMode))});\n})()`;
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
  return `(() => { ${shared}\n  return (${obsidianLiveControl})(${JSON.stringify(ref)}, ${JSON.stringify(Boolean(testMode))});\n})()`;
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
  const selector = JSON.stringify(`[data-acceptance-ref="${ref}"]`);
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
  if (/^arxiv-daily-run-window-/.test(field) || /^enable\s*·/i.test(name) || /check every \(minutes\)/i.test(name)) return "scheduling settings are fixed by the fixture";
  if (/^api base url$|^better pdf parser$|embedding|sidecar/i.test(name)) return "external service and embedding/parser settings are fixed by the fixture";
  if (/send test|how to send|your email|send verification email|from email|from name|daily auto-send|resend api key|verification code/i.test(name)) return "email delivery is excluded from exploration";
  if (/^(open arxiv|open bug report|report a bug|open feature request|request a feature|open docs|documentation|open repository|repository)$/i.test(name)) return "link leaves the plugin for an external destination";
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
        await evaluate(`(() => { document.querySelector(${JSON.stringify(refSelector(action.ref))})?.scrollIntoView({ block: "center" }); return true; })()`);
        const target = await targetCenter(action.ref, "Obsidian click target stable and hit-testable");
        await dispatchClick(target.x, target.y);
      } else if (action.type === "fill") {
        await evaluate(`(() => { document.querySelector(${JSON.stringify(refSelector(action.ref))})?.scrollIntoView({ block: "center" }); return true; })()`);
        const target = await targetCenter(action.ref, "Obsidian fill target stable and hit-testable");
        await dispatchClick(target.x, target.y);
        await dispatchSelectAll();
        await client.send("Input.insertText", { text: action.value });
        const after = await read(`document.querySelector(${JSON.stringify(refSelector(action.ref))})?.value`);
        if (after !== action.value) throw new ExplorationActionError("Fill did not take effect; the field still shows a different value");
        await dispatchKey("Tab");
      } else if (action.type === "select") {
        const applied = await read(`(() => {
          const node = document.querySelector(${JSON.stringify(refSelector(action.ref))});
          if (!node) return false;
          const option = Array.from(node.options || []).find(item => item.value === ${JSON.stringify(action.value)});
          if (!option || option.disabled) return false;
          node.value = ${JSON.stringify(action.value)};
          const EventCtor = node.ownerDocument.defaultView.Event;
          node.dispatchEvent(new EventCtor("input", { bubbles: true }));
          node.dispatchEvent(new EventCtor("change", { bubbles: true }));
          return node.value === ${JSON.stringify(action.value)};
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
        const scoped = await read(obsidianFocusInScopeExpression());
        if (!scoped) throw new ExplorationActionError("The focused element is outside the plugin's scope; the key press was not sent");
        await dispatchKey(action.key);
      } else if (action.type === "back") {
        const modalOpen = await read(`Boolean(document.querySelector(".arxiv-daily-modal"))`);
        if (!modalOpen) throw new ExplorationActionError("No plugin modal is open to go back from");
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
