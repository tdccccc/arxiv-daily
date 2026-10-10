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
 * - Marked plugin modals (modalEl/contentEl class set by our own code):
 *   HubModal `.arxiv-daily-hub-modal` (hub-modal.ts:29), SimilarPapersModal
 *   `.arxiv-daily-similar-modal` (similar-papers-modal.ts:58),
 *   PersonalLibraryInterestProfileModal `.arxiv-daily-interest-review-modal`
 *   (interest-profile-modal.ts:145), library authorization modal
 *   `.arxiv-daily-library-authorization-modal` (library/modal.ts:14,41).
 * - UNMARKED plugin modals (confirmed by reading — no addClass call at all
 *   on modalEl/contentEl): DatePickerModal (date-picker-modal.ts:45-107),
 *   ArxivIdModal/FullTextQueryModal/FullTextSearchResultsModal/PaperIdModal/
 *   PaperMarkModal/StateModal/RunHistoryModal/DiagnosticsModal/
 *   FullTextRuntimeDiagnosticsModal (commands.ts:656-1034). These are
 *   reported here rather than guessed at with a bespoke marker (per the
 *   chunk-1 contract); the chosen fallback — documented as a known gap, not
 *   a general-purpose guarantee — is: treat any single open generic
 *   Obsidian `.modal` that isn't the Settings modal as in-scope, since the
 *   acceptance fixture vault has only this one plugin installed. See the
 *   "known gap" test in acceptance-exploration-obsidian.test.mjs.
 * - Obsidian's `Menu` (dashboard "More" toolbar menu, view.ts:1874
 *   `showMoreMenu`) renders into a body-level `.menu` element outside any
 *   plugin-scoped root with no plugin-identifying class (confirmed by
 *   reading view.ts:1874-2040) — out of scope/invisible in Chunk 1 rather
 *   than guessed at.
 */

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
 */
export function obsidianObservation(id, testMode) {
  // Obsidian's own `.setting-item`/`.setting-item-name` convention (stable,
  // part of its public Setting API) is the label fallback, since Obsidian's
  // Setting-built inputs carry no `name="..."` attribute (confirmed by
  // reading tab.ts addText() call sites). Nested (not a top-level helper) so
  // this whole function stays self-contained when stringified for CDP.
  const resolveName = node => {
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
  };
  const visible = node => {
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    const sized = testMode ? true : rect.width > 0 && rect.height > 0;
    return sized && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0" && !node.closest("[hidden],[inert]");
  };

  const pluginModalMarkers = [".arxiv-daily-hub-modal", ".arxiv-daily-similar-modal", ".arxiv-daily-interest-review-modal", ".arxiv-daily-library-authorization-modal"];
  let surface, view = "none", modalOpen = false;
  for (const marker of pluginModalMarkers) {
    const found = document.querySelector(marker);
    if (found) { surface = found.closest(".modal") || found; view = "modal"; modalOpen = true; break; }
  }
  if (!surface) {
    // Documented gap (see file header): the only distinguishing signal left
    // for an unmarked plugin modal is "it's the one modal open," which only
    // holds in a single-plugin fixture vault.
    const generic = document.querySelector(".modal-container .modal:not(.mod-settings)");
    if (generic) { surface = generic; view = "unmarked-modal"; modalOpen = true; }
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

  const controls = [];
  if (surface) {
    for (const node of surface.querySelectorAll('button,summary,a[href],input:not([type="hidden"]),textarea,select,[role="button"],[role="tab"],[role="checkbox"],[role="combobox"],[tabindex]')) {
      if (!visible(node) || controls.length >= 220) continue;
      const rect = node.getBoundingClientRect();
      const name = resolveName(node).slice(0, 180);
      const field = node.id || node.getAttribute("name") || "";
      const hasRevealSibling = Boolean(node.parentElement?.querySelector(".arxiv-daily-settings__reveal-key"));
      const sensitive = node.type === "password" || hasRevealSibling || /api.?key|password|secret|token|verification code/i.test(name);
      const ref = `${id}-${controls.length + 1}`;
      node.setAttribute("data-acceptance-ref", ref);
      controls.push({
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
      });
    }
  }

  // Obsidian's `.notice` is a generic class (no plugin marker); surfaced as
  // text-only context per the chunk-1 contract, never as a control.
  const notices = Array.from(document.querySelectorAll(".notice")).map(node => (node.textContent || "").trim()).filter(Boolean).slice(0, 20);

  return {
    id, title: document.title,
    text: (surface ? surface.innerText ?? surface.textContent ?? "" : "").slice(0, 16000),
    controls,
    layout: { rootMounted: Boolean(surface), view, dialogOpen: modalOpen, viewport: { width: innerWidth, height: innerHeight }, notices },
  };
}

/** Builds the CDP `Runtime.evaluate` expression string for `obsidianObservation`, matching obsidian-ui.mjs's `read(expression)` convention (the caller wraps with `JSON.stringify(...)`). */
export function obsidianObservationExpression(id, testMode = false) {
  return `(${obsidianObservation})(${JSON.stringify(id)}, ${JSON.stringify(Boolean(testMode))})`;
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
 * report.
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
