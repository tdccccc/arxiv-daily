import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import zlib from "node:zlib";
import { Window } from "happy-dom";
import { evaluate as cdpEvaluate } from "../desktop-acceptance/cdp.mjs";
import {
  classifyControlRestriction,
  createObsidianActionExecutor,
  createObsidianExplorationDriver,
  obsidianMaskSecretFieldsExpression,
  obsidianObservation,
  obsidianObservationExpression,
  obsidianUnmaskSecretFieldsExpression,
} from "../acceptance/exploration-obsidian.mjs";

/**
 * Obsidian chrome that must never produce controls, in every scenario below:
 * ribbon icons, the file explorer, the status bar, and another community
 * plugin's own floating settings-nav items. These sit as always-present
 * siblings outside any arxiv-daily-marked surface.
 */
const CHROME = `
  <div class="workspace-ribbon"><div class="side-dock-ribbon-action" aria-label="Open quick switcher"></div></div>
  <div class="nav-files-container"><div class="nav-file" draggable="true"><div class="nav-file-title" tabindex="0">note.md</div></div></div>
  <div class="status-bar"><div class="status-bar-item" tabindex="0">0 words</div></div>
`;

// happy-dom's getBoundingClientRect always returns zeroes, so testMode
// defaults on here (bypassing the geometry gate, not the CSS-visibility
// gate) unless a test is specifically exercising that gate itself.
function buildWindow(bodyHtml, { appOverrides, testMode = true, innerWidth = 1200, innerHeight = 800 } = {}) {
  const window = new Window();
  window.document.body.innerHTML = CHROME + bodyHtml;
  const context = {
    document: window.document,
    getComputedStyle: (...args) => window.getComputedStyle(...args),
    innerWidth, innerHeight,
    app: appOverrides,
  };
  // vm.runInNewContext returns cross-realm objects (a different Array/Object
  // prototype than this file's), which trips assert.deepStrictEqual's exotic-
  // object check; JSON round-tripping also matches how the real CDP path
  // actually crosses the renderer boundary (obsidian-ui.mjs's `read()`).
  const observe = (id = "s1") => JSON.parse(JSON.stringify(vm.runInNewContext(obsidianObservationExpression(id, testMode), context)));
  return { window, observe, context };
}

test("no plugin surface open: chrome produces zero controls and layout.view is none", () => {
  const { observe } = buildWindow("");
  const snapshot = observe();
  assert.deepEqual(snapshot.controls, []);
  assert.equal(snapshot.layout.view, "none");
  assert.equal(snapshot.layout.rootMounted, false);
});

// P4 Chunk 2 Fix 2: with zero controls and no explanation, a model that wandered outside the plugin
// (e.g. opened a plain note) could only end the task — real P4 Chunk 1 evidence showed the first-time-user
// persona dead-ending exactly this way. `home` is always advertised as a recovery action, and the
// observation text must say the model is outside the plugin and how to get back, instead of being empty.
test("no plugin surface open: text explains the model is outside the plugin and how to return, and home is always offered", () => {
  const { observe } = buildWindow("");
  const snapshot = observe();
  assert.ok(snapshot.capabilities.includes("home"), "home must be offered even with zero controls");
  assert.ok(snapshot.text.length > 0, "text must not be empty when there is nothing else to tell the model");
  assert.match(snapshot.text, /plugin/i);
  assert.match(snapshot.text, /home/i);
});

test("home is always advertised, including when a normal plugin surface (the dashboard) is open", () => {
  const { observe } = buildWindow(dashboardFixture);
  const snapshot = observe();
  assert.ok(snapshot.capabilities.includes("home"));
});

// P4 Chunk 2 Fix 2 (real-Obsidian finding): a real host run showed a second, narrower dead-end this fix's
// first pass missed — after opening a note from the dashboard calendar, Obsidian kept the dashboard
// leaf's own `.arxiv-daily-dashboard` element mounted (so resolveSurface() still reports
// layout.view:"dashboard", not "none"), but every control inside it had become invisible (the leaf itself
// is backgrounded), so `controls` is still empty. The recovery text must trigger on "zero controls",
// not only on "no surface found at all" — otherwise a model stuck in exactly this real shape gets no
// guidance (home is already unconditionally offered either way).
test("a plugin surface is nominally matched (layout.view stays e.g. dashboard) but has zero currently-visible controls: the same recovery guidance still appears", () => {
  const html = `<div class="view-content arxiv-daily-dashboard"><button type="button">Hidden by real geometry</button></div>`;
  const { observe } = buildWindow(html, { testMode: false }); // testMode:false -> happy-dom's real (always-zero) getBoundingClientRect hides it
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "dashboard", "the surface is still nominally matched");
  assert.deepEqual(snapshot.controls, []);
  assert.ok(snapshot.capabilities.includes("home"));
  assert.match(snapshot.text, /home/i, "guidance must still appear even though a surface was nominally found");
});

function settingsFixture(markWithClass) {
  const containerClass = markWithClass ? "vertical-tab-content arxiv-daily-settings" : "vertical-tab-content";
  return `
    <div class="modal mod-settings">
      <div class="vertical-tab-nav">
        <div class="vertical-tab-nav-item" tabindex="0">Community plugins</div>
        <div class="vertical-tab-nav-item is-active" tabindex="0">arXiv Daily</div>
      </div>
      <div class="vertical-tab-content-container">
        <div class="vertical-tab-content other-plugin-tab" style="display:none">
          <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Other plugin field</div></div>
            <div class="setting-item-control"><input type="text" value="noise"></div></div>
        </div>
        <div class="${containerClass}" id="our-settings-container">
          <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Enable · Paused</div></div>
            <div class="setting-item-control"><input type="checkbox" role="checkbox"></div></div>
          <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">API key</div></div>
            <div class="setting-item-control"><input type="password" class="arxiv-daily-settings__llm-input" aria-label="LLM API key"><button type="button" class="arxiv-daily-settings__reveal-key" aria-label="Show LLM API key"></button></div></div>
          <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">API base URL</div></div>
            <div class="setting-item-control"><input type="text" value="https://api.deepseek.com/v1"></div></div>
          <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Daily paper limit</div></div>
            <div class="setting-item-control"><input type="number" value="20"></div></div>
          <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Library</div></div>
            <div class="setting-item-control"><button type="button">Choose folder</button></div></div>
          <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Send test email</div></div>
            <div class="setting-item-control"><button type="button" class="mod-cta">Send test</button></div></div>
        </div>
      </div>
    </div>
  `;
}

test("settings (legacy class marker): only our visible tab's controls are observed, labelled and classified", () => {
  const { observe } = buildWindow(settingsFixture(true), { appOverrides: { setting: { isOpen: true, activeTab: { id: "arxiv-daily" } } } });
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "settings");
  const names = snapshot.controls.map(c => c.name);
  assert.ok(!names.includes("Other plugin field"), "the other plugin's hidden tab must not surface controls");
  const key = snapshot.controls.find(c => c.name === "LLM API key");
  assert.equal(key.sensitive, true);
  assert.equal(classifyControlRestriction(key), "credentials are excluded from exploration");
  const baseUrl = snapshot.controls.find(c => c.name === "API base URL");
  assert.equal(classifyControlRestriction(baseUrl), "external service and embedding/parser settings are fixed by the fixture");
  const limit = snapshot.controls.find(c => c.name === "Daily paper limit");
  assert.equal(classifyControlRestriction(limit), undefined, "the editable default-task field must stay unrestricted");
  const enableToggle = snapshot.controls.find(c => c.name === "Enable · Paused");
  assert.match(classifyControlRestriction(enableToggle), /scheduling/);
  const chooseFolder = snapshot.controls.find(c => c.name === "Choose folder");
  assert.match(classifyControlRestriction(chooseFolder), /folder/);
  const sendTest = snapshot.controls.find(c => c.name === "Send test");
  assert.match(classifyControlRestriction(sendTest), /email/);
});

test("settings (object-graph marker only, no class — the 1.13+ declarative path): our tab is still found via app.setting.activeTab", () => {
  // No ".arxiv-daily-settings" class anywhere here (settingsFixture(false)) — the only
  // way to find our tab's container is Obsidian's own object graph.
  const { window } = buildWindow(settingsFixture(false));
  const ourContainer = window.document.getElementById("our-settings-container");
  const context = { document: window.document, getComputedStyle: (...a) => window.getComputedStyle(...a), innerWidth: 1200, innerHeight: 800,
    app: { setting: { isOpen: true, activeTab: { id: "arxiv-daily", containerEl: ourContainer } } } };
  const snapshot = JSON.parse(JSON.stringify(vm.runInNewContext(obsidianObservationExpression("s1", true), context)));
  assert.equal(snapshot.layout.view, "settings");
  assert.ok(snapshot.controls.some(c => c.name === "LLM API key"));
  assert.ok(snapshot.controls.some(c => c.name === "Daily paper limit"));
  assert.ok(!snapshot.controls.some(c => c.name === "Other plugin field"));
});

const dashboardFixture = `
  <div class="workspace-leaf mod-active">
    <div class="view-content arxiv-daily-dashboard">
      <div class="arxiv-daily-dashboard__tab is-active" role="tab" tabindex="0">All</div>
      <div class="arxiv-daily-dashboard__tab" role="tab" tabindex="0">Starred</div>
      <table><tr>
        <td><button type="button" class="clickable-icon arxiv-daily-dashboard__star" data-arxiv-id="2610.10001" aria-label="Star"></button></td>
        <td><button type="button" class="clickable-icon arxiv-daily-dashboard__action" aria-label="Open arXiv"></button></td>
        <td><button type="button" class="clickable-icon arxiv-daily-dashboard__action" aria-label="Open PDF"></button></td>
      </tr></table>
    </div>
  </div>
`;

test("reading dashboard: paper row star and guarded resource opens are operable, chrome stays excluded", () => {
  const { observe } = buildWindow(dashboardFixture);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "dashboard");
  assert.equal(snapshot.layout.rootMounted, true);
  const star = snapshot.controls.find(c => c.name === "Star");
  assert.equal(star.paperKey, "2610.10001");
  assert.equal(classifyControlRestriction(star), undefined);
  const openArxiv = snapshot.controls.find(c => c.name === "Open arXiv");
  assert.equal(classifyControlRestriction(openArxiv), undefined, "the renderer guard records external opens without forwarding them");
  const openPdf = snapshot.controls.find(c => c.name === "Open PDF");
  assert.equal(classifyControlRestriction(openPdf), undefined, "Open PDF can resolve to a local vault file and is not unconditionally external");
  assert.ok(!snapshot.controls.some(c => c.name === "Open quick switcher"), "ribbon chrome must stay invisible even when the dashboard is open");
});

test("a plugin modal carrying only the shared arxiv-daily-modal marker (e.g. the date picker) is in scope", () => {
  // Chunk 2: every plugin modal now carries the shared `.arxiv-daily-modal`
  // marker on modalEl (plugin/src/plugin-modal.ts), so this no longer relies
  // on being the only open modal.
  const modal = `
    <div class="modal-container">
      <div class="modal arxiv-daily-modal">
        <h2>Run arXiv Daily for date</h2>
        <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Date</div></div>
          <div class="setting-item-control"><input type="date" value="2026-10-01"></div></div>
        <button type="button" class="mod-cta">Run</button>
      </div>
    </div>
  `;
  const { observe } = buildWindow(modal);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "modal");
  assert.ok(snapshot.controls.some(c => c.tag === "input" && c.type === "date"));
  assert.ok(snapshot.controls.some(c => c.name === "Run"));
  assert.ok(snapshot.capabilities.includes("back"), "a plugin modal is open, so back (Escape) is advertised as supported");
});

test("capabilities never advertise back when no plugin modal is open, and never advertise reload", () => {
  const { observe } = buildWindow(dashboardFixture);
  const snapshot = observe();
  assert.ok(!snapshot.capabilities.includes("back"));
  assert.ok(!snapshot.capabilities.includes("reload"));
});

test("a marked plugin modal wins over a simultaneously open settings modal", () => {
  const combined = `
    <div class="modal mod-settings"><div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Settings noise</div></div>
      <div class="setting-item-control"><button type="button">Settings button</button></div></div></div>
    <div class="modal arxiv-daily-modal arxiv-daily-hub-modal"><div class="arxiv-daily-hub-modal__content">
      <button type="button" aria-label="Close hub">x</button>
    </div></div>
  `;
  const { observe } = buildWindow(combined);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "modal");
  assert.ok(snapshot.controls.some(c => c.name === "Close hub"));
  assert.ok(!snapshot.controls.some(c => c.name === "Settings button"));
});

test("known gap closed: an unmarked (non-plugin) modal is invisible, with no fallback to the only open modal", () => {
  const nonPluginModal = `
    <div class="modal-container"><div class="modal">
      <button type="button">Some other plugin's button</button>
    </div></div>
  `;
  const { observe } = buildWindow(nonPluginModal);
  const snapshot = observe();
  // Chunk 1's "single open non-settings modal" fallback is removed in Chunk 2
  // now that every plugin modal carries the shared marker: an unmarked modal
  // is, by construction, not ours, and must not surface any controls.
  assert.equal(snapshot.layout.view, "none");
  assert.equal(snapshot.layout.rootMounted, false);
  assert.deepEqual(snapshot.controls, []);
});

// Exploration safety and coverage P3 Chunk 7 (phases/03-obsidian-library-tasks.md): with a confirmation
// `.arxiv-daily-modal` (e.g. `chooseModal`) stacked on top of an already-open plugin modal (e.g. the
// personal-library review dialog), `resolveSurface()` used to resolve `document.querySelector(
// ".arxiv-daily-modal")` — the FIRST such element in DOM order, i.e. the older, underlying modal — so the
// newer confirmation dialog's own buttons were never observed and a discard-style flow could never be
// completed. Obsidian's `Modal.open()` appends a newly opened modal's container as the last child of
// `document.body`, so the topmost (most recently opened) modal is the LAST `.arxiv-daily-modal` in DOM
// order, not the first.
test("two stacked plugin modals: the topmost (last-opened) modal's controls are observed, and the older modal's controls are excluded", () => {
  const stacked = `
    <div class="modal-container"><div class="modal arxiv-daily-modal">
      <button type="button">Older modal button</button>
    </div></div>
    <div class="modal-container"><div class="modal arxiv-daily-modal">
      <button type="button">Confirm</button>
      <button type="button">Cancel</button>
    </div></div>
  `;
  const { observe } = buildWindow(stacked);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "modal");
  assert.ok(snapshot.controls.some(c => c.name === "Confirm"));
  assert.ok(snapshot.controls.some(c => c.name === "Cancel"));
  assert.ok(!snapshot.controls.some(c => c.name === "Older modal button"), "the older, covered modal's controls must not be observed");
});

test("two stacked plugin modals: live re-validation of a ref from the now-covered older modal reports it out of scope, not ok", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="modal-container"><div class="modal arxiv-daily-modal"><button type="button">Older modal button</button></div></div>');
  const firstSnapshot = observe();
  const olderRef = firstSnapshot.controls.find(c => c.name === "Older modal button").ref;
  // A second modal opens on top (same DOM-append-order convention Obsidian itself uses), covering the first.
  const secondModal = window.document.createElement("div");
  secondModal.className = "modal-container";
  secondModal.innerHTML = '<div class="modal arxiv-daily-modal"><button type="button">Confirm</button></div>';
  window.document.body.appendChild(secondModal);
  const result = await execute({ type: "click", ref: olderRef }, firstSnapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /scope/i);
  assert.equal(calls.length, 0);
});

test("two stacked plugin modals: press is rejected as out of scope when focus is still inside the now-covered older modal", async () => {
  const { window, calls, execute } = buildSession('<div class="modal-container"><div class="modal arxiv-daily-modal"><button type="button">Older modal button</button></div></div>');
  window.document.querySelector(".arxiv-daily-modal button").focus();
  const secondModal = window.document.createElement("div");
  secondModal.className = "modal-container";
  secondModal.innerHTML = '<div class="modal arxiv-daily-modal"><button type="button">Confirm</button></div>';
  window.document.body.appendChild(secondModal);
  const result = await execute({ type: "press", key: "Enter" }, { id: "s1" });
  assert.equal(result.ok, false);
  assert.match(result.error, /scope/i);
  assert.equal(calls.length, 0);
});

// Exploration safety and coverage P3 Chunk 7: `controlVisible()` only checked a node's own computed style
// and geometry, never whether an ancestor `<details>` was closed — a closed `<details>` hides every
// descendant except its own `<summary>` (native browser behaviour), so controls nested inside a still-
// collapsed section (e.g. the personal-library review dialog's per-topic/per-candidate `<details>`) were
// wrongly reported as present and clickable; a real-host run confirmed clicking one times out hit-testing
// (safe, but wastes model steps) instead of simply never being offered.
test("closed <details>: content is hidden except the summary; opening it reveals the content", () => {
  const html = `
    <div class="arxiv-daily-dashboard">
      <details>
        <summary>Topic heading</summary>
        <button type="button">Hidden while collapsed</button>
      </details>
    </div>
  `;
  const { window, observe } = buildWindow(html);
  const closedSnapshot = observe("s1");
  assert.ok(closedSnapshot.controls.some(c => c.name === "Topic heading"), "the summary itself must stay observable even while collapsed");
  assert.ok(!closedSnapshot.controls.some(c => c.name === "Hidden while collapsed"), "content inside a closed details must not be observed");
  window.document.querySelector("details").open = true;
  const openSnapshot = observe("s2");
  assert.ok(openSnapshot.controls.some(c => c.name === "Hidden while collapsed"), "content inside an opened details must be observed");
});

test("nested closed <details>: a control stays hidden if any ancestor details is closed, even when its own immediate details is open", () => {
  const html = `
    <div class="arxiv-daily-dashboard">
      <details>
        <summary>Outer (closed)</summary>
        <details open>
          <summary>Inner (open)</summary>
          <button type="button">Nested content</button>
        </details>
      </details>
    </div>
  `;
  const { observe } = buildWindow(html);
  const snapshot = observe();
  assert.ok(snapshot.controls.some(c => c.name === "Outer (closed)"));
  assert.ok(!snapshot.controls.some(c => c.name === "Inner (open)"), "the inner summary is inside the still-closed outer details and must stay hidden");
  assert.ok(!snapshot.controls.some(c => c.name === "Nested content"));
});

// P5 Chunk 2 (phases/05-real-model-release-run.md, real-model finding from library-edit-candidate):
// an observed `<summary>` carried no open/closed state at all, so the model could not tell an
// already-open review section from a closed one and repeatedly re-collapsed sections it had just
// opened (wasting most of a 20-step budget). `describeControl` must report the parent `<details>`'s
// own `open` state on its `<summary>` control, the same way `checked`/`value` are already reported
// for other stateful controls.
test("a <summary> control reports its parent <details> open state as `expanded`, so the model can tell an already-open section from a closed one", () => {
  const html = `
    <div class="arxiv-daily-dashboard">
      <details open>
        <summary>Open topic</summary>
        <button type="button">Inside open</button>
      </details>
      <details>
        <summary>Closed topic</summary>
      </details>
    </div>
  `;
  const { observe } = buildWindow(html);
  const snapshot = observe();
  const open = snapshot.controls.find(c => c.name === "Open topic");
  const closed = snapshot.controls.find(c => c.name === "Closed topic");
  assert.equal(open.expanded, true, "an already-open section's summary must report expanded:true");
  assert.equal(closed.expanded, false, "a collapsed section's summary must report expanded:false");
});

test("closed <details>: live re-validation rejects a ref that was observed while open but has since collapsed", async () => {
  const { window, observe, execute, calls } = buildSession(`
    <div class="arxiv-daily-dashboard">
      <details open>
        <summary>Topic heading</summary>
        <button type="button">Was visible</button>
      </details>
    </div>
  `);
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Was visible").ref;
  window.document.querySelector("details").open = false;
  const result = await execute({ type: "click", ref }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /hidden|disappeared|scope/i);
  assert.equal(calls.length, 0);
});

test("refs are unique per observation and tagged onto the DOM for re-finding", () => {
  const { window, observe } = buildWindow(dashboardFixture);
  const snapshot = observe("sref");
  const refs = snapshot.controls.map(c => c.ref);
  assert.equal(new Set(refs).size, refs.length);
  for (const ref of refs) assert.ok(window.document.querySelector(`[data-acceptance-ref="${ref}"]`));
});

test("a file input is restricted and a .notice is surfaced as text only, never as a control", () => {
  const html = `
    <div class="view-content arxiv-daily-dashboard">
      <input type="file">
    </div>
    <div class="notice">arXiv Daily: test email sent</div>
  `;
  const { observe } = buildWindow(html);
  const snapshot = observe();
  const file = snapshot.controls.find(c => c.type === "file");
  assert.equal(classifyControlRestriction(file), "file selection is outside the Obsidian capability");
  assert.ok(!snapshot.controls.some(c => c.tag === "div" && (c.name || "").includes("test email sent")));
  assert.ok(snapshot.layout.notices.some(n => n.includes("test email sent")));
});

test("test mode treats zero-geometry elements as visible but still honours display:none/[hidden]", () => {
  const html = `
    <div class="view-content arxiv-daily-dashboard">
      <button type="button">Zero rect but visible</button>
      <button type="button" style="display:none">Hidden by style</button>
      <button type="button" hidden>Hidden attribute</button>
    </div>
  `;
  const { observe } = buildWindow(html, { testMode: true });
  const snapshot = observe();
  const names = snapshot.controls.map(c => c.name);
  assert.ok(names.includes("Zero rect but visible"));
  assert.ok(!names.includes("Hidden by style"));
  assert.ok(!names.includes("Hidden attribute"));
  // Without testMode, happy-dom's always-zero getBoundingClientRect excludes
  // everything — proving the parameter (not some other code path) is what
  // makes the rest of this suite's fixtures visible.
  const { observe: observeProd } = buildWindow(html, { testMode: false });
  assert.deepEqual(observeProd().controls, []);
});

test("observation and control shape matches the workbench driver's contract", () => {
  const { observe } = buildWindow(dashboardFixture);
  const snapshot = observe("shape");
  assert.deepEqual(Object.keys(snapshot).sort(), ["capabilities", "controls", "focus", "id", "layout", "text", "title"].sort());
  assert.deepEqual(snapshot.capabilities, ["click", "fill", "select", "scroll", "wait", "press", "home"]);
  for (const control of snapshot.controls) {
    for (const key of ["ref", "tag", "type", "role", "name", "field", "disabled", "readOnly", "dataAction", "paperKey", "purpose", "rect", "inViewport"]) {
      assert.ok(key in control, `control is missing ${key}`);
    }
  }
  assert.ok("viewport" in snapshot.layout && "width" in snapshot.layout.viewport && "height" in snapshot.layout.viewport);
});

test("the exported classifier is a pure function usable without any DOM", () => {
  assert.equal(classifyControlRestriction({ type: "password", name: "x" }), "credentials are excluded from exploration");
  assert.equal(classifyControlRestriction({ type: "text", name: "Timezone" }), undefined);
  assert.equal(classifyControlRestriction({ type: "text", name: "", href: "https://example.com/x" }), "link leaves the plugin for an external destination");
  assert.equal(classifyControlRestriction({ type: "text", name: "", href: "obsidian://open?vault=x" }), undefined);
});

// New-P3 Chunk 1 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/03-obsidian-library-
// tasks.md), owner steer: the `embedding` clause in classifyControlRestriction was meant to fix only
// the endpoint/credential/model/dimension sub-fields (and the sidecar sub-fields), not the mode choice
// itself — but as an unanchored substring match it also caught the bare "Embedding" mode <select>
// (packages/core/src/settings/schema.ts's embeddingMode business setting, name: "Embedding", mirrored
// verbatim by the legacy <1.13 display() path at plugin/src/settings/tab.ts:1889), the one control the
// library-remote-consent task needs the model to actually switch. Every name below is read verbatim
// from schema.ts's `fields` map AND tab.ts's own display() (both render the identical literal string —
// confirmed by reading both), so this fixes the policy without guessing at a name that might not be
// what either rendering path actually produces.
test("the Embedding mode select itself must not be restricted — only its endpoint/credential/model/dimension sub-fields (and sidecar, API base URL, Better PDF parser) stay restricted", () => {
  // The one control this policy must newly allow.
  assert.equal(classifyControlRestriction({ type: "", name: "Embedding" }), undefined, "the bare Embedding mode dropdown must be allowed so the model can actually switch it");

  // Every embedding sub-field (schema.ts: embeddingBaseUrl/embeddingModel/embeddingDimension;
  // tab.ts:1912,1944,1963) must stay restricted under the same message as before.
  for (const name of ["Embedding API base URL", "Embedding model", "Embedding dimension"]) {
    assert.equal(classifyControlRestriction({ type: "", name }), "external service and embedding/parser settings are fixed by the fixture", `${name} must stay restricted`);
  }
  // "Embedding API key" (tab.ts:1932) and its own "Show Embedding API key" reveal button are both
  // independently caught by the earlier credentials rule (an "API key" substring, checked before the
  // embedding/sidecar clause is ever reached) — restricted regardless of this change, under the
  // credentials message rather than the embedding/sidecar one.
  for (const name of ["Embedding API key", "Show Embedding API key"]) {
    assert.equal(classifyControlRestriction({ type: "", name }), "credentials are excluded from exploration", `${name} must stay restricted`);
  }

  // The sidecar sub-fields (schema.ts: sidecarCapabilitiesUrl/sidecarParseUrl; tab.ts:1994,2014) must
  // stay restricted. "Better PDF parser" (sidecarEnabled, tab.ts:1979) was already anchored exactly and
  // is unaffected by this change — checked here only as a sanity guard against regressing it.
  for (const name of ["Sidecar capability URL", "Sidecar parse URL", "Better PDF parser"]) {
    assert.equal(classifyControlRestriction({ type: "", name }), "external service and embedding/parser settings are fixed by the fixture", `${name} must stay restricted`);
  }
  // The LLM's own "API base URL" (schema.ts: apiBaseUrl; tab.ts:1456) was already anchored exactly and
  // is untouched by this change — checked here only as a sanity guard.
  assert.equal(classifyControlRestriction({ type: "", name: "API base URL" }), "external service and embedding/parser settings are fixed by the fixture");

  // A name that merely starts with "Embedding" but is not one of the four known sub-fields must not
  // become restricted by accident (guards against an overly broad fix, e.g. a bare prefix match).
  assert.equal(classifyControlRestriction({ type: "", name: "Embedding" }), undefined);
});

// P5 Chunk 2 (phases/05-real-model-release-run.md, real-model finding carried over from the P3/P4
// "title wins over text" note): `resolveName`'s own title-before-own-text order meant an "Add"/
// "Generate" button whose `title` tooltip ("Add paper to library"/"Generate daily report now")
// differed from its rendered text was always named by the tooltip, so distinct buttons in the same
// row were indistinguishable by name to the model. Mirrors the workbench's cf2c8ad fix: aria-label
// first, then the element's own text, then title, then the existing row/field fallbacks.
test("a button's own text wins over its title tooltip, matching the workbench's cf2c8ad fix", () => {
  const html = `
    <div class="view-content arxiv-daily-dashboard">
      <button type="button" title="Generate daily report now">Generate</button>
    </div>
  `;
  const { observe } = buildWindow(html);
  const snapshot = observe();
  assert.ok(snapshot.controls.some(c => c.name === "Generate"), "the button's own visible text must be the observed name");
  assert.ok(!snapshot.controls.some(c => c.name === "Generate daily report now"), "the title tooltip must not win over the button's own text");
});

test("aria-label still wins over a button's own text and its title", () => {
  const html = `
    <div class="view-content arxiv-daily-dashboard">
      <button type="button" aria-label="Add paper to library" title="Add">Add</button>
    </div>
  `;
  const { observe } = buildWindow(html);
  const snapshot = observe();
  assert.ok(snapshot.controls.some(c => c.name === "Add paper to library"));
});

test("title is still used as a fallback name when a control has no own text (e.g. an icon-only button)", () => {
  const html = `
    <div class="view-content arxiv-daily-dashboard">
      <button type="button" class="clickable-icon" title="Open PDF"></button>
    </div>
  `;
  const { observe } = buildWindow(html);
  const snapshot = observe();
  assert.ok(snapshot.controls.some(c => c.name === "Open PDF"));
});

/**
 * Chunk 3 — createObsidianActionExecutor. A fake CDP client records every
 * `send(method, params)` call and answers `Runtime.evaluate` by running the
 * expression against the same happy-dom document the rest of this file
 * uses, through the real `cdp.mjs` `evaluate()` so the executor is exercised
 * exactly as it is over a real CDP client (same `returnByValue`/
 * `exceptionDetails` handling). `Input.insertText` and the `mousePressed`
 * half of `Input.dispatchMouseEvent` get minimal realistic side effects
 * (set the focused element's value; focus whatever `elementFromPoint`
 * reports), mirroring what a real renderer does for those two commands —
 * every other CDP call is pure bookkeeping.
 */
function buildSession(bodyHtml, { appOverrides, testMode = true, stability = { timeoutMs: 150, intervalMs: 2 } } = {}) {
  const window = new Window();
  window.document.body.innerHTML = CHROME + bodyHtml;
  const context = { document: window.document, getComputedStyle: (...args) => window.getComputedStyle(...args), innerWidth: 1200, innerHeight: 800, app: appOverrides };
  const calls = [];
  const client = {
    calls,
    async send(method, params = {}) {
      if (method === "Runtime.evaluate") {
        let value;
        try { value = vm.runInNewContext(params.expression, context); }
        catch (error) { return { result: {}, exceptionDetails: { text: "Uncaught", exception: { description: error.message } } }; }
        return { result: { value } };
      }
      calls.push({ method, params });
      if (method === "Input.insertText") {
        const active = window.document.activeElement;
        if (active && "value" in active) { active.value = params.text; active.dispatchEvent(new window.Event("input", { bubbles: true })); }
      }
      if (method === "Input.dispatchMouseEvent" && params.type === "mousePressed") window.document.elementFromPoint(params.x, params.y)?.focus?.();
      return {};
    },
  };
  const evaluate = expression => cdpEvaluate(client, expression);
  const observe = (id = "s1") => JSON.parse(JSON.stringify(vm.runInNewContext(obsidianObservationExpression(id, testMode), context)));
  const { execute } = createObsidianActionExecutor({ client, evaluate, testMode, stability });
  return { window, calls, observe, execute };
}

/** Stubs a found control's geometry and makes it the sole hit-test target, as acceptance-obsidian.test.mjs's dateModalFixture does for the date picker's submit button. */
function stubTarget(window, ref, rect) {
  const node = window.document.querySelector(`[data-acceptance-ref="${ref}"]`);
  node.getBoundingClientRect = () => rect;
  window.document.elementFromPoint = () => node;
  return node;
}

test("click scrolls into view, waits for a stable hit-tested target, then sends real moved/pressed/released at its center", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Generate").ref;
  stubTarget(window, ref, { x: 350, y: 240, width: 100, height: 40 });
  const result = await execute({ type: "click", ref }, snapshot);
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(c => c.method), ["Input.dispatchMouseEvent", "Input.dispatchMouseEvent", "Input.dispatchMouseEvent"]);
  assert.deepEqual(calls.map(c => c.params.type), ["mouseMoved", "mousePressed", "mouseReleased"]);
  for (const call of calls) { assert.equal(call.params.x, 400); assert.equal(call.params.y, 260); }
});

test("click is rejected with zero Input.* calls when the ref has gone stale", async () => {
  const { calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button></div>');
  const snapshot = observe();
  const result = await execute({ type: "click", ref: "s1-999" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /disappeared|scope/i);
  assert.equal(calls.length, 0);
});

test("click is rejected with zero Input.* calls when the ref is tagged but has left the plugin's current surface", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button></div>');
  const snapshot = observe();
  window.document.querySelector(".side-dock-ribbon-action").setAttribute("data-acceptance-ref", "outside-ref");
  const result = await execute({ type: "click", ref: "outside-ref" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /disappeared|scope/i);
  assert.equal(calls.length, 0);
});

test("click is rejected with zero Input.* calls when live re-validation finds the control now restricted", async () => {
  const { calls, observe, execute } = buildSession(settingsFixture(true), { appOverrides: { setting: { isOpen: true, activeTab: { id: "arxiv-daily" } } } });
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Choose folder").ref;
  const result = await execute({ type: "click", ref }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /folder/i);
  assert.equal(calls.length, 0);
});

test("fill is rejected with zero Input.* calls when live re-validation finds the control sensitive", async () => {
  const { calls, observe, execute } = buildSession(settingsFixture(true), { appOverrides: { setting: { isOpen: true, activeTab: { id: "arxiv-daily" } } } });
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "LLM API key").ref;
  const result = await execute({ type: "fill", ref, value: "x" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /restricted|secret|credential/i);
  assert.equal(calls.length, 0);
});

test("click is rejected with zero Input.* calls when the control is disabled", async () => {
  const { calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button" disabled>Generate</button></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Generate").ref;
  const result = await execute({ type: "click", ref }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /disabled/i);
  assert.equal(calls.length, 0);
});

test("fill is rejected with zero Input.* calls when the field is read-only", async () => {
  const { calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><input type="text" readonly value="x"></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.tag === "input").ref;
  const result = await execute({ type: "fill", ref, value: "y" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /read-only/i);
  assert.equal(calls.length, 0);
});

test("click is rejected when the target stays covered by another element, after exhausting stability retries with zero Input.* calls", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button><div id="overlay"></div></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Generate").ref;
  const node = window.document.querySelector(`[data-acceptance-ref="${ref}"]`);
  node.getBoundingClientRect = () => ({ x: 0, y: 0, width: 100, height: 40 });
  window.document.elementFromPoint = () => window.document.getElementById("overlay");
  const result = await execute({ type: "click", ref }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /timed out/i);
  assert.equal(calls.length, 0);
});

test("fill focuses by a real click, selects existing content, inserts text, verifies the value, then commits with Tab", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><input type="text" value="old"></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.tag === "input").ref;
  stubTarget(window, ref, { x: 0, y: 0, width: 200, height: 30 });
  const result = await execute({ type: "fill", ref, value: "new value" }, snapshot);
  assert.equal(result.ok, true);
  assert.equal(window.document.querySelector(`[data-acceptance-ref="${ref}"]`).value, "new value");
  const methods = calls.map(c => c.method);
  assert.deepEqual(methods.slice(0, 3), ["Input.dispatchMouseEvent", "Input.dispatchMouseEvent", "Input.dispatchMouseEvent"]);
  const keyEvents = calls.filter(c => c.method === "Input.dispatchKeyEvent");
  assert.ok(keyEvents.some(c => c.params.key === "a" && c.params.modifiers === 2), "Ctrl+A select-all is sent before inserting");
  assert.ok(methods.includes("Input.insertText"));
  const last = keyEvents.at(-1);
  assert.equal(last.params.key, "Tab");
});

/**
 * P3 Chunk 3 real-host finding: the "Run for date…"/"Force run for date…" dialog's native
 * `<input type="date">` does not accept `Input.insertText` on this Obsidian/Electron build (confirmed
 * on a real host — the field's `.value` stayed "" after a real click + Ctrl+A + insertText sequence),
 * unlike every text/number/textarea field the test directly above exercises. `fill` now special-cases
 * `type === "date"` to set `.value` directly and dispatch input/change, the same "not reliably drivable
 * with real input in Electron" exception the `select` action below already documents for native
 * `<select>` dropdowns — but only for date fields; non-date fields must keep using the real-insertText
 * path (confirmed by the test directly above, and recontrasted explicitly below).
 */
test("fill on a native date input sets the value via the DOM path (dispatching input/change) instead of Input.insertText, and still commits with Tab", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><input type="date" value=""></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.tag === "input" && c.type === "date").ref;
  stubTarget(window, ref, { x: 0, y: 0, width: 140, height: 30 });
  const node = window.document.querySelector(`[data-acceptance-ref="${ref}"]`);
  const fired = [];
  node.addEventListener("input", () => fired.push("input"));
  node.addEventListener("change", () => fired.push("change"));
  const result = await execute({ type: "fill", ref, value: "2026-10-02" }, snapshot);
  assert.equal(result.ok, true);
  assert.equal(node.value, "2026-10-02");
  assert.deepEqual(fired, ["input", "change"], "the DOM-set path must still dispatch real input/change events, like every other fill");
  const methods = calls.map(c => c.method);
  assert.deepEqual(methods.slice(0, 3), ["Input.dispatchMouseEvent", "Input.dispatchMouseEvent", "Input.dispatchMouseEvent"], "the real click to focus the field still happens");
  assert.ok(!methods.includes("Input.insertText"), "a date field must never go through the real-keystroke insertText path");
  const keyEvents = calls.filter(c => c.method === "Input.dispatchKeyEvent");
  assert.ok(!keyEvents.some(c => c.params.key === "a" && c.params.modifiers === 2), "Ctrl+A select-all is only needed for the real-insertText path, not the DOM-set path");
  assert.equal(keyEvents.at(-1)?.params.key, "Tab", "still commits with a trailing Tab like every other fill");
});

test("fill on a date field still goes through live re-validation first: a disabled date field is rejected with zero Input.* calls, never reaching the DOM-set path", async () => {
  const { calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><input type="date" value="" disabled></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.tag === "input" && c.type === "date").ref;
  const result = await execute({ type: "fill", ref, value: "2026-10-02" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /disabled/i);
  assert.equal(calls.length, 0, "a disabled date field must be rejected before any CDP call, including the DOM-set evaluate() for the date branch");
});

test("fill on a non-date text field is unaffected by the date special-case: it still uses the real Input.insertText path", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><input type="text" value=""></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.tag === "input" && c.type === "text").ref;
  stubTarget(window, ref, { x: 0, y: 0, width: 200, height: 30 });
  const result = await execute({ type: "fill", ref, value: "hello" }, snapshot);
  assert.equal(result.ok, true);
  assert.ok(calls.some(c => c.method === "Input.insertText"), "a plain text field must still be filled via real insertText, not the date DOM-set path");
});

test("select sets the value in-page and dispatches input/change without any Input.* call", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><select><option value="a">A</option><option value="b">B</option></select></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.tag === "select").ref;
  const result = await execute({ type: "select", ref, value: "b" }, snapshot);
  assert.equal(result.ok, true);
  assert.equal(window.document.querySelector(`[data-acceptance-ref="${ref}"]`).value, "b");
  assert.equal(calls.length, 0);
});

test("select is rejected when the requested option is disabled", async () => {
  const { observe, execute, calls } = buildSession('<div class="arxiv-daily-dashboard"><select><option value="a">A</option><option value="b" disabled>B</option></select></div>');
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.tag === "select").ref;
  const result = await execute({ type: "select", ref, value: "b" }, snapshot);
  assert.equal(result.ok, false);
  assert.equal(calls.length, 0);
});

test("scroll sends one mouseWheel event at the center of the current plugin surface", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><p>content</p></div>');
  const snapshot = observe();
  window.document.querySelector(".arxiv-daily-dashboard").getBoundingClientRect = () => ({ x: 0, y: 0, width: 800, height: 600 });
  const result = await execute({ type: "scroll", direction: "down", pixels: 300 }, snapshot);
  assert.equal(result.ok, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].method, "Input.dispatchMouseEvent");
  assert.equal(calls[0].params.type, "mouseWheel");
  assert.equal(calls[0].params.x, 400);
  assert.equal(calls[0].params.y, 300);
  assert.equal(calls[0].params.deltaY, 300);
  const up = await execute({ type: "scroll", direction: "up", pixels: 150 }, snapshot);
  assert.equal(up.ok, true);
  assert.equal(calls.at(-1).params.deltaY, -150);
});

test("scroll is rejected with zero Input.* calls when no plugin surface is open", async () => {
  const { calls, observe, execute } = buildSession("");
  const snapshot = observe();
  const result = await execute({ type: "scroll", direction: "down", pixels: 100 }, snapshot);
  assert.equal(result.ok, false);
  assert.equal(calls.length, 0);
});

test("wait resolves after the requested bounded sleep", async () => {
  const { observe, execute } = buildSession("");
  const snapshot = observe();
  const start = Date.now();
  const result = await execute({ type: "wait", milliseconds: 20 }, snapshot);
  assert.equal(result.ok, true);
  assert.ok(Date.now() - start >= 15);
});

test("wait honours an abort signal instead of sleeping the full duration", async () => {
  const { observe, execute } = buildSession("");
  const snapshot = observe();
  const controller = new AbortController();
  const pending = execute({ type: "wait", milliseconds: 2000 }, snapshot, { signal: controller.signal });
  controller.abort(new Error("exploration cancelled"));
  const result = await pending;
  assert.equal(result.ok, false);
  assert.match(result.error, /cancelled/i);
});

test("press sends a real keyDown/keyUp pair for an allowed key when focus is inside the plugin's scope", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button></div>');
  const snapshot = observe();
  window.document.querySelector(".arxiv-daily-dashboard button").focus();
  const result = await execute({ type: "press", key: "ArrowDown" }, snapshot);
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(c => c.params.type), ["keyDown", "keyUp"]);
  assert.equal(calls[0].params.key, "ArrowDown");
});

test("press sends the Shift modifier bit for Shift+Tab", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button></div>');
  const snapshot = observe();
  window.document.querySelector(".arxiv-daily-dashboard button").focus();
  const result = await execute({ type: "press", key: "Shift+Tab" }, snapshot);
  assert.equal(result.ok, true);
  assert.equal(calls[0].params.key, "Tab");
  assert.equal(calls[0].params.modifiers, 8);
});

test("press is rejected with zero Input.* calls when focus is outside the plugin's scope", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button></div>');
  const snapshot = observe();
  window.document.querySelector(".nav-file-title").focus();
  const result = await execute({ type: "press", key: "Enter" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /scope/i);
  assert.equal(calls.length, 0);
});

test("press defensively rejects a key outside the shared allowlist even if called directly, with zero Input.* calls", async () => {
  const { calls, observe, execute } = buildSession("");
  const snapshot = observe();
  const result = await execute({ type: "press", key: "F5" }, snapshot);
  assert.equal(result.ok, false);
  assert.equal(calls.length, 0);
});

test("back presses Escape only when a plugin modal is open", async () => {
  const { calls, observe, execute } = buildSession('<div class="modal-container"><div class="modal arxiv-daily-modal"><button type="button">Close</button></div></div>');
  const snapshot = observe();
  const result = await execute({ type: "back" }, snapshot);
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(c => c.params.type), ["keyDown", "keyUp"]);
  assert.equal(calls[0].params.key, "Escape");
});

test("back is rejected with zero Input.* calls when no plugin modal is open", async () => {
  const { calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"></div>');
  const snapshot = observe();
  const result = await execute({ type: "back" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /no plugin modal/i);
  assert.equal(calls.length, 0);
});

test("reload is rejected outright as unsupported in Obsidian, with zero Input.* calls", async () => {
  const { calls, observe, execute } = buildSession("");
  const snapshot = observe();
  const result = await execute({ type: "reload" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /not supported|reset/i);
  assert.equal(calls.length, 0);
});

/**
 * Chunk 1 of P3 (task-library) — plugin "More" menu access. The real
 * Obsidian `Menu`/`MenuItem` (resources/obsidian.asar, read-only inspection)
 * render as a body-level `<div class="menu">` containing plain
 * `<div class="menu-item tappable">`s (NO `role`/`tabindex`, so the shared
 * `CONTROL_SELECTOR` needs `.menu-item` added) with a `.menu-item-title`
 * child carrying the label text. `setDisabled(true)` toggles an
 * `is-disabled` class on the item's own dom, not a native `disabled`
 * property. The plugin now marks its own menu's root with
 * `.arxiv-daily-menu` (plugin/src/plugin-menu.ts, mirroring the
 * `.arxiv-daily-modal` marker), mirrored here as the fixture.
 */
function moreMenuFixture() {
  return `
    <div class="menu arxiv-daily-menu">
      <div class="menu-item tappable is-disabled"><div class="menu-item-icon"></div><div class="menu-item-title">Scheduler: Enabled</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Disable scheduler</div></div>
      <div class="menu-separator"></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Run for date…</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Force run for date…</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Retry failed dates</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Cancel active tasks</div></div>
      <div class="menu-separator"></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Run pending</div></div>
      <div class="menu-separator"></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Show logs & history</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Clear run state…</div></div>
      <div class="menu-separator"></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Report a bug</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Request a feature</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Open documentation</div></div>
      <div class="menu-item tappable"><div class="menu-item-icon"></div><div class="menu-item-title">Open repository</div></div>
    </div>
  `;
}

test("the plugin's own marked 'More' menu is observable: safe items are plain, the scheduler toggle and status are restricted/disabled", () => {
  const { observe } = buildWindow(moreMenuFixture());
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "menu");
  const byName = name => snapshot.controls.find(c => c.name === name);
  // Safe items: observable, unrestricted, enabled.
  for (const name of ["Run for date…", "Force run for date…", "Retry failed dates", "Cancel active tasks", "Run pending", "Show logs & history"]) {
    const control = byName(name);
    assert.ok(control, `${name} must be observed`);
    assert.equal(classifyControlRestriction(control), undefined, `${name} must not be restricted`);
    assert.equal(control.disabled, false, `${name} must not be disabled`);
  }
  // The disabled status item: present but non-interactive, and NOT matched by the scheduler-toggle restriction rule (it is disabled, not restricted-by-name).
  const status = byName("Scheduler: Enabled");
  assert.ok(status, "the disabled status item must still be observed (so the model can read it), just not interactive");
  assert.equal(status.disabled, true, "menu items use an is-disabled CLASS, not a native disabled property — describeControl must check for it");
  // The scheduler toggle itself: restricted regardless of enabled/disabled label text.
  const toggle = byName("Disable scheduler");
  assert.match(classifyControlRestriction(toggle), /scheduling/i);
  // Clear run state: restricted (goal.md: clearing run state must stay outside exploration).
  const clearRunState = byName("Clear run state…");
  assert.ok(clearRunState, "Clear run state… must be observed (so the model can see and avoid it), not silently hidden");
  assert.ok(classifyControlRestriction(clearRunState), "Clear run state… must be restricted");
  // External links: restricted. "Open documentation" is a regression-prone exact-label regex (not "open docs"/"documentation" alone).
  for (const name of ["Report a bug", "Request a feature", "Open documentation", "Open repository"]) {
    assert.equal(classifyControlRestriction(byName(name)), "link leaves the plugin for an external destination", `${name} must be classified as an external link`);
  }
});

test("capabilities advertise back when the plugin's own menu is open (so Escape can close it), and dialogOpen is true", () => {
  const { observe } = buildWindow(moreMenuFixture());
  const snapshot = observe();
  assert.ok(snapshot.capabilities.includes("back"), "back must be available while the plugin menu is open");
  assert.ok(!snapshot.capabilities.includes("reload"));
});

test("an unmarked .menu (Obsidian's own context menu, or another plugin's) is entirely invisible, with no fallback", () => {
  const foreignMenu = `
    <div class="menu">
      <div class="menu-item tappable"><div class="menu-item-title">Open in new tab</div></div>
    </div>
  `;
  const { observe } = buildWindow(foreignMenu);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "none");
  assert.deepEqual(snapshot.controls, []);
});

test("ambiguous menus are rejected by construction: when our marked menu and an unmarked foreign .menu are both open, only the marked one is observed", () => {
  const combined = `
    <div class="menu"><div class="menu-item tappable"><div class="menu-item-title">Foreign action</div></div></div>
    ${moreMenuFixture()}
  `;
  const { observe } = buildWindow(combined);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "menu");
  assert.ok(!snapshot.controls.some(c => c.name === "Foreign action"), "the unmarked menu must never leak controls, even alongside our own open menu");
  assert.ok(snapshot.controls.some(c => c.name === "Run for date…"), "our marked menu's own items must still be observed");
});

test("back presses Escape when only the plugin's own menu is open (no modal)", async () => {
  const { calls, observe, execute } = buildSession(moreMenuFixture());
  const snapshot = observe();
  const result = await execute({ type: "back" }, snapshot);
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(c => c.params.type), ["keyDown", "keyUp"]);
  assert.equal(calls[0].params.key, "Escape");
});

test("clicking the scheduler-toggle menu item is rejected with zero Input.* calls, and clicking the disabled status item is rejected as disabled", async () => {
  const { calls, observe, execute } = buildSession(moreMenuFixture());
  const snapshot = observe();
  const toggleRef = snapshot.controls.find(c => c.name === "Disable scheduler").ref;
  const toggleResult = await execute({ type: "click", ref: toggleRef }, snapshot);
  assert.equal(toggleResult.ok, false);
  assert.match(toggleResult.error, /restricted|scheduling/i);
  assert.equal(calls.length, 0);
  const statusRef = snapshot.controls.find(c => c.name === "Scheduler: Enabled").ref;
  const statusResult = await execute({ type: "click", ref: statusRef }, snapshot);
  assert.equal(statusResult.ok, false);
  assert.match(statusResult.error, /disabled/i);
  assert.equal(calls.length, 0);
});

/**
 * P1 Chunk 1 (review Finding 1): `press`'s scope check alone
 * (`obsidianFocusInScopeExpression`) only asks whether `document.activeElement` is
 * inside the current plugin surface — it never classifies the focused element itself,
 * unlike every ref-bearing action's `revalidateRef()`. Tab-focusing a restricted or
 * sensitive control and pressing Enter/Space therefore reached real product code
 * (external `openUrl()`, the scheduler toggle, "Clear run state…", the API-key reveal
 * button) with zero interception. These tests focus each of those controls directly
 * (same style as the existing "press is rejected … outside the plugin's scope" test
 * above — a real Tab traversal is not simulated, only the resulting focus state) and
 * assert the key press is refused before any `Input.*` call, mirroring the exact
 * restricted/sensitive/disabled error wording `revalidateRef()` already uses for
 * click/fill/select.
 */
test("press Enter is rejected with zero Input.* calls when focus is on a settings external-link button (e.g. 'Report a bug')", async () => {
  const html = `
    <div class="vertical-tab-content arxiv-daily-settings" id="our-settings-container">
      <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Support</div></div>
        <div class="setting-item-control"><button type="button">Report a bug</button></div></div>
    </div>
  `;
  const { window, calls, observe, execute } = buildSession(html);
  const snapshot = observe();
  window.document.querySelector(".arxiv-daily-settings button").focus();
  const result = await execute({ type: "press", key: "Enter" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /restricted|external/i);
  assert.equal(calls.length, 0);
});

test("press Enter is rejected with zero Input.* calls when focus is on the scheduler toggle menu item", async () => {
  const { window, calls, observe, execute } = buildSession(moreMenuFixture());
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Disable scheduler").ref;
  window.document.querySelector(`[data-acceptance-ref="${ref}"]`).focus();
  const result = await execute({ type: "press", key: "Enter" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /restricted|scheduling/i);
  assert.equal(calls.length, 0);
});

test("press Space is rejected with zero Input.* calls when focus is on the scheduler toggle menu item", async () => {
  const { window, calls, observe, execute } = buildSession(moreMenuFixture());
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Disable scheduler").ref;
  window.document.querySelector(`[data-acceptance-ref="${ref}"]`).focus();
  const result = await execute({ type: "press", key: "Space" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /restricted|scheduling/i);
  assert.equal(calls.length, 0);
});

test("press Enter is rejected with zero Input.* calls when focus is on the 'Clear run state…' menu item", async () => {
  const { window, calls, observe, execute } = buildSession(moreMenuFixture());
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Clear run state…").ref;
  window.document.querySelector(`[data-acceptance-ref="${ref}"]`).focus();
  const result = await execute({ type: "press", key: "Enter" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /restricted/i);
  assert.equal(calls.length, 0);
});

test("press Enter is rejected with zero Input.* calls when focus is on the API key reveal button", async () => {
  const { window, calls, observe, execute } = buildSession(settingsFixture(true));
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Show LLM API key").ref;
  window.document.querySelector(`[data-acceptance-ref="${ref}"]`).focus();
  const result = await execute({ type: "press", key: "Enter" }, snapshot);
  assert.equal(result.ok, false);
  assert.match(result.error, /restricted|secret|credential/i);
  assert.equal(calls.length, 0);
});

test("press Enter still activates an ordinary, unrestricted focused control", async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-dashboard"><button type="button">Generate</button></div>');
  const snapshot = observe();
  window.document.querySelector(".arxiv-daily-dashboard button").focus();
  const result = await execute({ type: "press", key: "Enter" }, snapshot);
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(c => c.params.type), ["keyDown", "keyUp"]);
  assert.equal(calls[0].params.key, "Enter");
});

// Tab/Shift+Tab/Escape never invoke the focused control's own activation handler (unlike
// Enter/Space, or Arrow/Home/End/Page on a native select), so they stay allowed regardless of
// the focused control's own restriction — otherwise a keyboard-only persona that lands on a
// restricted control (a legitimate observation, not an activation) would have no way to move
// past it short of "home", which abandons its current position entirely.
test("press Tab still moves focus away from a restricted control; navigation itself is not blocked by the new focus check", async () => {
  const { window, calls, observe, execute } = buildSession(moreMenuFixture());
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Disable scheduler").ref;
  window.document.querySelector(`[data-acceptance-ref="${ref}"]`).focus();
  const result = await execute({ type: "press", key: "Tab" }, snapshot);
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(c => c.params.type), ["keyDown", "keyUp"]);
  assert.equal(calls[0].params.key, "Tab");
});

test("press Escape still fires when focus is on a restricted control", async () => {
  const { window, calls, observe, execute } = buildSession(moreMenuFixture());
  const snapshot = observe();
  const ref = snapshot.controls.find(c => c.name === "Clear run state…").ref;
  window.document.querySelector(`[data-acceptance-ref="${ref}"]`).focus();
  const result = await execute({ type: "press", key: "Escape" }, snapshot);
  assert.equal(result.ok, true);
  assert.deepEqual(calls.map(c => c.params.type), ["keyDown", "keyUp"]);
  assert.equal(calls[0].params.key, "Escape");
});

// ---------------------------------------------------------------------------
// P1 Chunk 3 (review Finding 1 "Compounding"): Obsidian screenshots have no secret masking at all,
// unlike the workbench driver's `mask:` list — a revealed ("Show ... API key") or plain password
// input would be captured verbatim into a PNG artifact. `stubRect` fakes real layout the way
// `stubTarget` already does for a single ref, since happy-dom's own `getBoundingClientRect` is
// always zero (see buildWindow's own comment above).
// ---------------------------------------------------------------------------

function stubRect(node, rect) {
  node.getBoundingClientRect = () => ({ x: rect.x, y: rect.y, left: rect.x, top: rect.y, right: rect.x + rect.width, bottom: rect.y + rect.height, width: rect.width, height: rect.height });
}

test("obsidianMaskSecretFieldsExpression covers a revealed API-key input (reveal-sibling present even though its own type is now 'text') and a password input, never an ordinary field", () => {
  const html = `
    <div class="view-content arxiv-daily-dashboard">
      <div class="setting-item-control">
        <input type="text" class="arxiv-daily-settings__llm-input" aria-label="LLM API key" value="sk-revealed-secret">
        <button type="button" class="arxiv-daily-settings__reveal-key" aria-label="Show LLM API key"></button>
      </div>
      <div class="setting-item-control"><input type="password" value="hunter2"></div>
      <div class="setting-item-control"><input type="text" value="ordinary"></div>
    </div>
  `;
  const { window, context } = buildWindow(html);
  const revealedKey = window.document.querySelector(".arxiv-daily-settings__llm-input");
  const password = window.document.querySelector('input[type="password"]');
  const ordinary = window.document.querySelector('input[value="ordinary"]');
  stubRect(revealedKey, { x: 1, y: 2, width: 100, height: 20 });
  stubRect(password, { x: 3, y: 4, width: 100, height: 20 });
  stubRect(ordinary, { x: 5, y: 6, width: 100, height: 20 });
  const count = vm.runInNewContext(obsidianMaskSecretFieldsExpression(), context);
  assert.equal(count, 2, "exactly the two secret fields are masked");
  const overlays = [...window.document.querySelectorAll('[data-exploration-mask="1"]')];
  assert.equal(overlays.length, 2);
  const lefts = overlays.map(div => div.style.left);
  assert.ok(lefts.includes("1px") && lefts.includes("3px"), "both secret fields' own rects are covered");
  assert.ok(!lefts.includes("5px"), "the ordinary field's rect must never be covered");
});

test("obsidianUnmaskSecretFieldsExpression removes exactly the overlays obsidianMaskSecretFieldsExpression added, and nothing else", () => {
  const html = `<div class="view-content arxiv-daily-dashboard"><input type="password" value="x"><div class="keep-me"></div></div>`;
  const { window, context } = buildWindow(html);
  stubRect(window.document.querySelector('input[type="password"]'), { x: 0, y: 0, width: 50, height: 10 });
  vm.runInNewContext(obsidianMaskSecretFieldsExpression(), context);
  assert.equal(window.document.querySelectorAll('[data-exploration-mask="1"]').length, 1);
  const removed = vm.runInNewContext(obsidianUnmaskSecretFieldsExpression(), context);
  assert.equal(removed, 1);
  assert.equal(window.document.querySelectorAll('[data-exploration-mask="1"]').length, 0);
  assert.ok(window.document.querySelector(".keep-me"), "unrelated elements must not be touched");
});

/** A minimal 2x1 RGB PNG with two different pixels, so `judgeCapturedImage` (screenshots.mjs) accepts it as real content — the same technique desktop-acceptance-screenshots.test.mjs uses, kept to the one case this file needs (filter 0, no interlacing). */
function tinyVariedPng() {
  const raw = Buffer.from([0, 10, 10, 10, 200, 200, 200]); // filter byte, pixel 1 (dark), pixel 2 (light)
  const chunk = (type, data) => {
    const out = Buffer.alloc(12 + data.length);
    out.writeUInt32BE(data.length, 0);
    out.write(type, 4, "latin1");
    data.copy(out, 8);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(2, 0); // width
  ihdr.writeUInt32BE(1, 4); // height
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // color type: RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/**
 * A fake CDP session for `createObsidianExplorationDriver`'s `capture()`. Unlike `buildSession`
 * above (which only drives `createObsidianActionExecutor`), `Runtime.evaluate` here also has to
 * run screenshots.mjs's own rect-measurement expression (`captureTargetExpression`), which reads
 * `window.innerWidth`/`window.getComputedStyle` — so the vm context exposes the happy-dom `Window`
 * itself as `window`, not just its bare globals, and everything runs for real against one document.
 */
function buildCaptureSession(bodyHtml) {
  const window = new Window();
  window.document.body.innerHTML = bodyHtml;
  // createObsidianDriver's construction unconditionally installs the fixture HTTP proxy
  // (obsidian-ui.mjs, installHttpProxyExpression), which reaches into `app.plugins.plugins
  // ["arxiv-daily"].getHttpClient()` — needed here even though capture() itself never touches it.
  const httpClient = { request: async () => ({}) };
  const app = { plugins: { plugins: { "arxiv-daily": { getHttpClient: () => httpClient } } } };
  const context = { document: window.document, window, getComputedStyle: (...args) => window.getComputedStyle(...args), innerWidth: 1200, innerHeight: 800, app };
  const trace = [];
  const client = {
    async send(method, params = {}) {
      trace.push({ method, maskedBefore: window.document.querySelectorAll('[data-exploration-mask="1"]').length });
      if (method === "Runtime.evaluate") {
        let value;
        try { value = vm.runInNewContext(params.expression, context); }
        catch (error) { return { result: {}, exceptionDetails: { text: "Uncaught", exception: { description: error.message } } }; }
        return { result: { value } };
      }
      if (method === "Page.captureScreenshot") return { data: tinyVariedPng().toString("base64") };
      return {};
    },
  };
  const evaluate = expression => cdpEvaluate(client, expression);
  return { window, trace, session: { client, evaluate, diagnostics: { entries: () => [] } } };
}

async function withTempArtifactDir(run) {
  const root = await mkdtemp(path.join(tmpdir(), "exploration-obsidian-capture-test-"));
  try { return await run(root); }
  finally { await rm(root, { recursive: true, force: true }); }
}

const driverCaptureFixture = `
  <div class="workspace" style="opacity:1">
    <div class="vertical-tab-content arxiv-daily-settings">
      <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">API key</div></div>
        <div class="setting-item-control">
          <input type="text" class="arxiv-daily-settings__llm-input" aria-label="LLM API key" value="sk-real-secret-value">
          <button type="button" class="arxiv-daily-settings__reveal-key" aria-label="Show LLM API key"></button>
        </div></div>
      <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Password</div></div>
        <div class="setting-item-control"><input type="password" value="hunter2"></div></div>
      <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Daily paper limit</div></div>
        <div class="setting-item-control"><input type="number" value="20"></div></div>
    </div>
  </div>
`;

test("driver.capture() masks the revealed API-key input and the password input before the screenshot, and removes both masks afterward; the ordinary field is never covered", async () => {
  await withTempArtifactDir(async root => {
    const { window, trace, session } = buildCaptureSession(driverCaptureFixture);
    const workspace = window.document.querySelector(".workspace");
    stubRect(workspace, { x: 0, y: 0, width: 800, height: 600 });
    for (const selector of [".arxiv-daily-settings__llm-input", 'input[type="password"]', 'input[type="number"]']) {
      stubRect(window.document.querySelector(selector), { x: 10, y: 10, width: 100, height: 20 });
    }
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const artifactDir = path.join(root, "artifacts");
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const pngPath = await driver.capture("settings-with-secret");
    assert.match(pngPath, /settings-with-secret\.png$/);
    const screenshotCall = trace.find(entry => entry.method === "Page.captureScreenshot");
    assert.ok(screenshotCall, "the screenshot must still be taken");
    assert.equal(screenshotCall.maskedBefore, 2, "both secret fields must already be masked by the time the screenshot is taken");
    assert.equal(window.document.querySelectorAll('[data-exploration-mask="1"]').length, 0, "masks must be removed after a successful capture");
  });
});

test("driver.capture() still removes the mask when the screenshot itself throws (no '.workspace' element to capture)", async () => {
  await withTempArtifactDir(async root => {
    const { window, trace, session } = buildCaptureSession('<div class="arxiv-daily-settings"><input type="password" value="hunter2"></div>');
    stubRect(window.document.querySelector('input[type="password"]'), { x: 10, y: 10, width: 100, height: 20 });
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const artifactDir = path.join(root, "artifacts");
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    await assert.rejects(() => driver.capture("no-workspace"), /nothing matched/);
    assert.ok(!trace.some(entry => entry.method === "Page.captureScreenshot"), "the screenshot itself must never have been reached");
    assert.ok(trace.some(entry => entry.maskedBefore === 1), "the mask must have been applied before the refusal");
    assert.equal(window.document.querySelectorAll('[data-exploration-mask="1"]').length, 0, "the mask must be removed even though capture() threw");
  });
});

/**
 * P5 Chunk 2 (phases/05-real-model-release-run.md, real-model finding): a fake session for
 * createObsidianExplorationDriver's `home` action. Unlike `buildCaptureSession` (screenshots only),
 * this also fakes `app.setting.close()`/`app.commands.executeCommandById()` so the real
 * `createObsidianDriver`'s `openDashboard()`/`closeSettings()` helpers run unmodified against it, and
 * optionally simulates Obsidian's own documented behaviour (see this file's header comment on `back`,
 * confirmed by reading `resources/obsidian.asar`) that a real Escape keypress closes whichever plugin
 * modal/menu is open — gated by `escapeCloses` so one test can also exercise the "closing is
 * impossible" path without a second, divergent fake.
 */
function buildHomeSession(bodyHtml, { escapeCloses = true } = {}) {
  const window = new Window();
  window.document.body.innerHTML = bodyHtml;
  const httpClient = { request: async () => ({}) };
  const app = {
    plugins: { plugins: { "arxiv-daily": { getHttpClient: () => httpClient } } },
    setting: { close() {}, open() {}, openTabById() {}, isOpen: false, activeTab: undefined },
    commands: { executeCommandById: () => true },
  };
  const context = { document: window.document, window, getComputedStyle: (...args) => window.getComputedStyle(...args), innerWidth: 1200, innerHeight: 800, app };
  const client = {
    async send(method, params = {}) {
      if (method === "Runtime.evaluate") {
        let value;
        try { value = vm.runInNewContext(params.expression, context); }
        catch (error) { return { result: {}, exceptionDetails: { text: "Uncaught", exception: { description: error.message } } }; }
        return { result: { value } };
      }
      if (method === "Input.dispatchKeyEvent" && params.type === "keyUp" && params.key === "Escape" && escapeCloses) {
        window.document.querySelectorAll(".arxiv-daily-modal, .arxiv-daily-menu").forEach(node => {
          (node.closest(".modal-container") || node).remove();
        });
      }
      return {};
    },
  };
  const evaluate = expression => cdpEvaluate(client, expression);
  return { window, session: { client, evaluate, diagnostics: { entries: () => [] } } };
}

const homeFixtureHtml = `
  <div class="workspace">
    <div class="view-content arxiv-daily-dashboard">
      <div class="arxiv-daily-dashboard__tab is-active" role="tab" tabindex="0">All</div>
    </div>
  </div>
  <div class="modal-container"><div class="modal arxiv-daily-modal">
    <button type="button">Close</button>
  </div></div>
`;

// Real-host finding (P5 chunk 1 triage, library-edit-candidate): `home` went straight through
// `helper.openDashboard()`, which only closes Obsidian's own settings window — a plugin modal is a
// separate overlay appended to document.body and is unaffected by which workspace leaf/view is shown.
// `home` opened the dashboard BEHIND the still-open review modal and reported success, so the model
// repeated `home` 5 times with the modal never actually going away.
test("home closes an open plugin modal before opening the dashboard, and only then reports success", async () => {
  await withTempArtifactDir(async root => {
    const { window, session } = buildHomeSession(homeFixtureHtml, { escapeCloses: true });
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const artifactDir = path.join(root, "artifacts");
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const observation = await driver.observe();
    assert.equal(observation.layout.view, "modal", "the modal must be the observed surface before home runs");
    const result = await driver.execute({ type: "home" }, observation);
    assert.equal(result.ok, true);
    // Boolean, not assert.equal(element, null): a failing deep-equal on a happy-dom element makes node's
    // assertion diff walk the whole DOM object graph and hang instead of reporting the failure.
    assert.ok(!window.document.querySelector(".arxiv-daily-modal"), "home must close the plugin modal, not just open the dashboard behind it");
  });
});

test("home reports failure instead of success when an open plugin modal cannot be closed", async () => {
  await withTempArtifactDir(async root => {
    const { session } = buildHomeSession(homeFixtureHtml, { escapeCloses: false });
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const artifactDir = path.join(root, "artifacts");
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const observation = await driver.observe();
    const result = await driver.execute({ type: "home" }, observation);
    assert.equal(result.ok, false, "home must not report success while the plugin modal is still open");
    assert.match(result.error, /close/i);
  });
});

test("home still opens the dashboard normally when no plugin modal or menu is open", async () => {
  await withTempArtifactDir(async root => {
    const { window, session } = buildHomeSession(`
      <div class="workspace">
        <div class="view-content arxiv-daily-dashboard">
          <div class="arxiv-daily-dashboard__tab is-active" role="tab" tabindex="0">All</div>
        </div>
      </div>
    `);
    const fixture = { vaultRoot: path.join(root, "vault"), server: { url: "http://127.0.0.1:1" } };
    const artifactDir = path.join(root, "artifacts");
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const observation = await driver.observe();
    const result = await driver.execute({ type: "home" }, observation);
    assert.equal(result.ok, true);
    assert.equal(window.document.activeElement?.className, "arxiv-daily-dashboard__tab is-active", "openDashboard()'s own focus seeding must still run");
  });
});

test('setup checklist scheduler activation is restricted', () => {
  assert.match(classifyControlRestriction({ name: 'Turn on daily reports', tag: 'button' }), /scheduling/);
});

for (const key of ['Enter', 'Space']) test(`setup checklist blocks keyboard ${key}`, async () => {
  const { window, calls, observe, execute } = buildSession('<div class="arxiv-daily-settings"><button>Turn on daily reports</button></div>');
  const snapshot = observe();
  window.document.querySelector('button').focus();
  const result = await execute({ type: 'press', key }, snapshot);
  assert.equal(result.ok, false);
  assert.equal(calls.length, 0);
});

test('active note replaces the hidden dashboard in observation', () => {
  const { observe } = buildWindow('<div class="workspace-leaf"><div class="arxiv-daily-dashboard">Hidden dashboard<button>Generate</button></div></div><div class="workspace-leaf mod-active"><div class="view-header-title">Daily report</div><div data-type="markdown"><div class="markdown-preview-view">Actual report content<button>Untrusted note button</button></div></div></div>');
  const snapshot = observe();
  assert.equal(snapshot.layout.view, 'note');
  assert.equal(snapshot.layout.activeLeaf.type, 'markdown');
  assert.equal(snapshot.layout.activeLeaf.title, 'Daily report');
  assert.match(snapshot.text, /Actual report content/);
  assert.doesNotMatch(snapshot.text, /Hidden dashboard/);
  assert.deepEqual(snapshot.controls, []);
});
test('external open guard intercepts both windows and reports attempted URLs', async () => {
  await withTempArtifactDir(async root => {
    const { window, session } = buildCaptureSession('<div class="arxiv-daily-dashboard"></div>');
    let forwarded = 0;
    window.open = () => { forwarded++; };
    window.activeWindow = { open: () => { forwarded++; } };
    const driver = await createObsidianExplorationDriver({ session, fixture: { vaultRoot: path.join(root, 'vault'), server: { url: 'http://127.0.0.1:1' } }, artifactDir: root });
    window.open('https://arxiv.org/abs/2610.10001');
    window.activeWindow.open('https://arxiv.org/pdf/2610.10001');
    assert.equal(forwarded, 0);
    const snapshot = await driver.observe();
    assert.deepEqual(snapshot.layout.openedUrls, ['https://arxiv.org/abs/2610.10001', 'https://arxiv.org/pdf/2610.10001']);
    assert.match(snapshot.text, /blocked/i);
    for (let i = 0; i < 25; i++) window.open(`${i}:${'x'.repeat(1500)}`);
    const bounded = await driver.observe();
    assert.equal(bounded.layout.openedUrls.length, 20);
    assert.ok(bounded.layout.openedUrls.every(url => url.length === 1000));
    assert.ok(bounded.layout.openedUrls[0].startsWith('5:'));
    assert.equal(forwarded, 0);
  });
});
test('PDF external buttons share the guarded arXiv policy', () => {
  assert.equal(classifyControlRestriction({ name: 'Open PDF' }), classifyControlRestriction({ name: 'Open arXiv' }));
});

test('obsidian focus follows control refs and marks out-of-scope focus', () => {
  const { window, observe } = buildWindow(dashboardFixture);
  const buttons = window.document.querySelectorAll('button');
  buttons[0].focus();
  let snapshot = observe('s1');
  assert.deepEqual(snapshot.focus, { ref: 's1-3', tag: 'button', name: 'Star' });
  buttons[1].focus();
  snapshot = observe('s2');
  assert.deepEqual(snapshot.focus, { ref: 's2-4', tag: 'button', name: 'Open arXiv' });
  window.document.querySelector('.nav-file-title').focus();
  assert.deepEqual(observe().focus, { outside: true });
});

test('active dashboard wins over an earlier inactive dashboard leaf', () => {
  const { observe } = buildWindow('<div class="workspace-leaf"><div class="arxiv-daily-dashboard">Stale dashboard</div></div><div class="workspace-leaf mod-active"><div class="arxiv-daily-dashboard"><button>Active dashboard action</button></div></div>');
  const snapshot = observe();
  assert.equal(snapshot.layout.view, 'dashboard');
  assert.match(snapshot.text, /Active dashboard action/);
  assert.doesNotMatch(snapshot.text, /Stale dashboard/);
});
