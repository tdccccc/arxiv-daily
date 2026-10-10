import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { Window } from "happy-dom";
import { classifyControlRestriction, obsidianObservation, obsidianObservationExpression } from "../acceptance/exploration-obsidian.mjs";

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
  <div class="workspace-leaf">
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

test("reading dashboard: paper row star is operable, Open arXiv is restricted, chrome stays excluded", () => {
  const { observe } = buildWindow(dashboardFixture);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "dashboard");
  assert.equal(snapshot.layout.rootMounted, true);
  const star = snapshot.controls.find(c => c.name === "Star");
  assert.equal(star.paperKey, "2610.10001");
  assert.equal(classifyControlRestriction(star), undefined);
  const openArxiv = snapshot.controls.find(c => c.name === "Open arXiv");
  assert.equal(classifyControlRestriction(openArxiv), "link leaves the plugin for an external destination");
  const openPdf = snapshot.controls.find(c => c.name === "Open PDF");
  assert.equal(classifyControlRestriction(openPdf), undefined, "Open PDF can resolve to a local vault file and is not unconditionally external");
  assert.ok(!snapshot.controls.some(c => c.name === "Open quick switcher"), "ribbon chrome must stay invisible even when the dashboard is open");
});

test("an unmarked plugin modal (e.g. the date picker) is the only in-scope surface when it is the only open modal", () => {
  const modal = `
    <div class="modal-container">
      <div class="modal">
        <h2>Run arXiv Daily for date</h2>
        <div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Date</div></div>
          <div class="setting-item-control"><input type="date" value="2026-10-01"></div></div>
        <button type="button" class="mod-cta">Run</button>
      </div>
    </div>
  `;
  const { observe } = buildWindow(modal);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "unmarked-modal");
  assert.ok(snapshot.controls.some(c => c.tag === "input" && c.type === "date"));
  assert.ok(snapshot.controls.some(c => c.name === "Run"));
});

test("a marked plugin modal wins over a simultaneously open settings modal", () => {
  const combined = `
    <div class="modal mod-settings"><div class="setting-item"><div class="setting-item-info"><div class="setting-item-name">Settings noise</div></div>
      <div class="setting-item-control"><button type="button">Settings button</button></div></div></div>
    <div class="modal arxiv-daily-hub-modal"><div class="arxiv-daily-hub-modal__content">
      <button type="button" aria-label="Close hub">x</button>
    </div></div>
  `;
  const { observe } = buildWindow(combined);
  const snapshot = observe();
  assert.equal(snapshot.layout.view, "modal");
  assert.ok(snapshot.controls.some(c => c.name === "Close hub"));
  assert.ok(!snapshot.controls.some(c => c.name === "Settings button"));
});

test("known gap: a generic non-plugin modal with no marker is indistinguishable from an unmarked plugin modal", () => {
  const nonPluginModal = `
    <div class="modal-container"><div class="modal">
      <button type="button">Some other plugin's button</button>
    </div></div>
  `;
  const { observe } = buildWindow(nonPluginModal);
  const snapshot = observe();
  // Documented limitation (see exploration-obsidian.mjs comments): in the single-plugin
  // acceptance fixture this is acceptable, but it is not a general-purpose scoping guarantee.
  assert.equal(snapshot.layout.view, "unmarked-modal");
  assert.ok(snapshot.controls.some(c => c.name === "Some other plugin's button"));
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
  assert.deepEqual(Object.keys(snapshot).sort(), ["controls", "id", "layout", "text", "title"].sort());
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
