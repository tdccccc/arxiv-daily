import assert from "node:assert/strict";
import test from "node:test";
import {
  pdfPageLocationScenario,
  runScenarios,
  settingsMigrationScenario,
  sidecarDisabledScenario,
  sidecarFallbackScenario,
} from "../desktop-acceptance/scenarios.mjs";

/** Session stub: `answers` maps an expression substring to a value or function. */
function fakeSession(answers = [], diagnosticsErrors = []) {
  const calls = [];
  return {
    calls,
    evaluate: async (expression) => {
      calls.push(expression);
      for (const [needle, produce] of answers) {
        if (expression.includes(needle)) {
          return typeof produce === "function" ? produce(expression, calls.length) : produce;
        }
      }
      return null;
    },
    diagnostics: { errors: () => diagnosticsErrors },
  };
}

const pdfAnswers = (page) => [
  ["extension === \"pdf\"", "test_library/paper.pdf"],
  ["openLinkText", true],
  ["setTimeout", "waited"],
  ["getViewState", JSON.stringify({ type: "pdf", state: { file: "test_library/paper.pdf" } })],
  ["currentPageNumber", page],
];

test("the PDF scenario passes when the viewer really navigated to the requested page", async () => {
  const session = fakeSession(pdfAnswers(4));
  const result = await pdfPageLocationScenario({ session, page: 4 });
  assert.equal(result.passed, true);
  assert.match(result.detail, /page 4/);
});

test("the PDF scenario fails when the viewer stayed on another page", async () => {
  const session = fakeSession(pdfAnswers(1));
  const result = await pdfPageLocationScenario({ session, page: 4 });
  assert.equal(result.passed, false);
  assert.match(result.detail, /1/);
});

test("the PDF scenario fails when no PDF view opened at all", async () => {
  const session = fakeSession([
    ["extension === \"pdf\"", "test_library/paper.pdf"],
    ["openLinkText", true],
    ["setTimeout", "waited"],
    ["getViewState", JSON.stringify({ type: "markdown" })],
    ["currentPageNumber", null],
  ]);
  const result = await pdfPageLocationScenario({ session, page: 4 });
  assert.equal(result.passed, false);
  assert.match(result.detail, /pdf/i);
});

test("the PDF scenario reports honestly when the vault holds no PDF", async () => {
  const session = fakeSession([["extension === \"pdf\"", null]]);
  const result = await pdfPageLocationScenario({ session, page: 4 });
  assert.equal(result.passed, false);
  assert.match(result.detail, /no pdf/i);
});

/** Stand-in for the real loopback listener: records what the plugin sent. */
function fakeListener() {
  const requests = [];
  return {
    origin: "http://127.0.0.1:45001",
    capabilitiesUrl: "http://127.0.0.1:45001/v1/capabilities",
    parseUrl: "http://127.0.0.1:45001/v1/parse",
    requests: () => [...requests],
    record: () => requests.push({ method: "GET", path: "/v1/capabilities" }),
  };
}

/** Session stub whose parser build reaches the listener only when enabled. */
function sidecarSession({ enabled, built, probes, errorsOnChange = [], preexisting = [] }, listener) {
  const errors = [...preexisting];
  return {
    evaluate: async (expression) => {
      if (expression.includes("pdfParserSidecar ?? null")) {
        return JSON.stringify({
          enabled,
          capabilitiesUrl: listener.capabilitiesUrl,
          parseUrl: listener.parseUrl,
        });
      }
      if (expression.includes("settingsChanges.changeValue")) {
        errors.push(...errorsOnChange);
        return "changed";
      }
      if (expression.includes("buildFullTextDocumentParser")) {
        if (probes) listener.record();
        return JSON.stringify(built);
      }
      return "waited";
    },
    diagnostics: { errors: () => errors },
  };
}

const pdfjsOnly = { parser: true, parserSelector: false };
const sidecarAdopted = { parser: false, parserSelector: true };

test("the disabled scenario passes when building the parser sends nothing", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: false, built: pdfjsOnly, probes: false }, listener);
  const result = await sidecarDisabledScenario({ session, listener });
  assert.equal(result.passed, true);
});

test("the disabled scenario fails when a request still reached the listener", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: false, built: pdfjsOnly, probes: true }, listener);
  const result = await sidecarDisabledScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /capabilities/);
});

test("the disabled scenario fails when the setting defaulted to enabled", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, built: pdfjsOnly, probes: false }, listener);
  const result = await sidecarDisabledScenario({ session, listener });
  assert.equal(result.passed, false);
});

test("the fallback scenario passes when a refused probe produced PDF.js", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, built: pdfjsOnly, probes: true }, listener);
  const result = await sidecarFallbackScenario({ session, listener });
  assert.equal(result.passed, true);
  assert.match(result.detail, /probe request reached/);
});

test("the fallback scenario fails when no request ever reached the endpoint", async () => {
  // Without an observed request the pass would be vacuous: building returns the
  // PDF.js parser whenever the sidecar is off.
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, built: pdfjsOnly, probes: false }, listener);
  const result = await sidecarFallbackScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /no request reached/);
});

test("the fallback scenario fails when the sidecar was adopted despite a refusal", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, built: sidecarAdopted, probes: true }, listener);
  const result = await sidecarFallbackScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /selector/);
});

test("the fallback scenario fails when the settings transaction rejected the change", async () => {
  const listener = fakeListener();
  const session = {
    evaluate: async (expression) =>
      expression.includes("settingsChanges.changeValue")
        ? "ERROR: Invalid sidecar configuration"
        : "waited",
    diagnostics: { errors: () => [] },
  };
  const result = await sidecarFallbackScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /Invalid sidecar configuration/);
});

test("the fallback scenario fails when the failed probe raised a console error", async () => {
  const listener = fakeListener();
  const session = sidecarSession(
    {
      enabled: true,
      built: pdfjsOnly,
      probes: true,
      errorsOnChange: [{ source: "console", level: "error", text: "sidecar probe exploded" }],
    },
    listener,
  );
  const result = await sidecarFallbackScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /exploded/);
});

test("the migration scenario passes when old settings gained the sidecar defaults", async () => {
  const session = fakeSession([
    ["pdfParserSidecar", JSON.stringify({ enabled: false, capabilitiesUrl: "http://127.0.0.1:8765/capabilities", parseUrl: "http://127.0.0.1:8765/parse" })],
    ["Object.keys", ["llm", "arxiv", "output", "schedule", "advanced", "email", "detailSelection", "pdfParserSidecar"]],
  ]);
  const result = await settingsMigrationScenario({ session });
  assert.equal(result.passed, true);
});

test("the migration scenario fails when the migrated settings are missing a section", async () => {
  const session = fakeSession([
    ["pdfParserSidecar", JSON.stringify({ enabled: false, capabilitiesUrl: "", parseUrl: "" })],
    ["Object.keys", ["llm"]],
  ]);
  const result = await settingsMigrationScenario({ session });
  assert.equal(result.passed, false);
});

test("runScenarios reports every scenario and fails overall if any failed", async () => {
  const results = await runScenarios([
    async () => ({ name: "a", passed: true, detail: "fine" }),
    async () => ({ name: "b", passed: false, detail: "broken" }),
  ]);
  assert.equal(results.passed, false);
  assert.deepEqual(results.scenarios.map((s) => s.name), ["a", "b"]);
});

test("runScenarios turns a thrown scenario into a failure rather than aborting the run", async () => {
  const results = await runScenarios([
    async () => {
      throw new Error("scenario blew up");
    },
    async () => ({ name: "b", passed: true, detail: "fine" }),
  ]);
  assert.equal(results.passed, false);
  assert.match(results.scenarios[0].detail, /blew up/);
  assert.equal(results.scenarios[1].passed, true);
});
