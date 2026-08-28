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

test("the disabled-sidecar scenario passes when the setting is off and nothing was requested", async () => {
  const session = fakeSession([
    ["pdfParserSidecar", JSON.stringify({ enabled: false, capabilitiesUrl: "http://127.0.0.1:8765/capabilities", parseUrl: "http://127.0.0.1:8765/parse" })],
  ]);
  const result = await sidecarDisabledScenario({ session, requests: [] });
  assert.equal(result.passed, true);
});

test("the disabled-sidecar scenario fails when a request reached the sidecar endpoint", async () => {
  const session = fakeSession([
    ["pdfParserSidecar", JSON.stringify({ enabled: false, capabilitiesUrl: "http://127.0.0.1:8765/capabilities", parseUrl: "http://127.0.0.1:8765/parse" })],
  ]);
  const result = await sidecarDisabledScenario({
    session,
    requests: ["http://127.0.0.1:8765/capabilities"],
  });
  assert.equal(result.passed, false);
  assert.match(result.detail, /8765/);
});

test("the disabled-sidecar scenario fails when the setting defaulted to enabled", async () => {
  const session = fakeSession([
    ["pdfParserSidecar", JSON.stringify({ enabled: true, capabilitiesUrl: "http://127.0.0.1:8765/c", parseUrl: "http://127.0.0.1:8765/p" })],
  ]);
  const result = await sidecarDisabledScenario({ session, requests: [] });
  assert.equal(result.passed, false);
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

test("the fallback scenario passes when an unreachable sidecar produced no console error", async () => {
  const session = fakeSession([["pdfParserSidecar", JSON.stringify({ enabled: true })], ["setTimeout", "waited"]], []);
  const result = await sidecarFallbackScenario({ session, unreachablePort: 1 });
  assert.equal(result.passed, true);
});

test("the fallback scenario fails when enabling the sidecar produced a console error", async () => {
  // The error must appear as a consequence of enabling, so the scenario can
  // attribute it rather than inheriting an unrelated startup error.
  const raised = [];
  const session = {
    evaluate: async (expression) => {
      if (expression.includes("pdfParserSidecar")) {
        raised.push({ source: "console", level: "error", text: "sidecar probe exploded" });
        return "configured";
      }
      return "waited";
    },
    diagnostics: { errors: () => raised },
  };
  const result = await sidecarFallbackScenario({ session, unreachablePort: 1 });
  assert.equal(result.passed, false);
  assert.match(result.detail, /exploded/);
});

test("the fallback scenario ignores an error that predates it", async () => {
  const session = fakeSession(
    [["pdfParserSidecar", "configured"], ["setTimeout", "waited"]],
    [{ source: "console", level: "error", text: "unrelated startup error" }],
  );
  const result = await sidecarFallbackScenario({ session, unreachablePort: 1 });
  assert.equal(result.passed, true);
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
