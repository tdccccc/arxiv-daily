import assert from "node:assert/strict";
import test from "node:test";
import {
  getModelsScenario,
  pdfPageLocationScenario,
  runScenarios,
  settingsMigrationScenario,
  sidecarDisabledScenario,
  sidecarEnabledIgnoredScenario,
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

/**
 * Session stub for the full-text extractor build. `enabled` drives two
 * things at once, mirroring `buildFullTextExtractor` in plugin/main.ts: it is
 * read back from the settings JSON, and it is what makes the real plugin
 * append the "not used for indexing" note to its diagnostics *log buffer*
 * (`Logger.getBuffer()`) -- not the console, which `Logger.info` only writes
 * to when the log level is turned up to "debug" (see readLoggerBuffer's
 * comment in scenarios.mjs). `probes` simulates a regression where building
 * the extractor talks to the sidecar anyway.
 */
function sidecarSession(
  { enabled, builtProvenanceId = "obsidian-pdfjs", probes, errorsOnChange = [], preexisting = [] },
  listener,
) {
  const buffer = [];
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
      if (expression.includes("getBuffer")) {
        return JSON.stringify(buffer);
      }
      if (expression.includes("settingsChanges.changeValue")) {
        errors.push(...errorsOnChange);
        return "changed";
      }
      if (expression.includes("buildFullTextExtractor")) {
        if (probes) listener.record();
        if (enabled) {
          buffer.push(
            "[INFO] fulltext: the local PDF parser sidecar is not used for indexing; the index reads titles and abstracts with PDF.js",
          );
        }
        return JSON.stringify({ provenanceId: builtProvenanceId });
      }
      return "waited";
    },
    diagnostics: {
      errors: () => [...errors],
    },
  };
}

test("the disabled scenario passes when building the extractor sends nothing", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: false, probes: false }, listener);
  const result = await sidecarDisabledScenario({ session, listener });
  assert.equal(result.passed, true);
});

test("the disabled scenario fails when a request still reached the listener", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: false, probes: true }, listener);
  const result = await sidecarDisabledScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /capabilities/);
});

test("the disabled scenario fails when the setting defaulted to enabled", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, probes: false }, listener);
  const result = await sidecarDisabledScenario({ session, listener });
  assert.equal(result.passed, false);
});

test("the disabled scenario fails when the build did not return the PDF.js extractor", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: false, builtProvenanceId: "sidecar-docling", probes: false }, listener);
  const result = await sidecarDisabledScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /provenance/);
});

test("the enabled-ignored scenario passes when enabling it still sends nothing and logs the note", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, probes: false }, listener);
  const result = await sidecarEnabledIgnoredScenario({ session, listener });
  assert.equal(result.passed, true);
  assert.match(result.detail, /logged the documented note/);
});

test("the enabled-ignored scenario fails when a request reached the listener despite being enabled", async () => {
  // Without this check the pass would be vacuous against a regression that
  // reintroduced a probe: the extractor is PDF.js either way.
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, probes: true }, listener);
  const result = await sidecarEnabledIgnoredScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /must not consult it/);
});

test("the enabled-ignored scenario fails when the build did not return the PDF.js extractor", async () => {
  const listener = fakeListener();
  const session = sidecarSession({ enabled: true, builtProvenanceId: "sidecar-docling", probes: false }, listener);
  const result = await sidecarEnabledIgnoredScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /provenance/);
});

test("the enabled-ignored scenario fails when no documented note was logged", async () => {
  // enabled: false means the stub never pushes the note, standing in for a
  // regression where enabling the setting stopped logging it.
  const listener = fakeListener();
  const session = sidecarSession({ enabled: false, probes: false }, listener);
  const result = await sidecarEnabledIgnoredScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /documented note/);
});

test("the enabled-ignored scenario fails when the settings transaction rejected the change", async () => {
  const listener = fakeListener();
  const session = {
    evaluate: async (expression) => {
      if (expression.includes("getBuffer")) return JSON.stringify([]);
      if (expression.includes("settingsChanges.changeValue")) return "ERROR: Invalid sidecar configuration";
      return "waited";
    },
    diagnostics: { errors: () => [] },
  };
  const result = await sidecarEnabledIgnoredScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /Invalid sidecar configuration/);
});

test("the enabled-ignored scenario fails when the plugin exposes no logger buffer", async () => {
  const listener = fakeListener();
  const session = {
    evaluate: async (expression) => (expression.includes("getBuffer") ? JSON.stringify(null) : "waited"),
    diagnostics: { errors: () => [] },
  };
  const result = await sidecarEnabledIgnoredScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /logger buffer/);
});

test("the enabled-ignored scenario fails when enabling it raised a renderer error", async () => {
  const listener = fakeListener();
  const session = sidecarSession(
    {
      enabled: true,
      probes: false,
      errorsOnChange: [{ source: "console", level: "error", text: "sidecar settings write exploded" }],
    },
    listener,
  );
  const result = await sidecarEnabledIgnoredScenario({ session, listener });
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

test("a scenario that walks one page may report each of its checks separately", async () => {
  const results = await runScenarios([
    async () => [
      { name: "a", passed: true, detail: "fine" },
      { name: "b", passed: false, detail: "broken" },
    ],
    async () => ({ name: "c", passed: true, detail: "fine" }),
  ]);
  assert.equal(results.passed, false);
  assert.deepEqual(results.scenarios.map((s) => s.name), ["a", "b", "c"]);
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

/** Stand-in for startModelsListener: a fixed origin, model list and request count. */
function fakeModelsListener(models = ["stub-model-a", "stub-model-b"], requestCount = 1) {
  return {
    origin: "http://127.0.0.1:45999",
    models,
    requests: () => Array.from({ length: requestCount }, () => ({ method: "GET", path: "/v1/models" })),
  };
}

const modelsAnswers = ({ visible = true, options = ["", "stub-model-a", "stub-model-b"] } = {}) => [
  ["settingsChanges.changeValue", "ok"],
  ["app.setting.open", "opened"],
  ["setTimeout", "waited"],
  ["button.click()", "ok"],
  ["is-visible", JSON.stringify({ visible, options })],
];

test("the get-models scenario passes when the select lists every fetched model despite an unrelated typed name", async () => {
  const listener = fakeModelsListener();
  const session = fakeSession(modelsAnswers());
  const result = await getModelsScenario({ session, listener });
  assert.equal(result.passed, true);
  assert.match(result.detail, /2/);
});

test("the get-models scenario fails when the select stayed hidden", async () => {
  const listener = fakeModelsListener();
  const session = fakeSession(modelsAnswers({ visible: false }));
  const result = await getModelsScenario({ session, listener });
  assert.equal(result.passed, false);
});

test("the get-models scenario fails when a fetched model is missing from the select", async () => {
  // This is the regression this scenario exists to catch: a <datalist>
  // filtered by the typed-but-unrelated model name would hide it this way.
  const listener = fakeModelsListener();
  const session = fakeSession(modelsAnswers({ options: ["", "stub-model-a"] }));
  const result = await getModelsScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /stub-model-b/);
});

test("the get-models scenario fails when clicking Get models never reached the listener", async () => {
  const listener = fakeModelsListener(["stub-model-a", "stub-model-b"], 0);
  const session = fakeSession(modelsAnswers());
  const result = await getModelsScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /never reached/);
});

test("the get-models scenario reports honestly when no Model row is found", async () => {
  const listener = fakeModelsListener();
  const session = fakeSession([
    ["settingsChanges.changeValue", "ok"],
    ["app.setting.open", "opened"],
    ["setTimeout", "waited"],
    ["button.click()", "ERROR: no Model row found in the settings tab"],
  ]);
  const result = await getModelsScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /no Model row/);
});

test("the get-models scenario fails when the settings transaction rejected pointing at the listener", async () => {
  const listener = fakeModelsListener();
  const session = fakeSession([
    ["settingsChanges.changeValue", "ERROR: Invalid LLM configuration"],
  ]);
  const result = await getModelsScenario({ session, listener });
  assert.equal(result.passed, false);
  assert.match(result.detail, /Invalid LLM configuration/);
});
