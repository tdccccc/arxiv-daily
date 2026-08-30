/**
 * The four desktop acceptance scenarios P7 could not produce by hand, each an
 * independent assertion over one real Obsidian session.
 */

const SETTINGS_EXPRESSION = 'JSON.stringify(app.plugins.plugins["arxiv-daily"]?.settings?.pdfParserSidecar ?? null)';
const SETTINGS_SECTIONS_EXPRESSION = 'Object.keys(app.plugins.plugins["arxiv-daily"]?.settings ?? {})';

const REQUIRED_SETTINGS_SECTIONS = [
  "llm",
  "arxiv",
  "output",
  "schedule",
  "advanced",
  "email",
  "pdfParserSidecar",
];

async function buildParser(evaluate) {
  const raw = await evaluate(`(async () => {
    try {
      const built = await app.plugins.plugins["arxiv-daily"].buildFullTextDocumentParser();
      return JSON.stringify({
        parser: Boolean(built?.parser),
        parserSelector: Boolean(built?.parserSelector),
      });
    } catch (error) {
      return "ERROR: " + (error?.message ?? String(error));
    }
  })()`);
  if (typeof raw === "string" && raw.startsWith("ERROR:")) return { error: raw.slice(7).trim() };
  return JSON.parse(raw);
}

const wait = (evaluate, ms) => evaluate(`new Promise((resolve) => setTimeout(resolve, ${ms}))`);

function pass(name, detail) {
  return { name, passed: true, detail };
}

function fail(name, detail) {
  return { name, passed: false, detail };
}

/**
 * Proves the host honours `#page=N` rather than merely opening the file. The
 * embedded pdf.js viewer's own current page is the authority; `pdfViewer.page`
 * is read as a fallback because it is the value Obsidian sets from the subpath.
 */
export async function pdfPageLocationScenario({ session, page = 4, settleMs = 6000 }) {
  const name = "pdf-page-location";
  const { evaluate } = session;

  const pdfPath = await evaluate(
    'app.vault.getFiles().filter((f) => f.extension === "pdf" && f.stat.size < 3000000).map((f) => f.path)[0] ?? null',
  );
  if (typeof pdfPath !== "string" || pdfPath.length === 0) {
    return fail(name, "no PDF under 3 MB found in the vault, so page location cannot be exercised");
  }

  await evaluate(`app.workspace.openLinkText(${JSON.stringify(`${pdfPath}#page=${page}`)}, "", false)`);
  await wait(evaluate, settleMs);

  const rawState = await evaluate("JSON.stringify(app.workspace.activeLeaf?.getViewState?.() ?? null)");
  const viewType = rawState ? (JSON.parse(rawState)?.type ?? null) : null;
  if (viewType !== "pdf") {
    return fail(name, `expected a pdf view, the active leaf is ${JSON.stringify(viewType)}`);
  }

  const observed = await evaluate(`(() => {
    const child = app.workspace.activeLeaf?.view?.viewer?.child;
    return child?.pdfViewer?.pdfViewer?.currentPageNumber ?? child?.pdfViewer?.page ?? null;
  })()`);

  if (observed !== page) {
    return fail(
      name,
      `opened ${pdfPath} with #page=${page} but the viewer reports page ${JSON.stringify(observed)}`,
    );
  }
  return pass(name, `${pdfPath} opened at page ${page} in the embedded viewer`);
}

/**
 * Proves the optional sidecar is inert unless explicitly enabled: with the
 * endpoints pointed at a listener we control and the feature off, building the
 * parser performs no request at all.
 *
 * The listener is what makes the absence meaningful. The plugin's HTTP goes out
 * through Obsidian's `requestUrl` in the Electron main process, so watching the
 * renderer would show nothing either way.
 */
export async function sidecarDisabledScenario({ session, listener }) {
  const name = "sidecar-disabled-sends-nothing";
  const { evaluate } = session;

  const raw = await evaluate(SETTINGS_EXPRESSION);
  if (!raw) return fail(name, "the plugin exposes no pdfParserSidecar settings section");
  const settings = JSON.parse(raw);
  if (settings.enabled !== false) {
    return fail(name, `pdfParserSidecar.enabled is ${JSON.stringify(settings.enabled)}, expected false`);
  }

  const pointed = await evaluate(`(async () => {
    try {
      const plugin = app.plugins.plugins["arxiv-daily"];
      await plugin.settingsChanges.changeValue("pdfParserSidecar.capabilitiesUrl", ${JSON.stringify(listener.capabilitiesUrl)});
      await plugin.settingsChanges.changeValue("pdfParserSidecar.parseUrl", ${JSON.stringify(listener.parseUrl)});
      return "pointed";
    } catch (error) {
      return "ERROR: " + (error?.message ?? String(error));
    }
  })()`);
  if (typeof pointed === "string" && pointed.startsWith("ERROR:")) {
    return fail(name, `could not point the sidecar at the listener: ${pointed.slice(7).trim()}`);
  }

  const before = listener.requests().length;
  const built = await buildParser(evaluate);
  if (built.error) return fail(name, `building the parser threw: ${built.error}`);
  if (built.parserSelector) {
    return fail(name, `the sidecar is disabled but the build produced ${built.parserSelector}`);
  }

  const sent = listener.requests().slice(before);
  if (sent.length > 0) {
    return fail(
      name,
      `the sidecar is disabled but ${sent.length} request(s) reached it: ${sent.map((r) => `${r.method} ${r.path}`).join(", ")}`,
    );
  }
  return pass(name, `disabled, and building the parser sent nothing to ${listener.origin}`);
}

/**
 * Proves the documented probe-failure path end to end: the setting is turned on
 * through the real settings transaction, the parser build actually reaches the
 * configured endpoint, that endpoint refuses, and PDF.js is selected instead.
 *
 * The observed request matters. Building returns PDF.js whenever the sidecar is
 * off, so "got PDF.js" alone would prove nothing about a failed probe.
 */
export async function sidecarFallbackScenario({ session, listener, settleMs = 1000 }) {
  const name = "sidecar-probe-fails-to-pdfjs";
  const { evaluate, diagnostics } = session;
  const errorsBefore = diagnostics.errors().length;
  const requestsBefore = listener.requests().length;

  // The real transaction path: it validates the loopback URLs, persists the
  // change and cancels in-flight work, none of which a direct field assignment
  // would exercise.
  const changed = await evaluate(`(async () => {
    try {
      const plugin = app.plugins.plugins["arxiv-daily"];
      await plugin.settingsChanges.changeValue("pdfParserSidecar.capabilitiesUrl", ${JSON.stringify(listener.capabilitiesUrl)});
      await plugin.settingsChanges.changeValue("pdfParserSidecar.parseUrl", ${JSON.stringify(listener.parseUrl)});
      await plugin.settingsChanges.changeValue("pdfParserSidecar.enabled", true);
      return "changed";
    } catch (error) {
      return "ERROR: " + (error?.message ?? String(error));
    }
  })()`);
  if (typeof changed === "string" && changed.startsWith("ERROR:")) {
    return fail(name, `the settings transaction rejected the change: ${changed.slice(7).trim()}`);
  }

  const built = await buildParser(evaluate);
  if (built.error) return fail(name, `building the parser threw: ${built.error}`);
  await wait(evaluate, settleMs);

  const probes = listener.requests().slice(requestsBefore);
  if (probes.length === 0) {
    return fail(
      name,
      `no request reached ${listener.origin}, so the parser choice proves nothing about a failed probe`,
    );
  }
  // Class names are minified in a production bundle, so the discriminator is
  // structural: a selector means the sidecar was adopted, a bare parser means
  // the fallback was taken.
  if (built.parserSelector) {
    return fail(
      name,
      `the probe to ${listener.origin} failed but the build still produced a sidecar selector`,
    );
  }
  if (!built.parser) return fail(name, "the parser build returned neither a parser nor a selector");

  const introduced = diagnostics.errors().slice(errorsBefore);
  if (introduced.length > 0) {
    return fail(name, `the failed probe raised: ${introduced.map((entry) => entry.text).join("; ")}`);
  }

  return pass(
    name,
    `enabled through the settings transaction, ${probes.length} probe request reached ${listener.origin} and was refused, and PDF.js was selected without a renderer error`,
  );
}

/**
 * Proves settings persisted before the sidecar existed still load, gaining the
 * new section at its safe default instead of failing or enabling itself.
 */
export async function settingsMigrationScenario({ session }) {
  const name = "legacy-settings-migration";
  const raw = await session.evaluate(SETTINGS_EXPRESSION);
  if (!raw) return fail(name, "legacy settings did not gain a pdfParserSidecar section");
  const settings = JSON.parse(raw);
  if (settings.enabled !== false) {
    return fail(name, `migration produced enabled=${JSON.stringify(settings.enabled)}, expected false`);
  }
  const sections = (await session.evaluate(SETTINGS_SECTIONS_EXPRESSION)) ?? [];
  const missing = REQUIRED_SETTINGS_SECTIONS.filter((section) => !sections.includes(section));
  if (missing.length > 0) {
    return fail(name, `migrated settings are missing: ${missing.join(", ")}`);
  }
  return pass(name, `legacy settings migrated with ${sections.length} sections and the sidecar left disabled`);
}

/**
 * Run scenarios in order, converting a thrown scenario into a reported failure.
 *
 * A scenario may return several results: a walk through one settings page
 * checks several independent things, and reporting them as one verdict would
 * hide which behaviour actually broke.
 */
export async function runScenarios(scenarios) {
  const results = [];
  for (const [index, scenario] of scenarios.entries()) {
    try {
      const produced = await scenario();
      results.push(...(Array.isArray(produced) ? produced : [produced]));
    } catch (error) {
      results.push(fail(`scenario-${index + 1}`, `threw: ${error.message}`));
    }
  }
  return { passed: results.every((result) => result.passed), scenarios: results };
}
