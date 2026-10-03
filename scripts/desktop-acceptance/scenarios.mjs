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

/**
 * Builds the full-text index's PDF extractor the same way indexing does
 * (`buildFullTextExtractor` in plugin/main.ts). Since title+abstract indexing
 * replaced full-text chunking (docs/helm/2026-09-02-directions-inside-
 * topics), this always returns a PDF.js extractor and never consults the
 * sidecar; `provenanceId` is how a scenario tells which engine it got.
 */
async function buildExtractor(evaluate) {
  const raw = await evaluate(`(() => {
    try {
      const built = app.plugins.plugins["arxiv-daily"].buildFullTextExtractor();
      return JSON.stringify({ provenanceId: built?.provenance?.id ?? null });
    } catch (error) {
      return "ERROR: " + (error?.message ?? String(error));
    }
  })()`);
  if (typeof raw === "string" && raw.startsWith("ERROR:")) return { error: raw.slice(7).trim() };
  return JSON.parse(raw);
}

/** The real settings transaction path: validates the loopback URLs, persists
 * the change and cancels in-flight work, none of which a direct field
 * assignment would exercise. `enabled` is omitted to leave the toggle as-is. */
async function pointSidecarAt(evaluate, listener, { enabled } = {}) {
  const raw = await evaluate(`(async () => {
    try {
      const plugin = app.plugins.plugins["arxiv-daily"];
      await plugin.settingsChanges.changeValue("pdfParserSidecar.capabilitiesUrl", ${JSON.stringify(listener.capabilitiesUrl)});
      await plugin.settingsChanges.changeValue("pdfParserSidecar.parseUrl", ${JSON.stringify(listener.parseUrl)});
      ${enabled === undefined ? "" : `await plugin.settingsChanges.changeValue("pdfParserSidecar.enabled", ${JSON.stringify(enabled)});`}
      return "changed";
    } catch (error) {
      return "ERROR: " + (error?.message ?? String(error));
    }
  })()`);
  if (typeof raw === "string" && raw.startsWith("ERROR:")) return { error: raw.slice(7).trim() };
  return { ok: true };
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
 * Proves the sidecar sends nothing when disabled (its default): with the
 * endpoints pointed at a listener we control and the feature off, building the
 * full-text index's PDF extractor performs no request at all.
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

  const pointed = await pointSidecarAt(evaluate, listener);
  if (pointed.error) {
    return fail(name, `could not point the sidecar at the listener: ${pointed.error}`);
  }

  const before = listener.requests().length;
  const built = await buildExtractor(evaluate);
  if (built.error) return fail(name, `building the full-text extractor threw: ${built.error}`);
  if (built.provenanceId !== "obsidian-pdfjs") {
    return fail(name, `expected the PDF.js extractor, got provenance ${JSON.stringify(built.provenanceId)}`);
  }

  const sent = listener.requests().slice(before);
  if (sent.length > 0) {
    return fail(
      name,
      `the sidecar is disabled but ${sent.length} request(s) reached it: ${sent.map((r) => `${r.method} ${r.path}`).join(", ")}`,
    );
  }
  return pass(name, `disabled, and building the full-text extractor sent nothing to ${listener.origin}`);
}

/**
 * Proves the documented boundary end to end: title+abstract indexing
 * (docs/helm/2026-09-02-directions-inside-topics) replaced the structured
 * parser/parserSelector sidecar assembly on the index path, which was that
 * assembly's only caller. The sidecar settings and client code were
 * deliberately left in place pending a separate removal decision, so the UI
 * can still be switched on — but indexing must never act on it. This
 * replaces the old "probe fails, falls back to PDF.js" scenario, which
 * exercised a probe (`buildFullTextDocumentParser`) that no longer exists.
 *
 * Enabling it (through the real settings transaction) and pointing it at a
 * listener we control, rather than merely reading the setting back, is what
 * makes the absence meaningful: the harness proves indexing ignores a sidecar
 * that is actually reachable, not one that was never configured.
 */
export async function sidecarEnabledIgnoredScenario({ session, listener }) {
  const name = "sidecar-enabled-ignored-by-index";
  const { evaluate, diagnostics } = session;
  const entriesBefore = diagnostics.entries().length;
  const requestsBefore = listener.requests().length;

  const pointed = await pointSidecarAt(evaluate, listener, { enabled: true });
  if (pointed.error) {
    return fail(name, `the settings transaction rejected the change: ${pointed.error}`);
  }

  const built = await buildExtractor(evaluate);
  if (built.error) return fail(name, `building the full-text extractor threw: ${built.error}`);
  if (built.provenanceId !== "obsidian-pdfjs") {
    return fail(
      name,
      `expected the PDF.js extractor despite the sidecar being enabled, got provenance ${JSON.stringify(built.provenanceId)}`,
    );
  }

  const sent = listener.requests().slice(requestsBefore);
  if (sent.length > 0) {
    return fail(
      name,
      `the sidecar is enabled but indexing must not consult it, yet ${sent.length} request(s) reached it: ${sent.map((r) => `${r.method} ${r.path}`).join(", ")}`,
    );
  }

  const introduced = diagnostics.entries().slice(entriesBefore);
  const noted = introduced.some((entry) => /not used for indexing/.test(entry.text));
  if (!noted) {
    return fail(
      name,
      "enabling the sidecar did not log the documented note that indexing does not consult it",
    );
  }

  return pass(
    name,
    `enabled through the settings transaction and pointed at ${listener.origin}, building the full-text extractor still sent nothing, still returned PDF.js, and logged the documented note`,
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
