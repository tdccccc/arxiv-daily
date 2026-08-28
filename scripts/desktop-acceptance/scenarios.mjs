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
 * Proves the optional sidecar is inert unless explicitly enabled: the setting
 * is off, and no request reached either configured endpoint.
 */
export async function sidecarDisabledScenario({ session, requests }) {
  const name = "sidecar-disabled-by-default";
  const raw = await session.evaluate(SETTINGS_EXPRESSION);
  if (!raw) return fail(name, "the plugin exposes no pdfParserSidecar settings section");
  const settings = JSON.parse(raw);
  if (settings.enabled !== false) {
    return fail(name, `pdfParserSidecar.enabled is ${JSON.stringify(settings.enabled)}, expected false`);
  }
  const endpoints = [settings.capabilitiesUrl, settings.parseUrl].filter(Boolean);
  const contacted = requests.filter((url) => endpoints.some((endpoint) => endpoint && url.startsWith(endpoint)));
  if (contacted.length > 0) {
    return fail(name, `sidecar is disabled but these requests were sent: ${contacted.join(", ")}`);
  }
  return pass(
    name,
    `disabled, and none of ${requests.length} observed request(s) touched ${endpoints.join(" or ") || "any endpoint"}`,
  );
}

/**
 * Proves that enabling the sidecar against an unreachable endpoint degrades
 * quietly instead of surfacing an error to the user.
 */
export async function sidecarFallbackScenario({ session, unreachablePort, settleMs = 4000 }) {
  const name = "sidecar-unreachable-falls-back";
  const { evaluate, diagnostics } = session;
  const before = diagnostics.errors().length;

  const base = `http://127.0.0.1:${unreachablePort}`;
  await evaluate(`(async () => {
    const plugin = app.plugins.plugins["arxiv-daily"];
    plugin.settings.pdfParserSidecar.enabled = true;
    plugin.settings.pdfParserSidecar.capabilitiesUrl = ${JSON.stringify(`${base}/capabilities`)};
    plugin.settings.pdfParserSidecar.parseUrl = ${JSON.stringify(`${base}/parse`)};
    return "configured";
  })()`);
  await wait(evaluate, settleMs);

  const introduced = diagnostics.errors().slice(before);
  if (introduced.length > 0) {
    return fail(name, `enabling an unreachable sidecar raised: ${introduced.map((e) => e.text).join("; ")}`);
  }
  return pass(name, `enabled against unreachable ${base} without raising a renderer error`);
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

/** Run scenarios in order, converting a thrown scenario into a reported failure. */
export async function runScenarios(scenarios) {
  const results = [];
  for (const [index, scenario] of scenarios.entries()) {
    try {
      results.push(await scenario());
    } catch (error) {
      results.push(fail(`scenario-${index + 1}`, `threw: ${error.message}`));
    }
  }
  return { passed: results.every((result) => result.passed), scenarios: results };
}
