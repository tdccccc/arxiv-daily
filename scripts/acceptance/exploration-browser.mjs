import { mkdir, readdir, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { collapseOptionSets, createExplorationRedactor, ExplorationActionError, parseExplorationAction, PRESS_NAVIGATION_KEYS } from "./exploration.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const cliRequire = createRequire(path.resolve(here, "../../apps/cli/package.json"));

/** Browser capability restrictions apply to observed AND live controls. */
export function classifyControlRestriction(control, allowedUrl) {
  const field = control.field || "";
  const name = control.name || "";
  if (control.type === "password" || /api.?key|password|secret|token/i.test(field)) return "credentials are excluded from exploration";
  if (control.type === "file") return "file selection is outside the browser capability";
  if (/vault|root|folder|path|dailyDir|papersDir/i.test(field)) return "filesystem locations must remain owned by the fixture";
  if (/base.?url|endpoint|^provider$|^embedding\.|^pdfParserSidecar\.|^email\.|^schedule\./i.test(field)) return "external service and scheduling settings are fixed by the fixture";
  if (/email|library-connect|library-reconnect|library-change|reveal-secret|show-secret/i.test(control.dataAction || "")) return "external delivery, credentials, and folder selection are excluded";
  if (/choose folder|select folder|选择文件夹|选择目录|显示密钥|show.*key/i.test(name)) return "folder selection and credential disclosure are excluded";
  if (control.href) {
    try {
      const base = new URL(allowedUrl), target = new URL(control.href, base);
      if (target.origin !== base.origin || !target.pathname.startsWith(base.pathname) || !["http:", "https:"].includes(target.protocol)) return "link leaves the isolated workbench";
    } catch { return "link is not a permitted local address"; }
  }
  return undefined;
}

/**
 * Describes one live DOM node the same way for every caller that needs to re-check it against
 * `classifyControlRestriction` right before an effect fires: the ref-based live re-validation that
 * click/fill/select already did (previously an inline arrow function here) and, since P1 Chunk 2 of
 * docs/helm/2026-10-13-exploration-safety-and-coverage, `press`'s own focus check below — `press` has
 * no ref to resolve a target from (it goes to whatever has focus), so it calls this with no argument and
 * gets `document.activeElement` instead. One shape, one place it's built, so the two call sites can never
 * drift into checking different things. `outsideWorkbench` additionally flags focus that has left the
 * app's own root entirely (not just left the currently open dialog, which `outsideDialog` already
 * covers) — `document.body` itself is exempted because that is also the page's neutral "nothing is
 * explicitly focused yet" state (e.g. right after a fresh load/reload, before the model's first action),
 * not an escape, and a keyboard-only persona's very first `press` (typically Tab) must still work from it.
 */
function liveControlDescriptor(node = document.activeElement) {
  return { tag: node.tagName.toLowerCase(), type: node.getAttribute("type") || "", field: node.getAttribute("name") || "",
    name: node.getAttribute("aria-label") || node.textContent || "", href: node.getAttribute("href") || "",
    dataAction: node.getAttribute("data-settings-action") || node.getAttribute("data-settings") || node.getAttribute("data-action") || "", disabled: node.disabled === true,
    readOnly: node.readOnly === true, outsideDialog: Boolean(document.querySelector("dialog[open]") && !node.closest("dialog[open]")),
    outsideWorkbench: node !== document.body && !node.closest(".workbench") };
}

function contained(root, file) {
  const relative = path.relative(root, file);
  return relative !== "" && !relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative);
}

// Exported so exploration-tasks/workbench.mjs's `goalTasks(fixture)` can read the same fixture-contained
// settings this driver's own observe()/inspect() rely on, without a second copy of these checks drifting
// out of sync.
export async function assertFixture(fixture) {
  if (!fixture?.root || !fixture.vaultRoot || !fixture.configPath) throw new Error("Exploration needs an owned temporary fixture");
  const root = await realpath(fixture.root);
  for (const file of [fixture.vaultRoot, fixture.configPath]) if (!contained(root, await realpath(file))) throw new Error("Exploration fixture paths must remain under its owned temporary root");
}

export async function settings(fixture) {
  const { parse } = cliRequire("smol-toml");
  const config = parse(await readFile(fixture.configPath, "utf8"));
  if (path.resolve(config.vault_root) !== path.resolve(fixture.vaultRoot)) throw new Error("Fixture vault location changed outside the exploration contract");
  return config;
}

async function preferences(fixture) {
  try { return JSON.parse(await readFile(path.join(path.dirname(fixture.configPath), "workbench-ui.json"), "utf8")); }
  catch (error) { if (error.code === "ENOENT") return { sidebarCollapsed: false }; throw error; }
}

/**
 * Rewrites the fixture's CLI TOML like a second, independent process would — never through the
 * workbench's own settings save path — so an exploration task (P3 Chunk 5's `external-config-conflict`)
 * can simulate an out-of-band edit while the model's own settings dialog is open, and the server's
 * existing revision check (`saveWorkbenchSettings`'s `assertRevision`, apps/cli/src/workbench/settings.ts)
 * has something real to detect. `mutate` edits the parsed table in place.
 */
export async function writeConfigPatch(fixture, mutate) {
  const { parse, stringify } = cliRequire("smol-toml");
  const table = parse(await readFile(fixture.configPath, "utf8"));
  mutate(table);
  await writeFile(fixture.configPath, stringify(table));
}

function browserObservation(id) {
  const modal = Array.from(document.querySelectorAll("dialog[open]")).at(-1);
  const surface = modal || document.body;
  const visible = node => {
    const rect = node.getBoundingClientRect(), style = getComputedStyle(node);
    return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0" && !node.closest("[hidden],[inert]");
  };
  const controls = [];
  for (const node of surface.querySelectorAll('button,summary,a[href],input:not([type="hidden"]),textarea,select,[role="button"],[role="tab"],[role="checkbox"],[role="combobox"],[tabindex]')) {
    if (!visible(node) || controls.length >= 220) continue;
    const rect = node.getBoundingClientRect();
    const field = node.getAttribute("name") || "";
    const labelIds = (node.getAttribute("aria-labelledby") || "").split(/\s+/).filter(Boolean);
    // P4 Chunk 3 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/04-workbench-library-tasks.md):
    // a plain button's own rendered text (e.g. settings.ts's `button(action, label)` helper, which sets
    // no aria-label of its own) must win over the `[data-setting-name]` ancestor fallback below, not lose
    // to it. Found on the real host while wiring the library-build task: every button inside the "Personal
    // library" settings row ("Choose folder"/"Retry preparation"/"Revoke"/...) reported the row's own name
    // ("Library") instead of its own text, because the row fallback used to be checked first — so a model
    // (or a goal task's verify()) could never tell those buttons apart by name. The same bug hit every
    // other settings row with more than one plain-text button (e.g. a research topic's "Add direction" /
    // "Remove direction" / "Delete" all reporting the topic's own name). `[data-setting-name]` still
    // applies afterward, for a control that truly has no name of its own (an input/select the product
    // forgot to label).
    const ownButtonText = node.tagName === "BUTTON" ? node.textContent.trim() : "";
    const label = node.getAttribute("aria-label") || labelIds.map(labelId => document.getElementById(labelId)?.textContent || "").join(" ") ||
      Array.from(node.labels || []).map(item => item.textContent).join(" ") || ownButtonText ||
      node.closest("[data-setting-name]")?.getAttribute("data-setting-name") || node.getAttribute("title") || field || node.textContent;
    const sensitive = node.type === "password" || /api.?key|password|secret|token/i.test(field);
    const ref = `${id}-${controls.length + 1}`;
    node.setAttribute("data-acceptance-ref", ref);
    controls.push({ ref, tag: node.tagName.toLowerCase(), type: node.getAttribute("type") || "", role: node.getAttribute("role") || "",
      name: String(label || "").replace(/\s+/g, " ").trim().slice(0, 180), field,
      disabled: node.disabled === true || node.getAttribute("aria-disabled") === "true", readOnly: node.readOnly === true,
      ...(sensitive ? { sensitive: true } : {}),
      ...(typeof node.value === "string" ? { value: sensitive ? "" : node.value.slice(0, 1000) } : {}),
      ...(typeof node.checked === "boolean" ? { checked: node.checked } : {}),
      ...(node.hasAttribute("href") ? { href: node.getAttribute("href").slice(0, 1000) } : {}),
      ...(node.tagName === "SELECT" ? { options: Array.from(node.options).slice(0, 60).map(option => ({ value: option.value.slice(0, 256), label: option.text.slice(0, 96), disabled: option.disabled })) } : {}),
      dataAction: node.getAttribute("data-settings-action") || node.getAttribute("data-settings") || node.getAttribute("data-action") || "",
      paperKey: node.getAttribute("data-paper") || "",
      purpose: node.classList.contains("sidebar-toggle") ? "sidebar-toggle" : "",
      rect: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) },
      inViewport: rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth,
    });
  }
  const root = document.querySelector(".workbench");
  return { id, title: document.title, text: surface.innerText.slice(0, 16000), controls,
    layout: { rootMounted: Boolean(root), view: root?.getAttribute("data-view") || "", dialogOpen: Boolean(modal),
      sidebarCollapsed: root?.classList.contains("sidebar-collapsed") === true, viewport: { width: innerWidth, height: innerHeight } } };
}

/**
 * P4 Chunk 2 Fix 1 (see reset()'s own comment): reads the workbench server's currently-held config
 * revision through its own GET /api/settings and, when settings exist at all, performs one real, no-op
 * settings save (empty `values`, current `revision`) through its own POST /api/settings — syncing the
 * server's in-memory config to whatever is on disk right now, without reverting disk to any earlier
 * baseline. Factored out of reset() so `startWorkbenchLibraryFixture` (exploration-tasks/
 * workbench-library.mjs) can call the exact same resync after its own out-of-band CLI writes, through
 * the driver's own `resyncSettings()` below, instead of a second copy of this GET/POST dance.
 */
async function resyncConfigRevision(page, url) {
  const current = await page.request.get(new URL("api/settings", url).href, { timeout: 5000 });
  if (!current.ok()) throw new Error(`Could not read workbench settings to resync the config revision: ${current.status()}`);
  const { revision } = await current.json();
  if (revision !== null) {
    const saved = await page.request.post(new URL("api/settings", url).href, { data: { revision, values: {} }, timeout: 5000 });
    if (!saved.ok()) throw new Error(`Could not resync the workbench config revision: ${saved.status()}`);
  }
}

/**
 * P4 Chunk 2 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/04-workbench-library-tasks.md):
 * restores the fixture's own personal-library state to its baseline — no `[library]`/`[embedding]` TOML
 * table (the fixture never ships either; only `startWorkbenchLibraryFixture` or a model's own
 * library-build/remote-embedding flow ever adds them), no `origin: "library"` direction under
 * `arxiv.topics[].directions` (a manually-set direction, e.g. the fixture's own baseline one, is left
 * alone — this is a surgical restore, not a revert-to-snapshot, so a legitimately saved setting from an
 * earlier task, e.g. settings-persistence's own `max_daily_papers`, survives exactly like reset()'s own
 * comment above already guarantees for the config resync), no leftover `.index/personal-library-*`
 * catalog/search-index/profile state (same derivation exploration-obsidian.mjs's `clearLibraryState` uses
 * for Obsidian, and the same directory the real CLI itself writes to — confirmed by library-runtime-
 * cache.mjs's real-host probe), and no library PDF beyond whatever was already under `vault/library` when
 * this driver was built (`baselineLibraryFiles`, e.g. `startWorkbenchLibraryFixture`'s own extra
 * clusterable PDF). Skips the TOML rewrite entirely when there is nothing to remove, so the vast majority
 * of non-library tasks' reset() calls never touch the config file at all.
 *
 * P4 Chunk 4 fix: a topic that `library-accept-direction` auto-creates (acceptProposedTopics,
 * packages/core/src/settings/accept-proposed-topics.ts — a new topic only when no existing one shares
 * its name) holds only library-origin directions, so filtering those out above can leave it with zero
 * directions. Unlike the product's own proposal-side `removePersonalLibraryDirectionCandidate`, which
 * drops an emptied topic, this filter used to leave the empty shell behind — found on the real host
 * running `library-accept-direction` then `library-edit-candidate` in one pass: the leftover topic's
 * own name ("Research evidence 1") collided with the next real `library propose` call's own freshly
 * generated topic of the same name, which `decodeOrganizedTopics` (personal-library-topic-
 * organization.ts) rejects as "reference-out-of-scope" (a new topic may not share an existing one's
 * name). Splicing out only a topic this filter emptied — never one that still has its own manual
 * direction, like the fixture's baseline "Research" topic, nor one that had no directions to begin
 * with — closes this the same way the product does.
 */
async function clearFixtureLibraryState(fixture, baselineLibraryFiles) {
  const { parse } = cliRequire("smol-toml");
  const table = parse(await readFile(fixture.configPath, "utf8"));
  const hasLibraryOriginDirection = (table.arxiv?.topics ?? []).some(topic => (topic.directions ?? []).some(direction => direction.origin === "library"));
  if (table.library || table.embedding || hasLibraryOriginDirection) {
    await writeConfigPatch(fixture, current => {
      delete current.library;
      delete current.embedding;
      const emptied = new Set();
      for (const topic of current.arxiv?.topics ?? []) {
        if (!Array.isArray(topic.directions)) continue;
        const kept = topic.directions.filter(direction => direction.origin !== "library");
        if (kept.length === 0 && topic.directions.length > 0) emptied.add(topic);
        topic.directions = kept;
      }
      if (Array.isArray(current.arxiv?.topics)) current.arxiv.topics = current.arxiv.topics.filter(topic => !emptied.has(topic));
    });
  }
  const dailyDir = table.output?.daily_dir;
  if (typeof dailyDir === "string") {
    const indexDir = path.join(fixture.vaultRoot, path.dirname(dailyDir), ".index");
    const indexEntries = await readdir(indexDir).catch(() => []);
    await Promise.all(indexEntries.filter(entry => entry.startsWith("personal-library-")).map(entry => rm(path.join(indexDir, entry), { recursive: true, force: true })));
  }
  const libraryDir = path.join(fixture.vaultRoot, "library");
  const currentLibraryFiles = await readdir(libraryDir).catch(() => []);
  await Promise.all(currentLibraryFiles.filter(entry => !baselineLibraryFiles.has(entry)).map(entry => rm(path.join(libraryDir, entry), { recursive: true, force: true })));
}

export async function createExplorationDriver({ page, context, url, fixture, artifactDir, diagnostics, redact = createExplorationRedactor(), signal }) {
  await assertFixture(fixture);
  await mkdir(artifactDir, { recursive: true, mode: 0o700 });
  // P4 Chunk 2: captured once, at driver creation, so reset()'s own library-PDF cleanup (below) can tell
  // "already there when this task library started" apart from "a task's own prepare() added this" —
  // mirrors exploration-obsidian.mjs's own `baselineLibraryFiles`.
  const baselineLibraryFiles = new Set(await readdir(path.join(fixture.vaultRoot, "library")).catch(() => []));
  await page.locator(".workbench").waitFor({ state: "visible", timeout: 12000 });
  const base = new URL(url), pageErrors = [], boundaryEvents = [];
  let version = 0, latest;
  const inScope = target => { try { const value = new URL(target); return value.origin === base.origin && value.pathname.startsWith(base.pathname); } catch { return false; } };
  const assertScope = () => { signal?.throwIfAborted(); if (!inScope(page.url())) throw new Error("Browser left the isolated workbench scope"); };
  const onError = error => pageErrors.push(redact(error.message));
  const onPage = popup => { if (popup !== page) { boundaryEvents.push("Popup blocked: exploration stays in one local page"); void popup.close().catch(() => {}); } };
  const guard = async route => {
    if (inScope(route.request().url())) return route.fallback();
    boundaryEvents.push("Network request outside the isolated workbench was blocked");
    return route.abort("blockedbyclient");
  };
  page.on("pageerror", onError); context.on("page", onPage);
  await context.route("**/*", guard);
  const safe = value => {
    if (typeof value === "string") return redact(value);
    if (Array.isArray(value)) return value.map(safe);
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, safe(item)]));
    return value;
  };
  return {
    async observe() {
      assertScope();
      const snapshot = await page.evaluate(browserObservation, `s${++version}`);
      snapshot.url = page.url().replace(base.href, "./");
      for (const control of snapshot.controls) {
        const restriction = classifyControlRestriction(control, url);
        if (restriction) { control.restricted = restriction; if (control.sensitive) control.value = ""; }
      }
      // P4 Chunk 2 Fix 3: dedupe repeated <select> option lists (e.g. many identical arXiv-category rows)
      // into one shared observation.optionSets map, in Node rather than inside browserObservation's own
      // page.evaluate() scan, so the in-page DOM-reading code stays untouched.
      snapshot.optionSets = collapseOptionSets(snapshot.controls);
      snapshot.errors = [...new Set([...pageErrors, ...(diagnostics?.pageErrors || []).map(redact)])];
      if (!snapshot.layout.rootMounted) snapshot.errors.push("Workbench root disappeared from the page");
      snapshot.boundaryEvents = [...boundaryEvents];
      latest = safe(snapshot);
      return latest;
    },
    async execute(action, observation, options = {}) {
      assertScope();
      if (latest?.id !== observation.id) throw new ExplorationActionError("Stale observation; capture a new page state");
      parseExplorationAction(JSON.stringify(action), latest);
      options.signal?.throwIfAborted();
      const onAbort = () => { void page.close().catch(() => {}); };
      options.signal?.addEventListener("abort", onAbort, { once: true });
      try {
        let target;
        if (action.ref) {
          target = page.locator(`[data-acceptance-ref="${action.ref}"]`);
          if (await target.count() !== 1 || !await target.isVisible()) throw new ExplorationActionError("Ref disappeared or is no longer visible; re-observe the page");
          const live = await target.evaluate(liveControlDescriptor);
          const restriction = classifyControlRestriction(live, url);
          if (restriction || live.disabled || live.outsideDialog) throw new ExplorationActionError(restriction || "The control is no longer interactive");
          // optionSets is carried through from the latest observation (not rebuilt here) so a select's
          // optionsRef (P4 Chunk 2 Fix 3) still resolves during this live re-validation pass.
          parseExplorationAction(JSON.stringify(action), { controls: [{ ...latest.controls.find(control => control.ref === action.ref), ...live }], optionSets: latest.optionSets });
        }
        if (action.type === "click") await target.click({ timeout: 6000 });
        else if (action.type === "fill") await target.fill(action.value, { timeout: 6000 });
        else if (action.type === "select") await target.selectOption(action.value, { timeout: 6000 });
        else if (action.type === "reload") { await page.reload({ waitUntil: "domcontentloaded", timeout: 8000 }); await page.locator(".workbench").waitFor({ state: "visible", timeout: 6000 }); }
        else if (action.type === "back") await page.goBack({ waitUntil: "domcontentloaded", timeout: 8000 });
        // P4 Chunk 2 Fix 2: the workbench equivalent of Obsidian's "home" — a safe, known starting view —
        // is just reloading the start URL; unlike `reload`, it is always safe to call from any state
        // (including a page that has somehow lost its own root) because it is a fresh navigation, not a
        // same-document reload.
        else if (action.type === "home") { await page.goto(url, { waitUntil: "domcontentloaded", timeout: 8000 }); await page.locator(".workbench").waitFor({ state: "visible", timeout: 6000 }); }
        else if (action.type === "wait") await new Promise(resolve => setTimeout(resolve, action.milliseconds));
        // P1 Chunk 2 (docs/helm/2026-10-13-exploration-safety-and-coverage): `press` has no ref, so unlike
        // click/fill/select above it was never re-checking the live focused element against
        // classifyControlRestriction before acting — Tab-to-focus + Space/Enter could flip the
        // schedule.enabled/email.enabled switch or activate a restricted control with zero checks. Apply
        // the exact same policy here, against whatever document.activeElement currently is, before
        // sending any key event. Navigation keys skip the control policy (see PRESS_NAVIGATION_KEYS).
        else if (action.type === "press") {
          const focused = await page.evaluate(liveControlDescriptor);
          if (focused.outsideWorkbench) throw new ExplorationActionError("Focus left the isolated workbench; the key was not sent");
          if (!PRESS_NAVIGATION_KEYS.has(action.key)) {
            const restriction = classifyControlRestriction(focused, url);
            if (restriction || focused.disabled || focused.outsideDialog) throw new ExplorationActionError(restriction || "The focused control is no longer interactive; the key was not sent");
          }
          await page.keyboard.press(action.key);
        }
        else if (action.type === "scroll") {
          if (target) await target.hover({ timeout: 6000 });
          else { const region = page.locator("dialog[open] .settings-content, .reading-pane").last(); if (await region.isVisible()) await region.hover(); }
          await page.mouse.wheel(0, action.direction === "down" ? action.pixels : -action.pixels);
        }
        options.signal?.throwIfAborted(); assertScope();
        return { ok: true };
      } finally { options.signal?.removeEventListener("abort", onAbort); }
    },
    async reset({ signal: resetSignal } = {}) {
      assertScope();
      resetSignal?.throwIfAborted();
      // P4 Chunk 2 Fix 1: a prior task may have edited the CLI TOML out-of-band (writeConfigPatch, used
      // only by the external-config-conflict task to simulate a second process editing the config while
      // the workbench is open) without going through the server's own save path
      // (apps/cli/src/workbench/settings.ts's saveWorkbenchSettings / server.ts's activate()) — the only
      // place that updates the in-memory `config` launch.ts's createWorkbenchConfigGuard compares disk
      // against. Left alone, that mismatch (and its "配置文件已在外部修改..." 409 guard) leaks into every
      // later task and persona in the same session (observed in the P4 Chunk 1 real-model run). This is a
      // general reset-level guarantee, not a cleanup specific to that one task: every task boundary reads
      // the config currently on disk and, if settings exist at all, performs one real, no-op settings save
      // (empty `values`, current `revision`) through the server's own POST /api/settings — a legitimate
      // save, not a bypass — so the server's held config always matches disk again before the next task's
      // first observation. Deliberately resyncs server-to-disk rather than reverting disk to some earlier
      // baseline: the latter would also discard a legitimate settings change an earlier task already made
      // and verified (e.g. settings-persistence's maxDailyPapers) through the real save path, which must
      // survive into later tasks. Cost measured on a real host: ~10-20ms for the two same-origin HTTP
      // calls, negligible next to the rest of reset()'s own page navigation. Factored into
      // resyncConfigRevision() (above) so startWorkbenchLibraryFixture's own out-of-band CLI seed can
      // reuse exactly this, through resyncSettings() below.
      //
      // P4 Chunk 2: library state is restored to the fixture baseline (clearFixtureLibraryState, above)
      // BEFORE this resync, not after — the clean-up itself is a direct, out-of-band disk write (like
      // the external-config-conflict task's own writeConfigPatch), so the resync must run last to pick
      // up the now-cleaned disk content; doing it the other way round would resync the server to a disk
      // state this call is about to change again, reintroducing the exact mismatch this guards against.
      await clearFixtureLibraryState(fixture, baselineLibraryFiles);
      await resyncConfigRevision(page, url);
      resetSignal?.throwIfAborted();
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15000 });
      const dialog = page.locator("dialog[open]");
      if (await dialog.count()) {
        await dialog.locator('[data-action="close-dialog"]').first().click({ timeout: 3000 }).catch(() => {});
        await dialog.waitFor({ state: "hidden", timeout: 6000 }).catch(() => {});
      }
      await page.locator(".workbench").waitFor({ state: "visible", timeout: 12000 });
      resetSignal?.throwIfAborted();
      assertScope();
    },
    async capture(name) {
      assertScope();
      if (!/^[a-z0-9-]+$/.test(name)) throw new Error("Invalid exploration artifact name");
      const file = path.join(artifactDir, `exploration-${name}.png`);
      await page.screenshot({ path: file, animations: "disabled", mask: [page.locator('input[type="password"],input[name*="apiKey"],input[name*="token"],input[name*="Token"]')] });
      return file;
    },
    async inspect() {
      assertScope();
      const config = await settings(fixture);
      return { maxDailyPapers: config.output?.max_daily_papers, preferences: await preferences(fixture) };
    },
    async dailyEvidence(date) {
      assertScope();
      const config = await settings(fixture);
      const daily = config.output?.daily_dir;
      if (typeof daily !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("Invalid fixture report scope");
      const file = path.resolve(fixture.vaultRoot, daily, `${date}.md`);
      if (!contained(path.resolve(fixture.vaultRoot), file)) throw new Error("Report path escaped the fixture");
      // A completed run guarantees writes have finished; inspect the file after that observation.
      const response = await page.request.get(new URL("api/runs/current", url).href, { timeout: 5000 });
      if (!response.ok()) throw new Error("Cannot inspect the current workbench run");
      const run = (await response.json()).run;
      let report = false;
      try {
        if (!contained(await realpath(fixture.vaultRoot), await realpath(file))) throw new Error("Report symlink escaped the fixture");
        report = (await stat(file)).isFile() && (await readFile(file, "utf8")).includes("2610.10001");
      } catch (error) { if (error.code !== "ENOENT") throw error; }
      return { report, run };
    },
    /**
     * Deterministically guarantees a completed daily report for `date` exists, through the workbench's
     * own real generation run (never a synthetic file) — mirrors the Obsidian driver's `ensureDailyReport`
     * so P3 Chunk 5's reading-journey workbench tasks (`star-filter-restore`, `generate-paper-note`) don't
     * each need their own copy of this. A no-op if the report file is already there.
     */
    async ensureDailyReport(date) {
      assertScope();
      const config = await settings(fixture);
      const daily = config.output?.daily_dir;
      if (typeof daily !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("Invalid fixture report scope");
      const file = path.resolve(fixture.vaultRoot, daily, `${date}.md`);
      if (!contained(path.resolve(fixture.vaultRoot), file)) throw new Error("Report path escaped the fixture");
      try {
        if (contained(await realpath(fixture.vaultRoot), await realpath(file)) && (await stat(file)).isFile()) return;
      } catch (error) { if (error.code !== "ENOENT") throw error; }
      const started = await page.request.post(new URL("api/runs", url).href, { data: { kind: "daily", date } });
      if (!started.ok()) throw new Error(`Could not start a fixture daily run for ${date}: ${started.status()}`);
      const { run } = await started.json();
      const deadline = Date.now() + 60000;
      while (true) {
        const current = (await (await page.request.get(new URL("api/runs/current", url).href)).json()).run;
        if (current?.id === run.id && current.status !== "running") {
          if (current.status !== "completed" && current.status !== "skipped") throw new Error(`Fixture daily run for ${date} ended as ${current.status}`);
          return;
        }
        if (Date.now() >= deadline) throw new Error(`Fixture daily run for ${date} did not finish in time`);
        await new Promise(resolve => setTimeout(resolve, 150));
      }
    },
    /** Resets a paper's star mark through the real API (not a synthetic write), at most once per call. */
    async ensurePaperUnstarred(key) {
      assertScope();
      const current = await page.request.get(new URL(`api/paper?key=${encodeURIComponent(key)}`, url).href);
      if (!current.ok()) return;
      const { paper } = await current.json();
      if (paper.priority !== "high") return;
      const reset = await page.request.post(new URL("api/paper/mark", url).href, { data: { key, action: "star", value: false, expected: paper.priority } });
      if (!reset.ok()) throw new Error(`Could not reset the star mark for ${key}: ${reset.status()}`);
    },
    // P4 Chunk 2: exposed so exploration-tasks/workbench-library.mjs's `startWorkbenchLibraryFixture`
    // can read the same fixture paths/env this driver already validated (assertFixture, above) without a
    // second copy of them, and resync the server after its own out-of-band CLI writes through the exact
    // same path reset() uses (resyncConfigRevision) rather than a second implementation of it.
    fixture,
    async resyncSettings() { assertScope(); await resyncConfigRevision(page, url); },
    async dispose() { page.off("pageerror", onError); context.off("page", onPage); await context.unroute("**/*", guard); },
  };
}
