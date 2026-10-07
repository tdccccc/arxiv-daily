/**
 * The redesigned library direction review dialog, checked in the real renderer.
 *
 * `plugin/src/library/interest-profile-modal.ts` replaced a tabbed
 * suggestions/overview layout with one default page that leads with the
 * current result and exactly one primary action: "Update suggestions" when
 * coverage is stale, "Try again" on a read failure, or "Done" when nothing
 * new is needed. The unit suite (`plugin/tests/proposal-acceptance-ui.test.ts`)
 * renders the same markup into happy-dom and proves the branching — which
 * state produces which heading and button — but happy-dom has no layout
 * engine and no Obsidian stylesheet. It cannot say whether the long-direction
 * heading actually renders unbold (the redesign demoted it from `<strong>`
 * run-on styling to `font-weight: var(--font-normal)` in `plugin/styles.css`),
 * whether that text wraps inside the dialog instead of overflowing it, or
 * whether the "Add to research topics" bar really lands after the direction
 * list in the rendered document. Those are geometric and stylistic facts, so
 * only a real renderer can settle them — which is also why this scenario
 * writes screenshots: wording and emphasis still need a person to judge.
 *
 * Every state here is reached the way a user reaches it — opening Settings
 * and clicking "Review suggestions" — with the proposal/coverage state
 * injected directly onto the plugin instance beforehand, the same pattern
 * `topic-directions.mjs` uses for its topic fixture. Driving a real
 * generation or acceptance run would need a model endpoint and a real
 * library; what only the renderer can answer is what the page does with a
 * given state, so the state is handed to it directly rather than produced by
 * a real pipeline run.
 */
import { clearViewport, setViewport } from "./cdp.mjs";
import { CLEAR_LAST_INDEX_RUN_EXPRESSION, setLastIndexRunExpression } from "./library-settings.mjs";

const PLUGIN_ID = "arxiv-daily";

/**
 * Embed a value in the JavaScript this module generates for the renderer.
 * U+2028 and U+2029 are line terminators in source text, so a string carrying
 * one would end the statement it was embedded in. Written as escapes because
 * the raw characters are invisible in an editor.
 */
const asCode = (value) => JSON.stringify(value)
  .replace(/\u2028/g, "\\u2028")
  .replace(/\u2029/g, "\\u2029");

const PLUGIN = `app.plugins.plugins[${asCode(PLUGIN_ID)}]`;

export const MODAL_SELECTOR = ".modal-container .modal.arxiv-daily-interest-review-modal";

/** Long enough to wrap several times in any panel a person would use. */
export const LONG_DIRECTION =
  "Transfer-learning evaluation protocols for low-resource instruction tuning, covering "
  + "cross-lingual benchmark construction, contamination checks against pretraining corpora, "
  + "and the reporting practices that let two papers' numbers be compared at all.";

export const SHORT_DIRECTION = "Benchmark contamination checks.";

const REVIEW_TOPIC_ID = "acceptance-review-topic";
const LONG_DIRECTION_ID = "acceptance-review-direction-long";
const SHORT_DIRECTION_ID = "acceptance-review-direction-short";

const NARROW_VIEWPORT = { width: 900, height: 900 };
const WIDE_VIEWPORT = { width: 1440, height: 900 };

const wait = (evaluate, ms) => evaluate(`new Promise((resolve) => setTimeout(resolve, ${ms}))`);

function pass(name, detail) {
  return { name, passed: true, detail };
}

function fail(name, detail) {
  return { name, passed: false, detail };
}

async function readJson(evaluate, expression) {
  const raw = await evaluate(expression);
  if (typeof raw !== "string") {
    throw new Error(`expected a JSON string from the renderer, received ${asCode(raw)}`);
  }
  return JSON.parse(raw);
}

// ── fixtures ─────────────────────────────────────────────────────────────

const BASE_PROPOSAL_FIELDS = {
  schemaVersion: 6,
  catalogInputFingerprint: "sha256:acceptance-catalog-input",
  generationContractFingerprint: "sha256:acceptance-contract",
  generatedAt: "2026-09-30T12:00:00.000Z",
};

function proposalFixture(overrides) {
  return {
    ...BASE_PROPOSAL_FIELDS,
    revision: 0,
    proposalId: "acceptance-proposal",
    scopeFingerprint: "sha256:acceptance-scope",
    identificationFingerprint: "sha256:acceptance-identification",
    catalogInputPapers: [],
    topics: [],
    coveredPaperKeys: [],
    coverageEvidence: [],
    ...overrides,
  };
}

/**
 * A minimal catalog so `generationAvailability` finds a current
 * personal-library catalog to propose from — without it, every generate
 * button (including the primary "Update suggestions" action in the stale-
 * coverage state) is disabled for "scan and load the catalog first" instead
 * of for the reason this scenario actually wants to see.
 */
export const CATALOG_FIXTURE = {
  schemaVersion: 1,
  revision: 1,
  scopeFingerprint: "sha256:acceptance-scope",
  identificationFingerprint: "sha256:acceptance-identification",
  updatedAt: "2026-09-30T12:00:00.000Z",
  lastScan: null,
  files: {},
  papers: {
    "paper-stale-1": {
      paperKey: "paper-stale-1",
      source: "arxiv",
      externalId: "2501.00001",
      title: "A representative paper for the desktop acceptance fixture",
      authors: ["Acceptance Author"],
      abstract: "A minimal abstract, just enough to make the catalog non-empty.",
      published: "2025-01-01T00:00:00.000Z",
      updated: "2025-01-01T00:00:00.000Z",
      primaryCategory: "astro-ph.GA",
      categories: ["astro-ph.GA"],
      evidenceDepth: "metadata-and-abstract",
      filePaths: [],
    },
  },
};

/**
 * (a) Stale coverage: a proposal with no pending topics, whose recorded
 * coverage evidence no longer matches the settings direction it describes —
 * the state `needsCoverageUpdate` exists to catch (see
 * `interest-profile-modal.ts`).
 */
export const STALE_COVERAGE_SETTINGS_TOPICS = [{
  id: "acceptance-existing-topic",
  name: "Existing Topic",
  tag: "existing-topic",
  detail: false,
  description: "An existing research topic the researcher already follows.",
  directions: [{
    id: "acceptance-existing-direction",
    text: "Hand-edited direction text, changed since the proposal was generated",
    origin: "manual",
  }],
}];

export const STALE_COVERAGE_PROPOSAL = proposalFixture({
  proposalId: "acceptance-proposal-stale",
  topics: [],
  coveredPaperKeys: ["paper-stale-1"],
  coverageEvidence: [{
    topicId: "acceptance-existing-topic",
    directionId: "acceptance-existing-direction",
    directionText: "Direction text recorded when this coverage was generated",
    paperKeys: ["paper-stale-1"],
  }],
});

/** (b) Read failure: no proposal, and a load error that is not a retired-schema one. */
export const READ_FAILURE_ERROR = {
  kind: "proposal",
  code: "load-failed",
  message: "Personal library direction proposal could not be loaded (load-failed).",
};

/** (c) Nothing new needed: no pending topics, and recorded coverage still holds. */
export const NOTHING_NEW_PROPOSAL = proposalFixture({
  proposalId: "acceptance-proposal-done",
  topics: [],
  coveredPaperKeys: ["paper-done-1", "paper-done-2"],
  coverageEvidence: [],
});

/**
 * (d)/(e) A normal review page: one topic with an unprocessed long direction
 * and an unprocessed short one, so the direction list, the long-text
 * rendering and the Library overview round trip all have something to show.
 */
export const NORMAL_PROPOSAL = proposalFixture({
  proposalId: "acceptance-proposal-normal",
  topics: [{
    id: REVIEW_TOPIC_ID,
    suggestedName: "Acceptance review topic",
    directions: [
      {
        id: LONG_DIRECTION_ID,
        text: LONG_DIRECTION,
        discoveryCues: ["desktop acceptance probe"],
        representatives: [
          { paperKey: "paper-long-1", evidenceFingerprint: "sha256:long-1" },
          { paperKey: "paper-long-2", evidenceFingerprint: "sha256:long-2" },
        ],
        representativeSetFingerprint: "sha256:long-set",
        lineage: { candidateIds: [LONG_DIRECTION_ID] },
      },
      {
        id: SHORT_DIRECTION_ID,
        text: SHORT_DIRECTION,
        discoveryCues: ["desktop acceptance probe"],
        representatives: [
          { paperKey: "paper-short-1", evidenceFingerprint: "sha256:short-1" },
          { paperKey: "paper-short-2", evidenceFingerprint: "sha256:short-2" },
        ],
        representativeSetFingerprint: "sha256:short-set",
        lineage: { candidateIds: [SHORT_DIRECTION_ID] },
      },
    ],
  }],
});

// ── renderer expressions: injecting state and opening the dialog ───────────

/**
 * Writes the review state directly onto the running plugin instance, the way
 * `topic-directions.mjs` seeds `settings.arxiv.topics`. There is no on-disk
 * proposal document in this harness: `libraryProposal` only reaches disk
 * through `mutatePersonalLibraryProposal`, which this never calls, so nothing
 * here is written to the vault beyond the settings save below (already
 * restored by the harness's vault-state capture, same as the topic fixture).
 */
function injectReviewStateExpression({
  proposal, proposalLoadError = null, settingsTopics = [], catalog = CATALOG_FIXTURE,
}) {
  return `(async () => {
    const plugin = ${PLUGIN};
    if (!plugin) return "ERROR: the arXiv Daily plugin is not loaded";
    plugin.libraryProposal = ${asCode(proposal)};
    plugin.libraryProposalLoadError = ${asCode(proposalLoadError)};
    plugin.libraryProposalAcceptances = undefined;
    plugin.libraryCatalog = ${asCode(catalog)};
    plugin.libraryCatalogLoadError = null;
    plugin.settings.arxiv.topics = ${asCode(settingsTopics)};
    await plugin.saveSettings();
    return "injected";
  })()`;
}

/**
 * Opens the dialog the way a researcher does: Settings → arXiv Daily →
 * "Review suggestions" on the "Topics from library" row. Reused across every
 * fixture rather than calling `openPersonalLibraryDirectionReview` directly,
 * so the button's own disabled-state guard is exercised too.
 */
export const OPEN_REVIEW_EXPRESSION = `(() => {
  app.setting.open();
  app.setting.openTabById(${asCode(PLUGIN_ID)});
  const rows = Array.from(document.querySelectorAll(".setting-item"));
  const row = rows.find((el) => (el.querySelector(".setting-item-name")?.textContent ?? "").trim() === "Topics from library");
  if (!row) return JSON.stringify({ error: "the settings page has no Topics from library row" });
  const button = Array.from(row.querySelectorAll("button")).find((b) => (b.textContent ?? "").trim() === "Review suggestions");
  if (!button) return JSON.stringify({ error: "the Topics from library row has no Review suggestions button" });
  if (button.disabled) return JSON.stringify({ error: "the Review suggestions button is disabled" });
  button.click();
  return JSON.stringify({ clicked: true });
})()`;

/**
 * Earlier sessions in this run may have left the embedding mode on remote,
 * authorized, against a manifest that was never actually built for it —
 * `refreshLibraryIndexTrace` in `plugin/main.ts` then reports
 * `LIBRARY_INDEX_READ_FAILED` on its own. That preparation error would make
 * "Review suggestions" refuse to open regardless of `lastRun`, for a reason
 * that has nothing to do with the states this scenario drives, so it is
 * cleared directly alongside the synthetic last run below.
 */
export const CLEAR_PREPARATION_ERROR_EXPRESSION = `(() => {
  ${PLUGIN}.libraryIndexStatus.setPreparationError(undefined);
  return JSON.stringify({ cleared: true });
})()`;

export const MODAL_PRESENT_EXPRESSION = `JSON.stringify({ present: Boolean(document.querySelector(${asCode(MODAL_SELECTOR)})) })`;

export const CLOSE_MODAL_EXPRESSION = `(() => {
  const modal = document.querySelector(${asCode(MODAL_SELECTOR)});
  if (!modal) return JSON.stringify({ alreadyClosed: true });
  const button = modal.querySelector(".modal-close-button");
  if (!button) return JSON.stringify({ error: "the review modal has no close button" });
  button.click();
  return JSON.stringify({ clicked: true });
})()`;

async function waitForModalClosed(evaluate, { attempts = 20, intervalMs = 250 } = {}) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const state = await readJson(evaluate, MODAL_PRESENT_EXPRESSION);
    if (!state.present) return true;
    await wait(evaluate, intervalMs);
  }
  return false;
}

const MODAL_PRELUDE = `
  const modalRoot = () => document.querySelector(${asCode(MODAL_SELECTOR)});
  const content = () => modalRoot()?.querySelector(".arxiv-daily-interest-review") ?? null;
`;

const inModal = (body) => `(() => {${MODAL_PRELUDE}${body}})()`;

/**
 * A snapshot of the whole default page: the heading, the one state the
 * branching logic chose, every primary and fallback button on it, and
 * whether the direction list and its accept bar are there at all — and in
 * what order.
 */
export const READ_PAGE_EXPRESSION = inModal(`
  const root = content();
  if (!root) return JSON.stringify({ present: false });
  const h2 = root.querySelector("h2");
  const stateTitle = root.querySelector(".arxiv-daily-interest-review__state h3");
  const stateMessage = root.querySelector(".arxiv-daily-interest-review__state > p");
  const toolbar = root.querySelector(".arxiv-daily-interest-review__toolbar");
  const options = root.querySelector(".arxiv-daily-interest-review__options");
  const backToReview = toolbar
    ? Array.from(toolbar.querySelectorAll("button")).find((b) => (b.textContent ?? "").trim() === "Back to review")
    : undefined;
  const optionsButtons = options
    ? Array.from(options.querySelectorAll(".arxiv-daily-interest-review__toolbar-actions button"))
      .map((b) => ({ text: (b.textContent ?? "").trim(), disabled: b.disabled === true }))
    : [];
  const stateButtons = Array.from(
    root.querySelectorAll(".arxiv-daily-interest-review__state button"),
  ).map((b) => ({
    text: (b.textContent ?? "").trim(), disabled: b.disabled === true, modCta: b.classList.contains("mod-cta"),
  }));
  const generateButtons = Array.from(root.querySelectorAll(".arxiv-daily-interest-review__generate")).map((b) => ({
    text: (b.textContent ?? "").trim(),
    disabled: b.disabled === true,
    title: b.getAttribute("title") ?? "",
    insideOptions: Boolean(b.closest(".arxiv-daily-interest-review__options")),
    insideState: Boolean(b.closest(".arxiv-daily-interest-review__state")),
  }));
  const topics = Array.from(root.querySelectorAll(".arxiv-daily-interest-review__topic"));
  const acceptBar = root.querySelector(".arxiv-daily-interest-review__accept-bar");
  let acceptBarAfterTopics = null;
  if (acceptBar && topics.length > 0) {
    const last = topics[topics.length - 1];
    acceptBarAfterTopics = Boolean(last.compareDocumentPosition(acceptBar) & Node.DOCUMENT_POSITION_FOLLOWING);
  }
  return JSON.stringify({
    present: true,
    heading: (h2?.textContent ?? "").trim(),
    stateTitle: (stateTitle?.textContent ?? "").trim(),
    stateMessage: (stateMessage?.textContent ?? "").trim(),
    hasBackToReview: Boolean(backToReview),
    optionsButtons,
    stateButtons,
    generateButtons,
    topicCount: topics.length,
    hasAcceptBar: Boolean(acceptBar),
    acceptBarAfterTopics,
  });
`);

export const OPEN_MORE_OPTIONS_EXPRESSION = inModal(`
  const options = content()?.querySelector(".arxiv-daily-interest-review__options");
  if (!options) return JSON.stringify({ error: "the review page has no More options control" });
  options.open = true;
  return JSON.stringify({ open: options.open });
`);

export function clickOptionButtonExpression(label) {
  return inModal(`
    const options = content()?.querySelector(".arxiv-daily-interest-review__options");
    if (!options) return JSON.stringify({ error: "the review page has no More options control" });
    const button = Array.from(options.querySelectorAll(".arxiv-daily-interest-review__toolbar-actions button"))
      .find((b) => (b.textContent ?? "").trim() === ${asCode(label)});
    if (!button) return JSON.stringify({ error: "More options has no " + ${asCode(label)} + " button" });
    button.click();
    return JSON.stringify({ clicked: ${asCode(label)} });
  `);
}

export const CLICK_BACK_TO_REVIEW_EXPRESSION = inModal(`
  const toolbar = content()?.querySelector(".arxiv-daily-interest-review__toolbar");
  const button = toolbar
    ? Array.from(toolbar.querySelectorAll("button")).find((b) => (b.textContent ?? "").trim() === "Back to review")
    : null;
  if (!button) return JSON.stringify({ error: "the overview page has no Back to review button" });
  button.click();
  return JSON.stringify({ clicked: true });
`);

/**
 * Direction cards live inside a topic `<details>` that starts collapsed; a
 * person has to open it before the direction text is on the page at all. The
 * topic is found by its own class rather than by an id, matching how a reader
 * finds it: there is exactly one topic in the normal-review fixture this is
 * used against.
 */
export const EXPAND_REVIEW_TOPIC_EXPRESSION = inModal(`
  const root = content();
  if (!root) return JSON.stringify({ error: "the review modal is not open" });
  const topic = root.querySelector(".arxiv-daily-interest-review__topic");
  if (!topic) return JSON.stringify({ error: "no direction topic is on the page to expand" });
  if (!topic.open) {
    const summary = topic.querySelector(":scope > summary.arxiv-daily-interest-review__topic-heading");
    if (!summary) return JSON.stringify({ error: "the topic has no heading to click open" });
    summary.click();
  }
  topic.scrollIntoView({ block: "center" });
  return JSON.stringify({ open: topic.open });
`);

/**
 * The long direction's own heading element, measured the way a person would
 * judge it: is it actually on the page with a real size, does it stay inside
 * the dialog rather than running off the side, and does it occupy more than
 * one line.
 *
 * `clientWidth`/`scrollWidth` on an inline `<strong>` are not useful here —
 * both read 0 on an inline element regardless of whether it is visible,
 * wrapped, or hidden inside a closed `<details>`, which would make an
 * overflow check pass vacuously. `getBoundingClientRect` and
 * `getClientRects()` describe the element's real, rendered line boxes
 * instead, and are compared against the dialog content's own box so a
 * genuine horizontal overflow is still caught.
 */
export function cardHeadingExpression(text) {
  return inModal(`
    const root = content();
    if (!root) return JSON.stringify({ error: "the review modal is not open" });
    const strongs = Array.from(root.querySelectorAll(".arxiv-daily-interest-review__card-heading strong"));
    const el = strongs.find((s) => (s.textContent ?? "").trim() === ${asCode(text)});
    if (!el) return JSON.stringify({ error: "no direction heading reads " + JSON.stringify(${asCode(text)}) });
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    const contentRect = root.getBoundingClientRect();
    return JSON.stringify({
      fontWeight: style.fontWeight,
      lineHeight: parseFloat(style.lineHeight || "0"),
      width: rect.width,
      height: rect.height,
      right: rect.right,
      lineBoxCount: el.getClientRects().length,
      contentRight: contentRect.right,
      contentScrollWidth: root.scrollWidth,
      contentClientWidth: root.clientWidth,
    });
  `);
}

// ── pure judgements, unit-testable without a renderer ───────────────────────

/** (a) Stale coverage: the single default page, with "Update suggestions" as its only call to action. */
export function judgeStaleCoveragePrimary(page) {
  if (!page.present) return { ok: false, reason: "the review modal did not open" };
  const problems = [];
  if (page.stateTitle !== "This review needs an update") {
    problems.push(`the page is titled "${page.stateTitle}", expected "This review needs an update"`);
  }
  const primary = page.stateButtons.find((button) => button.modCta);
  if (!primary) problems.push("no primary action is shown on the default page");
  else if (primary.text !== "Update suggestions") problems.push(`the primary action reads "${primary.text}", expected "Update suggestions"`);
  else if (primary.disabled) problems.push('"Update suggestions" is disabled');
  if (page.hasBackToReview) problems.push("a Back to review control is showing, so this is not the single default page");
  if (page.hasAcceptBar) problems.push("an Add to research topics bar is shown even though there is no direction to review");
  return problems.length === 0
    ? { ok: true, reason: 'primary action "Update suggestions" on the single default page, no accept bar' }
    : { ok: false, reason: problems.join("; ") };
}

/** (b) Read failure: "Try again" is the primary action; generation stays visible but disabled, with a reason. */
export function judgeReadFailurePrimary(page) {
  if (!page.present) return { ok: false, reason: "the review modal did not open" };
  const problems = [];
  if (page.stateTitle !== "Couldn't open your suggestions") {
    problems.push(`the page is titled "${page.stateTitle}", expected "Couldn't open your suggestions"`);
  }
  const primary = page.stateButtons.find((button) => button.modCta);
  if (!primary) problems.push("no primary action is shown");
  else if (primary.text !== "Try again") problems.push(`the primary action reads "${primary.text}", expected "Try again"`);
  else if (primary.disabled) problems.push('"Try again" is disabled');
  const fallback = page.generateButtons.find((button) => button.insideOptions);
  if (!fallback) problems.push("no generation fallback is offered in More options");
  else {
    if (!fallback.disabled) problems.push(`the generation fallback "${fallback.text}" is not disabled`);
    if (!fallback.title) problems.push("the disabled generation fallback carries no explanatory reason");
  }
  return problems.length === 0
    ? { ok: true, reason: `primary "Try again"; fallback "${fallback.text}" disabled, titled "${fallback.title}"` }
    : { ok: false, reason: problems.join("; ") };
}

/** (c) Nothing new needed: "Done" is the only action. */
export function judgeNothingNewDone(page) {
  if (!page.present) return { ok: false, reason: "the review modal did not open" };
  const problems = [];
  if (page.stateTitle !== "No new directions to add") {
    problems.push(`the page is titled "${page.stateTitle}", expected "No new directions to add"`);
  }
  const primary = page.stateButtons.find((button) => button.modCta);
  if (!primary) problems.push("no primary action is shown");
  else if (primary.text !== "Done") problems.push(`the primary action reads "${primary.text}", expected "Done"`);
  return problems.length === 0
    ? { ok: true, reason: 'primary action "Done"' }
    : { ok: false, reason: problems.join("; ") };
}

/** The Add to research topics bar sits after the direction list, not above or beside it. */
export function judgeAcceptBarBelowList(page) {
  if (!page.present) return { ok: false, reason: "the review modal did not open" };
  const problems = [];
  if (page.topicCount === 0) problems.push("no direction topics are on the page to judge the bar's position against");
  if (!page.hasAcceptBar) problems.push("no Add to research topics bar is shown");
  if (page.acceptBarAfterTopics !== true) problems.push("the Add to research topics bar is not positioned after the direction list in the document");
  return problems.length === 0
    ? { ok: true, reason: `the accept bar follows ${page.topicCount} topic(s) in document order` }
    : { ok: false, reason: problems.join("; ") };
}

/** The review's default page is the only page: no persistent tab control sits on it. */
export function judgeNoTabBar(page) {
  if (!page.present) return { ok: false, reason: "the review modal did not open" };
  return page.hasBackToReview
    ? { ok: false, reason: "a Back to review control is visible on the default review page" }
    : { ok: true, reason: "the default page shows no tab-switching control" };
}

/** More options → Library overview → Back to review has to actually round-trip. */
export function judgeLibraryOverviewRoundTrip(before, afterOverview, afterBack) {
  const problems = [];
  if (afterOverview.heading !== "Library overview") {
    problems.push(`after Library overview the heading reads "${afterOverview.heading}", expected "Library overview"`);
  }
  if (!afterOverview.hasBackToReview) problems.push("Library overview shows no Back to review control");
  if (afterBack.heading !== before.heading) {
    problems.push(`after Back to review the heading reads "${afterBack.heading}", expected "${before.heading}"`);
  }
  if (afterBack.hasBackToReview) problems.push("Back to review left its own Back to review control showing");
  return problems.length === 0
    ? { ok: true, reason: `Library overview → Back to review returns to "${before.heading}"` }
    : { ok: false, reason: problems.join("; ") };
}

/** The long direction drops the bold run-on styling (`font-weight: var(--font-normal)` in styles.css). */
export function judgeDirectionNotBold(heading) {
  if (heading.error) return { ok: false, reason: heading.error };
  const weight = Number.parseInt(heading.fontWeight, 10);
  if (Number.isFinite(weight) && weight >= 600) {
    return { ok: false, reason: `the long direction renders at font-weight ${heading.fontWeight}, which reads as bold` };
  }
  if (!Number.isFinite(weight) && /bold/i.test(heading.fontWeight)) {
    return { ok: false, reason: `the long direction renders with font-weight "${heading.fontWeight}", which reads as bold` };
  }
  return { ok: true, reason: `font-weight ${heading.fontWeight}, not bold` };
}

const round = (value) => Math.round(value * 10) / 10;

/**
 * The long direction is actually on the page with a real, rendered size.
 *
 * A collapsed topic, or an inline element measured by `clientWidth` instead
 * of its rendered box, both read as zero-size — this is what turns "it
 * wraps" into a check that cannot fail no matter what the page looked like.
 */
export function judgeDirectionVisible(heading) {
  if (heading.error) return { ok: false, reason: heading.error };
  if (!(heading.width > 0) || !(heading.height > 0)) {
    return {
      ok: false,
      reason: `the long direction heading measured ${round(heading.width)}x${round(heading.height)} — it is hidden, collapsed, or not laid out`,
    };
  }
  return { ok: true, reason: `visible at ${round(heading.width)}x${round(heading.height)}` };
}

/**
 * The long direction stays inside the dialog and actually wraps onto more
 * than one line, rather than running off the side or being measured as a
 * single invisible or single-line box.
 *
 * Three independent facts, all required: a real size (`judgeDirectionVisible`),
 * no horizontal overflow — of the heading past the dialog's own content edge,
 * and of the dialog's content past its own scroll box — and more than one
 * rendered line box (or, failing that, a height at least 1.5× the line
 * height, in case a theme's line-box reporting ever collapses adjacent
 * boxes). A single line box with height close to one line height fails this
 * even if every other check above would have passed.
 */
export function judgeDirectionWraps(heading, label) {
  const visible = judgeDirectionVisible(heading);
  if (!visible.ok) return { ok: false, reason: `${label}: ${visible.reason}` };
  if (heading.right > heading.contentRight + 1) {
    return {
      ok: false,
      reason: `${label}: the long direction's right edge (${round(heading.right)}) runs ${round(heading.right - heading.contentRight)}px past the dialog's content edge (${round(heading.contentRight)})`,
    };
  }
  if (heading.contentScrollWidth > heading.contentClientWidth + 1) {
    return {
      ok: false,
      reason: `${label}: the dialog's own content overflows horizontally (scrollWidth ${heading.contentScrollWidth} > clientWidth ${heading.contentClientWidth})`,
    };
  }
  const wrapsByLineBoxes = heading.lineBoxCount > 1;
  const wrapsByHeight = heading.lineHeight > 0 && heading.height >= 1.5 * heading.lineHeight;
  if (!wrapsByLineBoxes && !wrapsByHeight) {
    return {
      ok: false,
      reason: `${label}: the long direction rendered as a single line (${heading.lineBoxCount} line box, height ${round(heading.height)} vs. line-height ${round(heading.lineHeight)}) instead of wrapping`,
    };
  }
  return {
    ok: true,
    reason: `${label}: ${heading.lineBoxCount} line boxes, ${round(heading.width)}x${round(heading.height)}, right edge ${round(heading.right)} within the dialog's content edge ${round(heading.contentRight)}`,
  };
}

// ── the scenario ────────────────────────────────────────────────────────────

/**
 * Walks the redesigned review dialog through the states the branch changed,
 * one real `Review suggestions` open per fixture, asserting each one and
 * leaving a screenshot behind.
 */
export async function directionReviewScenarios({
  session,
  screenshots,
  narrowViewport = NARROW_VIEWPORT,
  wideViewport = WIDE_VIEWPORT,
  settleMs = 600,
}) {
  const { evaluate, client, diagnostics } = session;
  const errorsBefore = diagnostics.errors().length;
  const results = [];
  const shot = async (name) => {
    if (!screenshots) return;
    await screenshots.capture(name, { selector: MODAL_SELECTOR });
  };

  // Earlier scenarios in this session may have left preparation activity or
  // an error behind; a fresh, successful last run with no preparation error
  // is what makes "Review suggestions" clickable, matching how a prepared
  // library actually offers it.
  await evaluate(CLEAR_PREPARATION_ERROR_EXPRESSION);
  await evaluate(setLastIndexRunExpression({ updatedAt: "2026-09-30T12:00:00.000Z", papers: 42 }));
  await setViewport(client, wideViewport);

  const openState = async (fixture) => {
    const injected = await evaluate(injectReviewStateExpression(fixture));
    if (typeof injected === "string" && injected.startsWith("ERROR:")) {
      return { error: injected.slice(7).trim() };
    }
    const opened = await readJson(evaluate, OPEN_REVIEW_EXPRESSION);
    if (opened.error) return opened;
    await wait(evaluate, settleMs * 2);
    return opened;
  };
  const closeModal = async () => {
    await evaluate(CLOSE_MODAL_EXPRESSION);
    await waitForModalClosed(evaluate);
    await wait(evaluate, settleMs);
  };

  // (a) — stale coverage: single default page, "Update suggestions" primary.
  {
    const opened = await openState({ proposal: STALE_COVERAGE_PROPOSAL, settingsTopics: STALE_COVERAGE_SETTINGS_TOPICS });
    if (opened.error) {
      results.push(fail("direction-review-stale-coverage-primary", opened.error));
    } else {
      const page = await readJson(evaluate, READ_PAGE_EXPRESSION);
      const verdict = judgeStaleCoveragePrimary(page);
      results.push((verdict.ok ? pass : fail)("direction-review-stale-coverage-primary", verdict.reason));
      await shot("direction-review-stale-coverage");
    }
    await closeModal();
  }

  // (b) — read failure: "Try again" primary; generation fallback visible, disabled, explained.
  {
    const opened = await openState({ proposal: null, proposalLoadError: READ_FAILURE_ERROR, settingsTopics: [] });
    if (opened.error) {
      results.push(fail("direction-review-read-failure-primary", opened.error));
    } else {
      await evaluate(OPEN_MORE_OPTIONS_EXPRESSION);
      await wait(evaluate, settleMs);
      const page = await readJson(evaluate, READ_PAGE_EXPRESSION);
      const verdict = judgeReadFailurePrimary(page);
      results.push((verdict.ok ? pass : fail)("direction-review-read-failure-primary", verdict.reason));
      await shot("direction-review-read-failure");
    }
    await closeModal();
  }

  // (c) — nothing new needed: "Done" is the only action.
  {
    const opened = await openState({ proposal: NOTHING_NEW_PROPOSAL, settingsTopics: [] });
    if (opened.error) {
      results.push(fail("direction-review-nothing-new-done", opened.error));
    } else {
      const page = await readJson(evaluate, READ_PAGE_EXPRESSION);
      const verdict = judgeNothingNewDone(page);
      results.push((verdict.ok ? pass : fail)("direction-review-nothing-new-done", verdict.reason));
      await shot("direction-review-nothing-new");
    }
    await closeModal();
  }

  // (d)+(e) — a normal review page: direction list structure, the Library
  // overview round trip, and the long direction's styling and wrapping.
  {
    const opened = await openState({ proposal: NORMAL_PROPOSAL, settingsTopics: [] });
    if (opened.error) {
      for (const name of [
        "direction-review-single-page-no-tabs",
        "direction-review-accept-bar-below-list",
        "direction-review-overview-round-trip",
        "direction-review-direction-not-bold",
        "direction-review-direction-wraps-wide",
        "direction-review-direction-wraps-narrow",
      ]) results.push(fail(name, opened.error));
    } else {
      const page = await readJson(evaluate, READ_PAGE_EXPRESSION);
      results.push((judgeNoTabBar(page).ok ? pass : fail)(
        "direction-review-single-page-no-tabs", judgeNoTabBar(page).reason,
      ));
      results.push((judgeAcceptBarBelowList(page).ok ? pass : fail)(
        "direction-review-accept-bar-below-list", judgeAcceptBarBelowList(page).reason,
      ));
      await shot("direction-review-candidate-list");

      // (d) More options → Library overview → Back to review.
      await evaluate(OPEN_MORE_OPTIONS_EXPRESSION);
      await wait(evaluate, settleMs);
      const overviewClick = await readJson(evaluate, clickOptionButtonExpression("Library overview"));
      if (overviewClick.error) {
        results.push(fail("direction-review-overview-round-trip", overviewClick.error));
      } else {
        await wait(evaluate, settleMs);
        const overviewPage = await readJson(evaluate, READ_PAGE_EXPRESSION);
        await shot("direction-review-library-overview");
        const backClick = await readJson(evaluate, CLICK_BACK_TO_REVIEW_EXPRESSION);
        if (backClick.error) {
          results.push(fail("direction-review-overview-round-trip", backClick.error));
        } else {
          await wait(evaluate, settleMs);
          const backPage = await readJson(evaluate, READ_PAGE_EXPRESSION);
          const verdict = judgeLibraryOverviewRoundTrip(page, overviewPage, backPage);
          results.push((verdict.ok ? pass : fail)("direction-review-overview-round-trip", verdict.reason));
        }
      }

      // (e) — long direction text: not bold, wraps at wide and narrow widths.
      // The direction lives inside a topic that starts collapsed (same as
      // any real review page); it has to actually be opened, the way a
      // reader opens it, before the heading is on the page to measure at all.
      const expanded = await readJson(evaluate, EXPAND_REVIEW_TOPIC_EXPRESSION);
      if (expanded.error || !expanded.open) {
        const why = expanded.error ?? "the topic reported closed after being clicked open";
        for (const name of ["direction-review-direction-not-bold", "direction-review-direction-wraps-wide", "direction-review-direction-wraps-narrow"]) {
          results.push(fail(name, why));
        }
      } else {
        await wait(evaluate, settleMs);
        const wideHeading = await readJson(evaluate, cardHeadingExpression(LONG_DIRECTION));
        results.push((judgeDirectionNotBold(wideHeading).ok ? pass : fail)(
          "direction-review-direction-not-bold", judgeDirectionNotBold(wideHeading).reason,
        ));
        results.push((judgeDirectionWraps(wideHeading, "wide").ok ? pass : fail)(
          "direction-review-direction-wraps-wide", judgeDirectionWraps(wideHeading, "wide").reason,
        ));
        await shot("direction-review-long-direction-wide");

        await setViewport(client, narrowViewport);
        await wait(evaluate, settleMs);
        const narrowHeading = await readJson(evaluate, cardHeadingExpression(LONG_DIRECTION));
        results.push((judgeDirectionWraps(narrowHeading, "narrow").ok ? pass : fail)(
          "direction-review-direction-wraps-narrow", judgeDirectionWraps(narrowHeading, "narrow").reason,
        ));
        await shot("direction-review-long-direction-narrow");
        await setViewport(client, wideViewport);
        await wait(evaluate, settleMs);
      }
    }
    await closeModal();
  }

  await clearViewport(client);
  await evaluate(CLEAR_LAST_INDEX_RUN_EXPRESSION);

  const newErrors = diagnostics.errors().slice(errorsBefore);
  results.push(
    newErrors.length === 0
      ? pass("direction-review-console-clean", "no renderer errors or pageerrors while reviewing directions")
      : fail("direction-review-console-clean", `${newErrors.length} renderer error(s): ${newErrors[0]?.text ?? ""}`),
  );

  return results;
}
