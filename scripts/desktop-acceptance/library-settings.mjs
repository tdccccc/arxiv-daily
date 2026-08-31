/**
 * The personal library settings page, checked in the real renderer.
 *
 * These assertions exist because the unit suite renders the same rows into
 * happy-dom, which has no layout engine and no Obsidian stylesheet: it can say
 * which buttons a row asks for, but not whether they end up on one line, inside
 * their container, or on top of the description. It also cannot show a person
 * what the page looks like, which is why this scenario writes screenshots.
 */
import { clearViewport, setViewport } from "./cdp.mjs";

const PLUGIN_ID = "arxiv-daily";
const PLUGIN = `app.plugins.plugins[${JSON.stringify(PLUGIN_ID)}]`;

/**
 * The disclosure dialog is located by a class the plugin puts on its root, not
 * by its heading.
 *
 * The heading is product copy, and it follows the processing depth — so pinning
 * the lookup to it meant every wording change broke the scenario at "no dialog
 * opened", which reads like the consent step stopped working. Located by a
 * stable class, a wording change fails on the title assertion below instead,
 * which names what actually changed. The class is guarded on the plugin side by
 * `plugin/tests/library-modal.test.ts`.
 */
export const DISCLOSURE_MODAL_CLASS = "arxiv-daily-library-authorization-modal";

/**
 * The dialog's two answers, located the same way and for the same reason.
 *
 * Clicking them by label was the same mistake one level down: the confirm
 * label follows the processing depth, so rewording it broke the scenario at
 * "the modal has no Authorize button", which reads like the dialog lost its
 * button. Marked buttons make a reworded label fail on `judgeDisclosureButtons`
 * instead. Guarded on the plugin side by `plugin/tests/library-modal.test.ts`.
 */
export const DISCLOSURE_CONFIRM_BUTTON_CLASS = "arxiv-daily-library-authorization-confirm";
export const DISCLOSURE_CANCEL_BUTTON_CLASS = "arxiv-daily-library-authorization-cancel";

/**
 * What the heading has to read at each processing depth. Two entries because
 * the depth is not fixed: a grant that covers full text says so, and one that
 * covers metadata and abstracts must not claim otherwise.
 *
 * Only `full-text` is reachable from this scenario — the settings page refuses
 * to disclose at all unless there is an embedding endpoint to name, so every
 * dialog it can open is a full-text one. The metadata heading is asserted in
 * `plugin/tests/library-modal.test.ts`, which can call the dialog directly.
 */
export const DISCLOSURE_TITLES = {
  "full-text": "Send full text off this device?",
  "metadata-and-abstracts": "Send titles and abstracts off this device?",
};

/**
 * What the confirm button has to read at each depth — the heading's question
 * answered in the heading's own words, rather than "Authorize", which answers a
 * question the dialog never asked.
 *
 * Keyed by depth alongside `DISCLOSURE_TITLES` because both come from one
 * branch on the depth in the plugin (`libraryAuthorizationCopy`); a third depth
 * has to gain an entry in both, and a missing entry fails loudly below.
 * `metadata-and-abstracts` is unreachable from this scenario for the reason
 * given above, and is asserted in `plugin/tests/library-modal.test.ts`.
 */
export const DISCLOSURE_CONFIRM_LABELS = {
  "full-text": "Send full text",
  "metadata-and-abstracts": "Send titles and abstracts",
};

/** The negative answer does not follow the depth; there is only one way out. */
export const DISCLOSURE_CANCEL_LABEL = "Cancel";

/** The three groups whose order the settings page is supposed to present. */
export const EXPECTED_GROUP_ORDER = ["Output & schedule", "Personal library", "Email delivery"];

/**
 * Viewport widths, not panel widths: Obsidian derives the settings panel from
 * the window, and driving the window is the only way to reach a layout a user
 * can actually reach. The resulting panel width is measured and reported.
 */
export const NARROW_VIEWPORT = { width: 700, height: 900 };
export const WIDE_VIEWPORT = { width: 1400, height: 900 };
/** Tall enough that a whole settings section fits in one capture. */
export const CAPTURE_VIEWPORT = { width: 1100, height: 1400 };
/**
 * Below this window width Obsidian stacks each setting row vertically and
 * stretches its controls to the full row width. That layout gets its own
 * assertion: one line and right alignment are meaningless once every button is
 * full width, but "stays inside the row" still has to hold.
 */
export const STACKED_VIEWPORT = { width: 560, height: 900 };

/** Sub-pixel noise from fractional layout; anything larger is a real offset. */
const ALIGNMENT_TOLERANCE_PX = 1.5;

/**
 * How narrow the description column is allowed to get before the row stops
 * being a sentence next to some buttons and becomes a column of letters.
 *
 * Two floors, because either one alone can be satisfied by a broken layout.
 * A pixel floor alone says nothing about whether the text reads — a wide box
 * holding one word per line would pass it. A characters-per-line floor alone
 * can be met by a sliver of box holding a short description on two lines. The
 * pair is what the failing screenshot violates on both counts: a 6px column
 * spelling the description out one letter at a time.
 */
export const MIN_DESCRIPTION_WIDTH_PX = 150;
export const MIN_DESCRIPTION_CHARACTERS_PER_LINE = 12;

const RENDERER_PRELUDE = `
  const content = document.querySelector(".vertical-tab-content.arxiv-daily-settings")
    ?? document.querySelector(".vertical-tab-content.is-active")
    ?? document.querySelector(".vertical-tab-content");
  const allRows = () => Array.from(content ? content.querySelectorAll(".setting-item") : []);
  const rowName = (el) => (el.querySelector(".setting-item-name")?.textContent ?? "").trim();
  const isHeading = (el) => el.classList.contains("setting-item-heading");
  const buttonTexts = (el) => Array.from(el.querySelectorAll(".setting-item-control button"))
    .map((b) => (b.textContent ?? "").trim());
  const namedRow = (name) => allRows().find((el) => !isHeading(el) && rowName(el) === name);
  const groupRows = (heading) => {
    const rows = allRows();
    const start = rows.findIndex((el) => isHeading(el) && rowName(el) === heading);
    if (start < 0) return null;
    let end = rows.length;
    for (let i = start + 1; i < rows.length; i += 1) if (isHeading(rows[i])) { end = i; break; }
    return rows.slice(start, end);
  };
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height };
  };
  /*
   * How much of the description a reader actually gets. \`lines\` is counted from
   * the range's own client rects — one per line box the text was laid out into —
   * rather than from height divided by an assumed line height, so it stays true
   * whatever the theme's typography is.
   */
  const describe = (el) => {
    if (!el) return null;
    const text = (el.textContent ?? "").replace(/\\s+/g, " ").trim();
    const style = window.getComputedStyle(el);
    const contentWidth = el.clientWidth
      - parseFloat(style.paddingLeft || "0")
      - parseFloat(style.paddingRight || "0");
    const range = document.createRange();
    range.selectNodeContents(el);
    const rects = Array.from(range.getClientRects()).filter((r) => r.width > 0 && r.height > 0);
    const lines = new Set(rects.map((r) => Math.round(r.top))).size;
    return {
      ...box(el),
      contentWidth,
      characters: text.length,
      lines,
      widestLine: rects.length > 0 ? Math.max(...rects.map((r) => r.width)) : 0,
      text,
    };
  };
`;

const inRenderer = (body) => `(() => {${RENDERER_PRELUDE}${body}})()`;

async function readJson(evaluate, expression) {
  const raw = await evaluate(expression);
  if (typeof raw !== "string") {
    throw new Error(`expected a JSON string from the renderer, received ${JSON.stringify(raw)}`);
  }
  return JSON.parse(raw);
}

const wait = (evaluate, ms) => evaluate(`new Promise((resolve) => setTimeout(resolve, ${ms}))`);

function pass(name, detail) {
  return { name, passed: true, detail };
}

function fail(name, detail) {
  return { name, passed: false, detail };
}

const round = (value) => Math.round(value * 10) / 10;

// ── renderer queries ────────────────────────────────────────────────────────

export const OPEN_SETTINGS_EXPRESSION = `(() => {
  app.setting.open();
  app.setting.openTabById(${JSON.stringify(PLUGIN_ID)});
  return "opened";
})()`;

export const HEADINGS_EXPRESSION = inRenderer(`
  if (!content) return JSON.stringify({ error: "the arXiv Daily settings tab is not mounted" });
  return JSON.stringify({ headings: allRows().filter(isHeading).map(rowName) });
`);

export const LIBRARY_ROW_EXPRESSION = inRenderer(`
  if (!content) return JSON.stringify({ error: "the arXiv Daily settings tab is not mounted" });
  const group = groupRows("Personal library");
  if (!group) return JSON.stringify({ error: "the settings page has no Personal library section" });
  const row = namedRow("Library");
  if (!row) return JSON.stringify({ error: "the Personal library section has no Library row" });
  return JSON.stringify({
    rowButtons: buttonTexts(row),
    /*
     * The same buttons with the two facts a label cannot carry: whether the
     * button can be pressed, and the mark a previous read left on it. The mark
     * is how an in-place rewrite is told apart from a re-render that happened
     * to produce the same text — see \`judgeInPlaceProgressUpdate\`.
     */
    rowButtonStates: Array.from(row.querySelectorAll(".setting-item-control button"))
      .map((b) => ({
        text: (b.textContent ?? "").trim(),
        disabled: b.disabled === true,
        mark: b.dataset ? (b.dataset.acceptanceMark ?? null) : null,
      })),
    rowDescription: (row.querySelector(".setting-item-description")?.textContent ?? "").trim(),
    groupRowNames: group.map(rowName),
    groupButtons: group.flatMap(buttonTexts),
  });
`);

/**
 * Stamp every button in the row so the next read can tell whether these are the
 * same elements. Marks live on the element, so a re-render loses them.
 */
export const MARK_LIBRARY_BUTTONS_EXPRESSION = inRenderer(`
  const row = namedRow("Library");
  if (!row) return JSON.stringify({ error: "the Personal library section has no Library row" });
  const marked = Array.from(row.querySelectorAll(".setting-item-control button"))
    .map((b, index) => { b.dataset.acceptanceMark = "mark-" + index; return b.dataset.acceptanceMark; });
  return JSON.stringify({ marked });
`);

export const LIBRARY_GEOMETRY_EXPRESSION = inRenderer(`
  if (!content) return JSON.stringify({ error: "the arXiv Daily settings tab is not mounted" });
  const row = namedRow("Library");
  if (!row) return JSON.stringify({ error: "the Personal library section has no Library row" });
  const control = row.querySelector(".setting-item-control");
  const info = row.querySelector(".setting-item-info");
  const buttons = Array.from(control ? control.querySelectorAll("button") : []);
  if (!control || buttons.length === 0) {
    return JSON.stringify({ error: "the Library row rendered no buttons to measure" });
  }
  return JSON.stringify({
    panelWidth: box(content).width,
    windowWidth: window.innerWidth,
    row: box(row),
    info: info ? box(info) : null,
    description: describe(row.querySelector(".setting-item-description")),
    control: { ...box(control), scrollWidth: control.scrollWidth, clientWidth: control.clientWidth },
    buttons: buttons.map((b) => ({ text: (b.textContent ?? "").trim(), ...box(b) })),
  });
`);

/**
 * Scrolling and measuring are separate steps on purpose. The settings panel is
 * its own scroll container with smooth scrolling, so a rectangle read in the
 * same turn as the scroll describes where the row was, not where it lands —
 * and a screenshot clipped to it shows the neighbouring row.
 */
export const SCROLL_TO_SECTION_EXPRESSION = inRenderer(`
  const group = groupRows("Personal library");
  if (!group || group.length === 0) return JSON.stringify({ found: false });
  group[0].scrollIntoView({ block: "start", behavior: "instant" });
  return JSON.stringify({ found: true });
`);

export const SCROLL_TO_LIBRARY_ROW_EXPRESSION = inRenderer(`
  const row = namedRow("Library");
  if (!row) return JSON.stringify({ found: false });
  row.scrollIntoView({ block: "center", behavior: "instant" });
  return JSON.stringify({ found: true });
`);

export const LIBRARY_SECTION_RECT_EXPRESSION = inRenderer(`
  const group = groupRows("Personal library");
  if (!group || group.length === 0) return null;
  const first = box(group[0]);
  const last = box(group[group.length - 1]);
  return JSON.stringify({
    x: Math.min(first.left, last.left) + window.scrollX,
    y: Math.min(first.top, last.top) + window.scrollY,
    width: Math.max(first.right, last.right) - Math.min(first.left, last.left),
    height: Math.max(first.bottom, last.bottom) - Math.min(first.top, last.top),
  });
`);

export const LIBRARY_ROW_RECT_EXPRESSION = inRenderer(`
  const row = namedRow("Library");
  if (!row) return null;
  const r = box(row);
  return JSON.stringify({ x: r.left + window.scrollX, y: r.top + window.scrollY, width: r.width, height: r.height });
`);

/**
 * Where an indexing probe keeps the operation it started, so a later step can
 * finish it. A renderer global, because these are separate `evaluate` calls in
 * one page.
 */
const INDEX_PROBE_GLOBAL = "__arxivDailyAcceptanceIndexRun";

/**
 * Start a real, cancellable operation and tell the settings row about it.
 *
 * The operation is genuine — it goes through the plugin's own registry, so the
 * Cancel button on the row aborts a signal something could be listening to.
 * What is not genuine is the work: an actual index run needs a local embedding
 * model and minutes of PDF extraction, none of which says anything about how the
 * row looks while it happens. The wiring from a real run to this same store is
 * asserted in `plugin/tests/fulltext-index-lifecycle.test.ts`; what only the
 * renderer can answer is what the row does with it.
 */
export function beginIndexRunExpression({ phase, completed, total }) {
  return `(() => {
    const plugin = ${PLUGIN};
    const operation = plugin.operations.begin(
      "personal-library-fulltext-index",
      "Personal library full-text index",
      "acceptance-index-probe",
    );
    window[${JSON.stringify(INDEX_PROBE_GLOBAL)}] = operation;
    plugin.libraryIndexStatus.beginRun(operation.id, ${JSON.stringify(phase)});
    plugin.libraryIndexStatus.report(${JSON.stringify({ phase, completed, total })});
    return JSON.stringify({ operationId: operation.id });
  })()`;
}

export function reportIndexProgressExpression({ phase, completed, total }) {
  return `(() => {
    ${PLUGIN}.libraryIndexStatus.report(${JSON.stringify({ phase, completed, total })});
    return JSON.stringify({ reported: ${completed} });
  })()`;
}

export const END_INDEX_RUN_EXPRESSION = `(() => {
  const plugin = ${PLUGIN};
  const operation = window[${JSON.stringify(INDEX_PROBE_GLOBAL)}];
  if (operation) operation.finish();
  delete window[${JSON.stringify(INDEX_PROBE_GLOBAL)}];
  plugin.libraryIndexStatus.endRun();
  return JSON.stringify({ ended: true });
})()`;

export function setLastIndexRunExpression({ updatedAt, papers }) {
  return `(() => {
    ${PLUGIN}.libraryIndexStatus.setLastRun(${JSON.stringify({ updatedAt, papers })});
    return JSON.stringify({ recorded: ${papers} });
  })()`;
}

/**
 * Put the trace back to "nothing has been indexed here". The probe's timestamp
 * is invented, and leaving it on the row would put a claim in every screenshot
 * taken after it that no run in this vault ever earned.
 */
export const CLEAR_LAST_INDEX_RUN_EXPRESSION = `(() => {
  ${PLUGIN}.libraryIndexStatus.setLastRun(undefined);
  return JSON.stringify({ cleared: true });
})()`;

export const INDEX_OPERATIONS_EXPRESSION = `JSON.stringify({
  operations: ${PLUGIN}.operations.snapshot()
    .filter((operation) => operation.kind === "personal-library-fulltext-index")
    .map((operation) => ({ id: operation.id, cancellationRequested: operation.cancellationRequested })),
})`;

export const PLUGIN_STATE_EXPRESSION = `JSON.stringify({
  embedding: ${PLUGIN}.settings.embedding,
  status: ${PLUGIN}.getLibraryConnectionStatus(),
  operations: ${PLUGIN}.operations.snapshot().map((operation) => operation.kind),
  notices: Array.from(document.querySelectorAll(".notice")).map((n) => (n.textContent ?? "").trim()),
})`;

export const EMBEDDING_SELECT_VALUE_EXPRESSION = inRenderer(`
  const row = namedRow("Embedding");
  const select = row?.querySelector("select");
  return JSON.stringify({ value: select ? select.value : null });
`);

export function selectEmbeddingModeExpression(mode) {
  return inRenderer(`
    const row = namedRow("Embedding");
    if (!row) return JSON.stringify({ error: "the Personal library section has no Embedding row" });
    const select = row.querySelector("select");
    if (!select) return JSON.stringify({ error: "the Embedding row rendered no dropdown" });
    select.value = ${JSON.stringify(mode)};
    select.dispatchEvent(new Event("change"));
    return JSON.stringify({ dispatched: select.value });
  `);
}

export function clickLibraryRowButtonExpression(label) {
  return inRenderer(`
    const row = namedRow("Library");
    if (!row) return JSON.stringify({ error: "the Personal library section has no Library row" });
    const button = Array.from(row.querySelectorAll(".setting-item-control button"))
      .find((b) => (b.textContent ?? "").trim() === ${JSON.stringify(label)});
    if (!button) return JSON.stringify({ error: "the Library row has no " + ${JSON.stringify(label)} + " button" });
    button.click();
    return JSON.stringify({ clicked: ${JSON.stringify(label)} });
  `);
}

/**
 * Reads the dialog located by `modalClass`, and reports its heading and the
 * labels of its two marked answers as data rather than matching on any of them
 * — so the wording can be asserted separately from the lookups that found the
 * dialog and its buttons.
 */
export function modalExpression(modalClass = DISCLOSURE_MODAL_CLASS) {
  return `(() => {
    const modal = document.querySelector(
      ".modal-container .modal." + ${JSON.stringify(modalClass)},
    );
    if (!modal) return JSON.stringify({ present: false });
    const r = modal.getBoundingClientRect();
    const marked = (cls) => {
      const button = modal.querySelector("button." + cls);
      return button ? (button.textContent ?? "").trim() : null;
    };
    return JSON.stringify({
      present: true,
      title: (modal.querySelector(".modal-title")?.textContent ?? "").trim(),
      buttons: Array.from(modal.querySelectorAll("button")).map((b) => (b.textContent ?? "").trim()),
      confirm: marked(${JSON.stringify(DISCLOSURE_CONFIRM_BUTTON_CLASS)}),
      cancel: marked(${JSON.stringify(DISCLOSURE_CANCEL_BUTTON_CLASS)}),
      text: (modal.querySelector(".modal-content")?.textContent ?? "").replace(/\\s+/g, " ").trim(),
      rect: { x: r.left + window.scrollX, y: r.top + window.scrollY, width: r.width, height: r.height },
    });
  })()`;
}

/**
 * Clicks the answer marked with `buttonClass`. The argument is a mark, never a
 * label: the confirm label follows the processing depth, so selecting on it
 * would make every copy change read as a missing button.
 */
export function clickModalButtonExpression(buttonClass, modalClass = DISCLOSURE_MODAL_CLASS) {
  return `(() => {
    const modal = document.querySelector(
      ".modal-container .modal." + ${JSON.stringify(modalClass)},
    );
    if (!modal) return JSON.stringify({ error: "no ." + ${JSON.stringify(modalClass)} + " modal is open" });
    const button = modal.querySelector("button." + ${JSON.stringify(buttonClass)});
    if (!button) return JSON.stringify({ error: "the modal has no ." + ${JSON.stringify(buttonClass)} + " button" });
    button.click();
    return JSON.stringify({ clicked: ${JSON.stringify(buttonClass)}, label: (button.textContent ?? "").trim() });
  })()`;
}

async function waitForModal(evaluate, { attempts = 40, intervalMs = 250 } = {}) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const state = await readJson(evaluate, modalExpression());
    if (state.present) return state;
    await wait(evaluate, intervalMs);
  }
  return { present: false };
}

async function waitForModalGone(evaluate, { attempts = 20, intervalMs = 250 } = {}) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const state = await readJson(evaluate, modalExpression());
    if (!state.present) return true;
    await wait(evaluate, intervalMs);
  }
  return false;
}

// ── pure judgements, unit-testable without a renderer ───────────────────────

/** The three groups have to be adjacent and in order, not merely all present. */
export function judgeGroupOrder(headings, expected = EXPECTED_GROUP_ORDER) {
  const missing = expected.filter((heading) => !headings.includes(heading));
  if (missing.length > 0) {
    return { ok: false, reason: `the settings page has no ${missing.join(", ")} heading` };
  }
  const start = headings.indexOf(expected[0]);
  const actual = headings.slice(start, start + expected.length);
  if (actual.join(" → ") !== expected.join(" → ")) {
    return {
      ok: false,
      reason: `headings run ${headings.join(" → ")}; ${expected.join(" → ")} are not adjacent in that order`,
    };
  }
  return { ok: true, reason: `${expected.join(" → ")} are adjacent, in order` };
}

/**
 * The row is supposed to offer at most the next step, a way back to the folder
 * picker, and — once granted — a way out. Anything called "authorize" would
 * mean the consent step became a chore of its own again, and a "Manage" button
 * would mean the menu came back.
 */
export function judgeLibraryButtons(snapshot, { expected } = {}) {
  const { rowButtons, groupButtons } = snapshot;
  const problems = [];
  if (rowButtons.length > 3) {
    problems.push(`the row shows ${rowButtons.length} buttons (${rowButtons.join(", ")}), more than three`);
  }
  if (expected && rowButtons.join(" | ") !== expected.join(" | ")) {
    problems.push(`the row shows [${rowButtons.join(", ")}], expected [${expected.join(", ")}]`);
  }
  const authorize = groupButtons.filter((text) => /authorize/i.test(text));
  if (authorize.length > 0) {
    problems.push(`the section still offers an authorization button: ${authorize.join(", ")}`);
  }
  const manage = groupButtons.filter((text) => /^manage$/i.test(text));
  if (manage.length > 0) problems.push(`the row still offers a Manage menu: ${manage.join(", ")}`);
  return problems.length === 0
    ? { ok: true, reason: `[${rowButtons.join(", ")}], no authorization button, no Manage` }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * What a layout engine can settle and happy-dom cannot: the buttons sit on one
 * line, hug the right edge of their control, and stay inside it.
 *
 * "Inside the control" is the load-bearing one. `.setting-item-control` is a
 * flex item that shares the row with the name and description; when it is
 * allowed to shrink below its content the buttons do not shrink with it, they
 * spill leftwards over the text.
 *
 * No library scenario applies this any more: demanding one line at every panel
 * width is what crushed the two-button row's description, so both button states
 * are now judged by `judgeLibraryWrappedGeometry`, which holds everything here
 * except the one-line rule and adds the readability floor that outranks it.
 * Kept because a row that must not wrap is still a thing a settings page can
 * want, and this is the tested statement of it.
 */
export function judgeLibraryGeometry(geometry, { tolerance = ALIGNMENT_TOLERANCE_PX } = {}) {
  const { buttons, control, row, info } = geometry;
  const problems = [];
  const notes = [];

  const tops = buttons.map((button) => button.top);
  const spread = Math.max(...tops) - Math.min(...tops);
  if (spread > tolerance) {
    problems.push(
      `the buttons are on ${new Set(tops.map(round)).size} different lines (tops ${tops.map(round).join(", ")})`,
    );
  } else {
    notes.push(`one line (top spread ${round(spread)}px)`);
  }

  const rightMost = Math.max(...buttons.map((button) => button.right));
  const rightGap = control.right - rightMost;
  if (Math.abs(rightGap) > tolerance) {
    problems.push(`the last button ends ${round(rightGap)}px from the control's right edge, not flush with it`);
  } else {
    notes.push("right-aligned");
  }

  const leftMost = Math.min(...buttons.map((button) => button.left));
  const leftOverflow = control.left - leftMost;
  const rightOverflow = rightMost - control.right;
  if (leftOverflow > tolerance || rightOverflow > tolerance) {
    problems.push(
      `the buttons span ${round(leftMost)}..${round(rightMost)} but their control box is `
        + `${round(control.left)}..${round(control.right)} — they overflow it by `
        + `${round(Math.max(leftOverflow, 0))}px on the left and ${round(Math.max(rightOverflow, 0))}px on the right`,
    );
  } else {
    notes.push("inside its control");
  }

  if (leftMost < row.left - tolerance || rightMost > row.right + tolerance) {
    problems.push(
      `the buttons span ${round(leftMost)}..${round(rightMost)}, outside the row's ${round(row.left)}..${round(row.right)}`,
    );
  }

  if (info && leftMost < info.right - tolerance) {
    problems.push(
      `the leftmost button starts at ${round(leftMost)}, ${round(info.right - leftMost)}px inside the name and `
        + `description column that ends at ${round(info.right)} — the buttons cover the description`,
    );
  }

  return problems.length === 0
    ? { ok: true, reason: notes.join(", ") }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * Whether the description is still a sentence.
 *
 * "No overlap" was never the whole of readable: the three-button row passed
 * every geometric assertion above while giving the description a six-pixel
 * column and spelling it out one letter per line. Nothing overlapped, nothing
 * overflowed, and nobody could read it. So this judges what a reader gets: how
 * wide the column is, and how much of the sentence fits on a line of it.
 */
export function judgeDescriptionReadable(description, {
  minWidth = MIN_DESCRIPTION_WIDTH_PX,
  minCharactersPerLine = MIN_DESCRIPTION_CHARACTERS_PER_LINE,
} = {}) {
  if (!description) {
    return { ok: false, reason: "the Library row has no description to measure" };
  }
  const problems = [];
  if (description.contentWidth < minWidth) {
    problems.push(
      `the description column is ${round(description.contentWidth)}px wide, under the ${minWidth}px `
        + "a sentence needs to read as one",
    );
  }
  const perLine = description.lines > 0 ? description.characters / description.lines : 0;
  if (description.lines === 0) {
    problems.push(`the description "${description.text.slice(0, 60)}" laid out into no line boxes at all`);
  } else if (perLine < minCharactersPerLine) {
    problems.push(
      `its ${description.characters} characters are spread over ${description.lines} lines — `
        + `${round(perLine)} characters a line, under ${minCharactersPerLine}`,
    );
  }
  return problems.length === 0
    ? {
        ok: true,
        reason: `description ${round(description.contentWidth)}px wide, `
          + `${round(perLine)} characters over ${description.lines} line${description.lines === 1 ? "" : "s"}`,
      }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * The library row at any panel width, carrying any number of buttons.
 *
 * One line is not the promise; a readable row is. Holding the strip on one line
 * is what crushed the description into a column of letters, so this drops the
 * one-line rule and states the invariant that outranks it: whatever line a
 * button lands on, it stays inside the control, inside the row and off the
 * description; every line still hugs the right edge, so wrapping cannot become
 * an excuse for a ragged strip; the description stays readable; and the main
 * call to action is laid out somewhere a person can see and click it.
 *
 * Deliberately says nothing about how many lines the strip uses. The layout is
 * built so one line is what happens whenever one line fits — see the library
 * row rules in `plugin/styles.css` — and the line count each panel width
 * actually produces is reported in the passing detail rather than asserted,
 * because the number of buttons the row carries is not this judge's business.
 */
export function judgeLibraryWrappedGeometry(geometry, {
  tolerance = ALIGNMENT_TOLERANCE_PX,
  mainCallToAction = "Build index",
  ...readability
} = {}) {
  const { buttons, control, row, info, description } = geometry;
  const problems = [];
  const notes = [];

  const leftMost = Math.min(...buttons.map((button) => button.left));
  const rightMost = Math.max(...buttons.map((button) => button.right));

  const leftOverflow = control.left - leftMost;
  const rightOverflow = rightMost - control.right;
  if (leftOverflow > tolerance || rightOverflow > tolerance) {
    problems.push(
      `the buttons span ${round(leftMost)}..${round(rightMost)} but their control box is `
        + `${round(control.left)}..${round(control.right)} — they overflow it by `
        + `${round(Math.max(leftOverflow, 0))}px on the left and ${round(Math.max(rightOverflow, 0))}px on the right`,
    );
  }

  if (leftMost < row.left - tolerance || rightMost > row.right + tolerance) {
    problems.push(
      `the buttons span ${round(leftMost)}..${round(rightMost)}, outside the row's `
        + `${round(row.left)}..${round(row.right)}`,
    );
  }

  if (info && leftMost < info.right - tolerance) {
    problems.push(
      `the leftmost button starts at ${round(leftMost)}, ${round(info.right - leftMost)}px inside the name and `
        + `description column that ends at ${round(info.right)} — the buttons cover the description`,
    );
  }

  // Wrapping is allowed; drifting away from the right edge is not. Every line
  // the buttons land on has to end flush with the control, or the strip has
  // stopped being right-aligned and merely happens to fit.
  const lines = new Map();
  for (const button of buttons) {
    const key = round(button.top);
    lines.set(key, [...(lines.get(key) ?? []), button]);
  }
  const ragged = [...lines.entries()]
    .map(([top, line]) => ({ top, gap: control.right - Math.max(...line.map((b) => b.right)) }))
    .filter((line) => Math.abs(line.gap) > tolerance);
  if (ragged.length > 0) {
    problems.push(
      `${ragged.length} of ${lines.size} button lines do not end at the control's right edge `
        + `(${ragged.map((line) => `line at ${line.top} is ${round(line.gap)}px short`).join(", ")})`,
    );
  } else {
    notes.push(`${lines.size} line${lines.size === 1 ? "" : "s"}, each right-aligned`);
  }

  const cta = buttons.find((button) => button.text === mainCallToAction);
  if (!cta) {
    problems.push(
      `the row has no ${mainCallToAction} button to keep visible; it shows `
        + `[${buttons.map((b) => b.text).join(", ")}]`,
    );
  } else if (cta.width <= 0 || cta.height <= 0) {
    problems.push(`${mainCallToAction} is laid out ${round(cta.width)}x${round(cta.height)} — it is not visible`);
  } else if (cta.left < row.left - tolerance || cta.right > row.right + tolerance) {
    problems.push(
      `${mainCallToAction} spans ${round(cta.left)}..${round(cta.right)}, clipped by the row's `
        + `${round(row.left)}..${round(row.right)}`,
    );
  } else {
    notes.push(`${mainCallToAction} visible at ${round(cta.left)}..${round(cta.right)}`);
  }

  const readable = judgeDescriptionReadable(description, readability);
  if (readable.ok) notes.push(readable.reason);
  else problems.push(readable.reason);

  return problems.length === 0
    ? { ok: true, reason: notes.join(", ") }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * The same row below Obsidian's stacking breakpoint, where the row becomes a
 * column and every button is stretched to the full width. One line and right
 * alignment stop meaning anything there, so this judges only what still has to
 * hold: whatever line a button lands on, it stays inside the control and inside
 * the row. Buttons held on one line here run straight out of the panel.
 */
export function judgeLibraryStackedGeometry(geometry, { tolerance = ALIGNMENT_TOLERANCE_PX } = {}) {
  const { buttons, control, row } = geometry;
  const problems = [];

  const leftMost = Math.min(...buttons.map((button) => button.left));
  const rightMost = Math.max(...buttons.map((button) => button.right));

  const leftOverflow = control.left - leftMost;
  const rightOverflow = rightMost - control.right;
  if (leftOverflow > tolerance || rightOverflow > tolerance) {
    problems.push(
      `the buttons span ${round(leftMost)}..${round(rightMost)} but their control box is `
        + `${round(control.left)}..${round(control.right)} — they overflow it by `
        + `${round(Math.max(leftOverflow, 0))}px on the left and ${round(Math.max(rightOverflow, 0))}px on the right`,
    );
  }

  if (leftMost < row.left - tolerance || rightMost > row.right + tolerance) {
    problems.push(
      `the buttons span ${round(leftMost)}..${round(rightMost)}, outside the row's ${round(row.left)}..${round(row.right)}`,
    );
  }

  const lines = new Set(buttons.map((button) => round(button.top))).size;
  return problems.length === 0
    ? { ok: true, reason: `stacked over ${lines} line${lines === 1 ? "" : "s"}, inside its control and its row` }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * The row while a run is in flight.
 *
 * Progress already existed — the run pushes every step to the status bar — and
 * the settings modal is drawn over that status bar, so this judges the one thing
 * that was missing: that the row a person starts the run from is the row that
 * shows it. Three claims, because a row that shows the count while still
 * offering to start a second run, or to change the folder out from under one,
 * has not actually said a run is happening.
 */
export function judgeIndexingRow(snapshot, { expectedLabel, phase } = {}) {
  const states = snapshot?.rowButtonStates;
  if (!Array.isArray(states) || states.length === 0) {
    return { ok: false, reason: "the Library row rendered no buttons while a run was in flight" };
  }
  const problems = [];
  const labels = states.map((button) => button.text);
  const expected = ["Change folder", expectedLabel, "Cancel"];
  if (labels.join(" | ") !== expected.join(" | ")) {
    problems.push(`the row shows [${labels.join(", ")}], expected [${expected.join(", ")}]`);
  }
  const main = states.find((button) => button.text === expectedLabel);
  if (main && !main.disabled) {
    problems.push(`"${expectedLabel}" is still clickable, so the row offers to start a second run`);
  }
  const folder = states.find((button) => button.text === "Change folder");
  if (folder && !folder.disabled) {
    problems.push("Change folder is still clickable, so the folder can be moved out from under the run");
  }
  const cancel = states.find((button) => button.text === "Cancel");
  if (!cancel) problems.push("the row offers no way to stop the run");
  else if (cancel.disabled) problems.push("the row's Cancel is disabled, so the run cannot be stopped");
  if (phase && !(snapshot.rowDescription ?? "").includes(phase)) {
    problems.push(
      `the description does not say what the run is doing; it reads "${(snapshot.rowDescription ?? "").slice(0, 120)}"`,
    );
  }
  return problems.length === 0
    ? { ok: true, reason: `[${labels.join(", ")}], only Cancel clickable, description names the phase` }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * That a reported step rewrote the row rather than redrawing the page.
 *
 * The counts arrive once per paper. Re-rendering the settings page at that rate
 * would discard whatever else is being edited on it, so the row is written in
 * place — and "in place" is only checkable by identity: the marks were stamped
 * on the button elements, so surviving marks mean the same elements, while the
 * changed label means they were actually updated.
 */
export function judgeInPlaceProgressUpdate(before, after, { expectedLabel } = {}) {
  const problems = [];
  const marksBefore = (before?.rowButtonStates ?? []).map((button) => button.mark);
  const marksAfter = (after?.rowButtonStates ?? []).map((button) => button.mark);
  if (marksBefore.some((mark) => mark === null)) {
    return { ok: false, reason: "the row's buttons were never marked, so identity cannot be compared" };
  }
  if (marksAfter.join("|") !== marksBefore.join("|")) {
    problems.push(
      `the buttons were replaced (marks ${marksBefore.join(",")} → ${marksAfter.join(",")}), `
        + "so the page re-rendered instead of updating the row",
    );
  }
  const label = (after?.rowButtonStates ?? []).map((button) => button.text)
    .find((text) => text.startsWith("Indexing"));
  if (label !== expectedLabel) {
    problems.push(`the main button reads "${label}", expected "${expectedLabel}" after the report`);
  }
  return problems.length === 0
    ? { ok: true, reason: `the same buttons now read "${expectedLabel}"` }
    : { ok: false, reason: problems.join("; ") };
}

/** The row once cancellation has been asked for, and the run it asked. */
export function judgeCancellingRow(snapshot, operations) {
  const problems = [];
  const cancel = (snapshot?.rowButtonStates ?? []).find((button) => /^Cancel/.test(button.text));
  if (!cancel) {
    problems.push("the row lost its stop control instead of showing that it is stopping");
  } else {
    if (cancel.text !== "Cancelling…") {
      problems.push(`the stop control still reads "${cancel.text}", inviting a second press`);
    }
    if (!cancel.disabled) problems.push("the stop control is still clickable after being pressed");
  }
  const requested = (operations ?? []).filter((operation) => operation.cancellationRequested);
  if (requested.length === 0) {
    problems.push(
      `pressing it asked no operation to stop; the registry holds ${(operations ?? []).length} `
        + "indexing operation(s), none cancelled",
    );
  }
  return problems.length === 0
    ? { ok: true, reason: `the row reads "Cancelling…" and ${requested.length} run was asked to stop` }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * The sentence a finished run leaves behind.
 *
 * The status bar hides its completion after four seconds and the notice after
 * ten, so without this an index that ran overnight leaves nothing saying whether
 * it finished. Both halves are required: a time alone does not say what is
 * searchable, and a count alone does not say when.
 */
export function judgeIndexTraceSentence(snapshot, { papers } = {}) {
  const description = snapshot?.rowDescription ?? "";
  const problems = [];
  if (!/Last indexed \d{4}-\d{2}-\d{2} \d{2}:\d{2}/.test(description)) {
    problems.push(`the row does not say when the index was last built: "${description.slice(0, 160)}"`);
  }
  const expected = papers === 1 ? "1 paper searchable" : `${papers} papers searchable`;
  if (!description.includes(expected)) {
    problems.push(`the row does not say "${expected}"; it reads "${description.slice(0, 160)}"`);
  }
  return problems.length === 0
    ? { ok: true, reason: `the idle row reads "${description.slice(-60)}"` }
    : { ok: false, reason: problems.join("; ") };
}

/**
 * The disclosure heading, judged against the depth the dialog is disclosing.
 *
 * Separate from finding the dialog on purpose. The lookup uses the stable class,
 * so a reworded heading reaches this judge and fails as "titled X, expected Y"
 * rather than as "no dialog opened" — the second reads like the consent step
 * broke, which is a lie about what happened.
 */
export function judgeDisclosureTitle(modal, depth, titles = DISCLOSURE_TITLES) {
  const expected = titles[depth];
  if (!expected) {
    return { ok: false, reason: `no heading is specified for ${depth} depth` };
  }
  const actual = (modal?.title ?? "").trim();
  if (actual !== expected) {
    return {
      ok: false,
      reason: `the disclosure is titled "${actual}", expected "${expected}" at ${depth} depth`,
    };
  }
  return { ok: true, reason: `titled "${actual}" at ${depth} depth` };
}

/**
 * The two answers, judged against the depth the dialog is disclosing.
 *
 * Separate from finding them, exactly as the heading is separate from finding
 * the dialog. The buttons are located by their marks, so a reworded confirm
 * label reaches this judge and fails as `reads "Authorize", expected "Send full
 * text"` rather than as "the modal has no Authorize button" — the second reads
 * like the dialog lost its affirmative, which is a lie about what happened.
 *
 * A missing mark is still reported, and says so in those words, because that is
 * a different failure: the button really is not there.
 */
export function judgeDisclosureButtons(modal, depth, {
  confirmLabels = DISCLOSURE_CONFIRM_LABELS,
  cancelLabel = DISCLOSURE_CANCEL_LABEL,
} = {}) {
  const expected = confirmLabels[depth];
  if (!expected) {
    return { ok: false, reason: `no confirm label is specified for ${depth} depth` };
  }
  const problems = [];
  const confirm = modal?.confirm ?? null;
  const cancel = modal?.cancel ?? null;
  if (confirm === null) {
    problems.push(`the disclosure has no .${DISCLOSURE_CONFIRM_BUTTON_CLASS} button`);
  } else if (confirm !== expected) {
    problems.push(
      `the confirm button reads "${confirm}", expected "${expected}" at ${depth} depth`,
    );
  }
  if (cancel === null) {
    problems.push(`the disclosure has no .${DISCLOSURE_CANCEL_BUTTON_CLASS} button`);
  } else if (cancel !== cancelLabel) {
    problems.push(`the cancel button reads "${cancel}", expected "${cancelLabel}"`);
  }
  return problems.length === 0
    ? { ok: true, reason: `answers "${cancel}" / "${confirm}" at ${depth} depth` }
    : { ok: false, reason: problems.join("; ") };
}

/** Nothing about the grant or the mode may move when a disclosure is declined. */
export function judgeUnchanged(before, after) {
  const problems = [];
  if (after.embedding.mode !== before.embedding.mode) {
    problems.push(`embedding.mode moved ${before.embedding.mode} → ${after.embedding.mode}`);
  }
  if (after.status.kind !== before.status.kind) {
    problems.push(`the connection moved ${before.status.kind} → ${after.status.kind}`);
  }
  if (JSON.stringify(after.embedding) !== JSON.stringify(before.embedding)) {
    problems.push("the embedding settings changed");
  }
  return problems.length === 0 ? { ok: true, reason: "" } : { ok: false, reason: problems.join("; ") };
}

// ── the scenario ────────────────────────────────────────────────────────────

/**
 * Walks the personal library settings page through the states the branch
 * changed, asserting each one and leaving a screenshot behind.
 *
 * Returns one result per assertion rather than a single verdict, so a failure
 * names the behaviour that broke instead of the whole page.
 */
export async function librarySettingsScenarios({
  session,
  screenshots,
  narrowViewport = NARROW_VIEWPORT,
  wideViewport = WIDE_VIEWPORT,
  captureViewport = CAPTURE_VIEWPORT,
  stackedViewport = STACKED_VIEWPORT,
  settleMs = 600,
}) {
  const { evaluate, client, diagnostics } = session;
  const errorsBefore = diagnostics.errors().length;
  const results = [];
  const shot = async (name, where) => {
    if (!screenshots) return;
    await screenshots.capture(name, where);
  };
  const settledRect = async (scrollExpression, rectExpression) => {
    await evaluate(scrollExpression);
    await wait(evaluate, settleMs);
    const raw = await evaluate(rectExpression);
    return typeof raw === "string" ? JSON.parse(raw) : null;
  };
  const sectionRect = () => settledRect(SCROLL_TO_SECTION_EXPRESSION, LIBRARY_SECTION_RECT_EXPRESSION);
  const rowRect = () => settledRect(SCROLL_TO_LIBRARY_ROW_EXPRESSION, LIBRARY_ROW_RECT_EXPRESSION);

  await setViewport(client, captureViewport);
  await evaluate(OPEN_SETTINGS_EXPRESSION);
  await wait(evaluate, settleMs * 3);

  // 1 — group order
  const headings = await readJson(evaluate, HEADINGS_EXPRESSION);
  if (headings.error) {
    results.push(fail("library-settings-group-order", headings.error));
  } else {
    const verdict = judgeGroupOrder(headings.headings);
    results.push(
      (verdict.ok ? pass : fail)("library-settings-group-order", verdict.reason),
    );
  }

  // 2 — the row's buttons, with a local library selected and no grant
  const local = await readJson(evaluate, LIBRARY_ROW_EXPRESSION);
  if (local.error) {
    results.push(fail("library-row-buttons-local", local.error));
  } else {
    const verdict = judgeLibraryButtons(local, { expected: ["Change folder", "Build index"] });
    results.push((verdict.ok ? pass : fail)("library-row-buttons-local", verdict.reason));
  }
  await shot("personal-library-section-local-embedding", { rect: await sectionRect() });

  // 3 — layout, at two panel widths a user can reach
  //
  // Judged by the same rule as the granted three-button row: the description
  // stays readable, and the strip wraps rather than squeeze it. This assertion
  // used to demand one line at every width, which is what pinned the two-button
  // row's description to an 81px column at the narrow panel.
  const measurements = [];
  for (const [label, viewport] of [["narrow", narrowViewport], ["wide", wideViewport]]) {
    await setViewport(client, viewport);
    await wait(evaluate, settleMs);
    const rect = await rowRect();
    const geometry = await readJson(evaluate, LIBRARY_GEOMETRY_EXPRESSION);
    measurements.push({ label, viewport, geometry });
    await shot(`library-row-${label}-panel`, { rect });
  }
  // The stacked layout below Obsidian's own breakpoint is a different layout,
  // so it is judged by a different rule — see judgeLibraryStackedGeometry.
  await setViewport(client, stackedViewport);
  await wait(evaluate, settleMs);
  const stackedRect = await rowRect();
  const stacked = await readJson(evaluate, LIBRARY_GEOMETRY_EXPRESSION);
  await shot("library-row-stacked-panel", { rect: stackedRect });
  await setViewport(client, captureViewport);
  await wait(evaluate, settleMs);

  const geometryProblems = [];
  const geometryNotes = [];
  for (const { label, geometry } of measurements) {
    if (geometry.error) {
      geometryProblems.push(`${label}: ${geometry.error}`);
      continue;
    }
    const verdict = judgeLibraryWrappedGeometry(geometry);
    const width = `${label} panel ${round(geometry.panelWidth)}px (window ${geometry.windowWidth})`;
    if (verdict.ok) geometryNotes.push(`${width}: ${verdict.reason}`);
    else geometryProblems.push(`${width}: ${verdict.reason}`);
  }
  results.push(
    geometryProblems.length === 0
      ? pass("library-row-geometry", geometryNotes.join("; "))
      : fail("library-row-geometry", geometryProblems.join(" | ")),
  );

  if (stacked.error) {
    results.push(fail("library-row-geometry-stacked", stacked.error));
  } else {
    const verdict = judgeLibraryStackedGeometry(stacked);
    const width = `stacked panel ${round(stacked.panelWidth)}px (window ${stacked.windowWidth})`;
    results.push(
      (verdict.ok ? pass : fail)("library-row-geometry-stacked", `${width}: ${verdict.reason}`),
    );
  }

  // 3b — the run on the row: progress, an in-place rewrite, and a way to stop
  //
  // Started through the plugin's own operation registry, so Cancel below aborts
  // a real signal. The indexing work itself is not run — it needs an embedding
  // model and minutes of extraction, and none of that changes what the row
  // looks like. That a real run reports to this same store is asserted in the
  // plugin units; what only a renderer can answer is what the row does with it.
  const started = await readJson(evaluate, beginIndexRunExpression({
    phase: "extracting and embedding PDF text",
    completed: 12,
    total: 40,
  }));
  // The row is expected to pick the run up on its own — nothing re-opens the
  // settings tab here.
  await wait(evaluate, settleMs * 2);
  const running = await readJson(evaluate, LIBRARY_ROW_EXPRESSION);
  if (running.error) {
    results.push(fail("library-row-shows-the-run", running.error));
  } else {
    const verdict = judgeIndexingRow(running, {
      expectedLabel: "Indexing… (12/40)",
      phase: "extracting and embedding PDF text",
    });
    results.push((verdict.ok ? pass : fail)(
      "library-row-shows-the-run",
      `${verdict.reason} (operation ${started.operationId})`,
    ));
  }
  await shot("library-row-indexing", { rect: await rowRect() });

  // Geometry gets its own verdict because the run puts a third button on the
  // row and a longer sentence beside it: the state that has to stay readable is
  // not the one the earlier assertions measured.
  const runningGeometry = [];
  for (const [label, viewport] of [["narrow", narrowViewport], ["wide", wideViewport]]) {
    await setViewport(client, viewport);
    await wait(evaluate, settleMs);
    const rect = await rowRect();
    runningGeometry.push({ label, geometry: await readJson(evaluate, LIBRARY_GEOMETRY_EXPRESSION) });
    await shot(`library-row-indexing-${label}-panel`, { rect });
  }
  await setViewport(client, captureViewport);
  await wait(evaluate, settleMs);
  const runningProblems = [];
  const runningNotes = [];
  for (const { label, geometry } of runningGeometry) {
    if (geometry.error) {
      runningProblems.push(`${label}: ${geometry.error}`);
      continue;
    }
    const verdict = judgeLibraryWrappedGeometry(geometry, { mainCallToAction: "Indexing… (12/40)" });
    const width = `${label} panel ${round(geometry.panelWidth)}px (window ${geometry.windowWidth})`;
    if (verdict.ok) runningNotes.push(`${width}: ${verdict.reason}`);
    else runningProblems.push(`${width}: ${verdict.reason}`);
  }
  results.push(
    runningProblems.length === 0
      ? pass("library-row-indexing-geometry", runningNotes.join("; "))
      : fail("library-row-indexing-geometry", runningProblems.join(" | ")),
  );

  // The next count must rewrite the row rather than redraw the page.
  await evaluate(MARK_LIBRARY_BUTTONS_EXPRESSION);
  const marked = await readJson(evaluate, LIBRARY_ROW_EXPRESSION);
  await evaluate(reportIndexProgressExpression({
    phase: "extracting and embedding PDF text",
    completed: 13,
    total: 40,
  }));
  await wait(evaluate, settleMs);
  const advanced = await readJson(evaluate, LIBRARY_ROW_EXPRESSION);
  if (marked.error || advanced.error) {
    results.push(fail("library-row-progress-updates-in-place", marked.error ?? advanced.error));
  } else {
    const verdict = judgeInPlaceProgressUpdate(marked, advanced, {
      expectedLabel: "Indexing… (13/40)",
    });
    results.push((verdict.ok ? pass : fail)("library-row-progress-updates-in-place", verdict.reason));
  }

  // Cancel, from the same row, and the run has to hear it.
  const cancelClicked = await readJson(evaluate, clickLibraryRowButtonExpression("Cancel"));
  if (cancelClicked.error) {
    results.push(fail("library-row-cancels-the-run", cancelClicked.error));
  } else {
    await wait(evaluate, settleMs * 2);
    const stopping = await readJson(evaluate, LIBRARY_ROW_EXPRESSION);
    const registry = await readJson(evaluate, INDEX_OPERATIONS_EXPRESSION);
    const verdict = judgeCancellingRow(stopping, registry.operations);
    results.push((verdict.ok ? pass : fail)("library-row-cancels-the-run", verdict.reason));
    await shot("library-row-cancelling", { rect: await rowRect() });
  }

  // 3c — what the finished run leaves behind
  //
  // The status bar hides its completion after four seconds and the notice after
  // ten, so this sentence is the whole of what an overnight run leaves for the
  // person who comes back to it.
  await evaluate(setLastIndexRunExpression({ updatedAt: "2026-08-30T21:15:00.000Z", papers: 128 }));
  await evaluate(END_INDEX_RUN_EXPRESSION);
  await wait(evaluate, settleMs * 2);
  const afterRun = await readJson(evaluate, LIBRARY_ROW_EXPRESSION);
  if (afterRun.error) {
    results.push(fail("library-row-keeps-the-index-trace", afterRun.error));
  } else {
    const problems = [];
    const trace = judgeIndexTraceSentence(afterRun, { papers: 128 });
    if (!trace.ok) problems.push(trace.reason);
    // The row also has to come back: a run that ends leaving its own controls
    // behind would be worse than never showing them.
    const backToIdle = judgeLibraryButtons(afterRun, { expected: ["Change folder", "Build index"] });
    if (!backToIdle.ok) problems.push(backToIdle.reason);
    results.push(
      problems.length === 0
        ? pass("library-row-keeps-the-index-trace", `${trace.reason}; ${backToIdle.reason}`)
        : fail("library-row-keeps-the-index-trace", problems.join("; ")),
    );
    await shot("library-row-last-indexed", { rect: await rowRect() });
  }
  await evaluate(CLEAR_LAST_INDEX_RUN_EXPRESSION);
  await wait(evaluate, settleMs);

  // 4 — switching to remote asks in place
  const beforeSwitch = await readJson(evaluate, PLUGIN_STATE_EXPRESSION);
  const dispatched = await readJson(evaluate, selectEmbeddingModeExpression("remote"));
  if (dispatched.error) {
    results.push(fail("remote-switch-asks-in-place", dispatched.error));
    results.push(fail("remote-disclosure-title", dispatched.error));
    results.push(fail("remote-disclosure-buttons", dispatched.error));
  } else {
    const modal = await waitForModal(evaluate);
    if (!modal.present) {
      results.push(fail(
        "remote-switch-asks-in-place",
        `switching Embedding to remote opened no .${DISCLOSURE_MODAL_CLASS} dialog`,
      ));
    } else if (!/full text/i.test(modal.text)) {
      results.push(fail(
        "remote-switch-asks-in-place",
        `the dialog opened but never mentions full text: ${modal.text.slice(0, 200)}`,
      ));
    } else {
      await shot("remote-full-text-disclosure-modal", { rect: modal.rect });
      results.push(pass(
        "remote-switch-asks-in-place",
        `the dropdown alone opened the disclosure offering ${modal.buttons.join(" / ")}`,
      ));
    }

    // 4b — the heading, judged apart from the lookup that found the dialog.
    //
    // Remote embedding is the only depth this page can disclose at, so this is
    // the full-text heading; the metadata one is asserted in the plugin units.
    if (!modal.present) {
      results.push(fail(
        "remote-disclosure-title",
        `no .${DISCLOSURE_MODAL_CLASS} dialog opened, so its heading could not be read`,
      ));
    } else {
      const verdict = judgeDisclosureTitle(modal, "full-text");
      results.push((verdict.ok ? pass : fail)("remote-disclosure-title", verdict.reason));
    }

    // 4c — the two answers, judged the same way and apart from the marks the
    // clicks below use to find them. The confirm button has to answer the
    // heading in the heading's own words; "Authorize" would be a different
    // concept the reader has to translate. Only the full-text pair is reachable
    // here, for the same reason as the heading.
    if (!modal.present) {
      results.push(fail(
        "remote-disclosure-buttons",
        `no .${DISCLOSURE_MODAL_CLASS} dialog opened, so its buttons could not be read`,
      ));
    } else {
      const verdict = judgeDisclosureButtons(modal, "full-text");
      results.push((verdict.ok ? pass : fail)("remote-disclosure-buttons", verdict.reason));
    }

    // 5 — declining puts everything back
    if (modal.present) {
      await evaluate(clickModalButtonExpression(DISCLOSURE_CANCEL_BUTTON_CLASS));
      await waitForModalGone(evaluate);
      await wait(evaluate, settleMs * 2);
      const afterCancel = await readJson(evaluate, PLUGIN_STATE_EXPRESSION);
      const select = await readJson(evaluate, EMBEDDING_SELECT_VALUE_EXPRESSION);
      const unchanged = judgeUnchanged(beforeSwitch, afterCancel);
      if (select.value !== "local") {
        results.push(fail(
          "declined-remote-switch-changes-nothing",
          `the dropdown still shows ${JSON.stringify(select.value)} after the dialog was cancelled`,
        ));
      } else if (!unchanged.ok) {
        results.push(fail("declined-remote-switch-changes-nothing", unchanged.reason));
      } else {
        results.push(pass(
          "declined-remote-switch-changes-nothing",
          `the dropdown returned to local, embedding.mode stayed ${afterCancel.embedding.mode} and the `
            + `connection stayed ${afterCancel.status.kind}`,
        ));
      }
    }
  }

  // 6 — remote but ungranted: indexing asks first, and declining starts nothing
  //
  // The mode is moved through the plugin's own settings transaction rather than
  // the dropdown, because accepting the dropdown's disclosure would also grant.
  // That is exactly the configuration this gate exists for: remote embedding
  // arriving without having passed the in-place consent.
  await evaluate(
    `${PLUGIN}.settingsChanges.changeValue("embedding.mode", "remote").then(() => "changed")`,
  );
  await evaluate(OPEN_SETTINGS_EXPRESSION);
  await wait(evaluate, settleMs * 2);

  const remote = await readJson(evaluate, LIBRARY_ROW_EXPRESSION);
  if (remote.error) {
    results.push(fail("library-row-buttons-remote", remote.error));
  } else {
    const verdict = judgeLibraryButtons(remote, { expected: ["Change folder", "Build index"] });
    results.push((verdict.ok ? pass : fail)("library-row-buttons-remote", verdict.reason));
  }
  await shot("personal-library-section-remote-embedding", { rect: await sectionRect() });

  const beforeIndex = await readJson(evaluate, PLUGIN_STATE_EXPRESSION);
  const clicked = await readJson(evaluate, clickLibraryRowButtonExpression("Build index"));
  if (clicked.error) {
    results.push(fail("build-index-asks-before-remote-indexing", clicked.error));
  } else {
    const modal = await waitForModal(evaluate);
    if (!modal.present) {
      results.push(fail(
        "build-index-asks-before-remote-indexing",
        `Build index started without opening the .${DISCLOSURE_MODAL_CLASS} dialog`,
      ));
    } else {
      await evaluate(clickModalButtonExpression(DISCLOSURE_CANCEL_BUTTON_CLASS));
      await waitForModalGone(evaluate);
      await wait(evaluate, settleMs * 3);
      const afterCancel = await readJson(evaluate, PLUGIN_STATE_EXPRESSION);
      const unchanged = judgeUnchanged(beforeIndex, afterCancel);
      const indexing = afterCancel.operations.filter((kind) => kind.includes("fulltext"));
      const started = afterCancel.notices.filter((text) => /indexing personal library full text/i.test(text));
      if (!unchanged.ok) {
        results.push(fail("build-index-asks-before-remote-indexing", unchanged.reason));
      } else if (indexing.length > 0) {
        results.push(fail(
          "build-index-asks-before-remote-indexing",
          `the disclosure was cancelled but ${indexing.join(", ")} is running`,
        ));
      } else if (started.length > 0) {
        results.push(fail(
          "build-index-asks-before-remote-indexing",
          `the disclosure was cancelled but indexing announced itself: ${started.join(" / ")}`,
        ));
      } else {
        results.push(pass(
          "build-index-asks-before-remote-indexing",
          `Build index asked first; cancelling started no indexing operation and left embedding.mode `
            + `${afterCancel.embedding.mode} with the connection ${afterCancel.status.kind}`,
        ));
      }
    }
  }

  // 7 — the granted state, where the row carries three buttons
  //
  // Reached the way a person reaches it: the mode goes back to local, the
  // dropdown asks for remote, and this time the disclosure is accepted. Going
  // through Build index would grant too, but it would also start indexing.
  await evaluate(
    `${PLUGIN}.settingsChanges.changeValue("embedding.mode", "local").then(() => "changed")`,
  );
  await evaluate(OPEN_SETTINGS_EXPRESSION);
  await wait(evaluate, settleMs * 2);
  const granting = await readJson(evaluate, selectEmbeddingModeExpression("remote"));
  let granted = false;
  if (!granting.error) {
    const modal = await waitForModal(evaluate);
    if (modal.present) {
      await evaluate(clickModalButtonExpression(DISCLOSURE_CONFIRM_BUTTON_CLASS));
      await waitForModalGone(evaluate);
      await wait(evaluate, settleMs * 3);
      await evaluate(OPEN_SETTINGS_EXPRESSION);
      await wait(evaluate, settleMs * 2);
      granted = true;
    }
  }

  const authorized = granted ? await readJson(evaluate, LIBRARY_ROW_EXPRESSION) : { error: null };
  if (!granted || authorized.error || authorized.rowButtons.length !== 3) {
    const why = !granted
      ? "accepting the disclosure from the Embedding dropdown never produced a granted library"
      : authorized.error
        ?? `the granted row shows [${authorized.rowButtons.join(", ")}], not the three-button state`;
    results.push(fail("library-row-three-buttons-geometry", why));
    results.push(fail("library-row-three-buttons-geometry-stacked", why));
  } else {
    await shot("personal-library-section-authorized", { rect: await sectionRect() });

    const wide = [];
    for (const [label, viewport] of [["narrow", narrowViewport], ["wide", wideViewport]]) {
      await setViewport(client, viewport);
      await wait(evaluate, settleMs);
      const rect = await rowRect();
      wide.push({ label, geometry: await readJson(evaluate, LIBRARY_GEOMETRY_EXPRESSION) });
      await shot(`library-row-three-buttons-${label}-panel`, { rect });
    }
    await setViewport(client, stackedViewport);
    await wait(evaluate, settleMs);
    const threeStackedRect = await rowRect();
    const threeStacked = await readJson(evaluate, LIBRARY_GEOMETRY_EXPRESSION);
    await shot("library-row-three-buttons-stacked-panel", { rect: threeStackedRect });
    await setViewport(client, captureViewport);
    await wait(evaluate, settleMs);

    const threeProblems = [];
    const threeNotes = [];
    for (const { label, geometry } of wide) {
      if (geometry.error) {
        threeProblems.push(`${label}: ${geometry.error}`);
        continue;
      }
      const verdict = judgeLibraryWrappedGeometry(geometry);
      const width = `${label} panel ${round(geometry.panelWidth)}px (window ${geometry.windowWidth})`;
      if (verdict.ok) threeNotes.push(`${width}: ${verdict.reason}`);
      else threeProblems.push(`${width}: ${verdict.reason}`);
    }
    results.push(
      threeProblems.length === 0
        ? pass("library-row-three-buttons-geometry", threeNotes.join("; "))
        : fail("library-row-three-buttons-geometry", threeProblems.join(" | ")),
    );

    if (threeStacked.error) {
      results.push(fail("library-row-three-buttons-geometry-stacked", threeStacked.error));
    } else {
      const verdict = judgeLibraryStackedGeometry(threeStacked);
      const width = `stacked panel ${round(threeStacked.panelWidth)}px (window ${threeStacked.windowWidth})`;
      results.push(
        (verdict.ok ? pass : fail)("library-row-three-buttons-geometry-stacked", `${width}: ${verdict.reason}`),
      );
    }
  }

  // 8 — nothing above was paid for with a renderer error
  const introduced = diagnostics.errors().slice(errorsBefore);
  results.push(
    introduced.length === 0
      ? pass("library-settings-renderer-clean", "the settings page walk logged no console error and no pageerror")
      : fail(
          "library-settings-renderer-clean",
          `the walk raised ${introduced.length}: ${introduced.map((entry) => entry.text.slice(0, 160)).join("; ")}`,
        ),
  );

  await clearViewport(client);
  return results;
}
