import assert from "node:assert/strict";
import test from "node:test";
import {
  DISCLOSURE_CANCEL_BUTTON_CLASS,
  DISCLOSURE_CANCEL_LABEL,
  DISCLOSURE_CONFIRM_BUTTON_CLASS,
  DISCLOSURE_CONFIRM_LABELS,
  DISCLOSURE_MODAL_CLASS,
  DISCLOSURE_TITLES,
  EXPECTED_GROUP_ORDER,
  judgeDisclosureButtons,
  judgeDisclosureTitle,
  judgeCancellingRow,
  judgeGroupOrder,
  judgeInPlaceProgressUpdate,
  judgeIndexTraceSentence,
  judgeIndexingRow,
  judgeLibraryButtons,
  judgeLibraryGeometry,
  judgeLibraryWrappedGeometry,
  judgeDescriptionReadable,
  judgeUnchanged,
  clickLibraryRowButtonExpression,
  clickModalButtonExpression,
  modalExpression,
  selectEmbeddingModeExpression,
} from "../desktop-acceptance/library-settings.mjs";

const headings = (...between) => [
  "AI model",
  "arXiv",
  ...between,
  "Advanced",
];

test("the three groups have to be adjacent, in order", () => {
  const verdict = judgeGroupOrder(headings(...EXPECTED_GROUP_ORDER));
  assert.equal(verdict.ok, true);
});

test("another group wedged between them is a failure, not a pass", () => {
  const verdict = judgeGroupOrder(
    headings("Output & schedule", "Personal library", "Something else", "Email delivery"),
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /not adjacent/);
});

test("the old order — library before output — fails", () => {
  const verdict = judgeGroupOrder(
    headings("Personal library", "Output & schedule", "Email delivery"),
  );
  assert.equal(verdict.ok, false);
});

test("a missing group is named", () => {
  const verdict = judgeGroupOrder(headings("Output & schedule", "Email delivery"));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /Personal library/);
});

const buttons = (rowButtons, extra = []) => ({
  rowButtons,
  groupButtons: [...rowButtons, ...extra],
});

test("the expected two-button row passes", () => {
  const verdict = judgeLibraryButtons(buttons(["Change folder", "Retry preparation"]), {
    expected: ["Change folder", "Retry preparation"],
  });
  assert.equal(verdict.ok, true);
});

const preparedLibrary = (overrides = {}) => ({
  ...buttons(["Change folder"], ["Review suggestions"]),
  selected: true, searchablePapers: 12, preparing: false,
  legacySuggestionRows: 0,
  reviewButtons: [{ text: "Review suggestions", disabled: false }],
  ...overrides,
});

test("a prepared library offers review without a permanent rebuild button", () => {
  assert.equal(judgeLibraryButtons(preparedLibrary()).ok, true);
  const retired = judgeLibraryButtons(preparedLibrary({ ...buttons(["Change folder", "Build index"], ["Review suggestions"]) }));
  assert.equal(retired.ok, false);
  assert.match(retired.reason, /Build index/);
  assert.equal(judgeLibraryButtons(preparedLibrary({ ...buttons(["Change folder", "Retry preparation"], ["Review suggestions"]) })).ok, false);
});

test("an unprepared library offers retry and keeps review disabled", () => {
  const state = preparedLibrary({
    ...buttons(["Change folder", "Retry preparation"], ["Review suggestions"]),
    searchablePapers: 0, reviewButtons: [{ text: "Review suggestions", disabled: true }],
  });
  assert.equal(judgeLibraryButtons(state).ok, true);
  assert.equal(judgeLibraryButtons({ ...state, reviewButtons: [{ text: "Review suggestions", disabled: false }] }).ok, false);
  assert.equal(judgeLibraryButtons({ ...state, ...buttons(["Change folder"], ["Review suggestions"]) }).ok, false);
});

test("a preparation error requires retry and disables review despite old searchable papers", () => {
  const state = preparedLibrary({
    ...buttons(["Change folder", "Retry preparation"], ["Review suggestions"]),
    preparationError: "The embedding model changed.",
    reviewButtons: [{ text: "Review suggestions", disabled: true }],
  });
  assert.equal(judgeLibraryButtons(state).ok, true);
  assert.equal(judgeLibraryButtons({ ...state, reviewButtons: [{ text: "Review suggestions", disabled: false }] }).ok, false);
  assert.equal(judgeLibraryButtons({ ...state, ...buttons(["Change folder"], ["Review suggestions"]) }).ok, false);
});

test("preparation keeps Cancel and disables review even with an older index", () => {
  const state = preparedLibrary({
    ...buttons(["Change folder", "Indexing…", "Cancel"], ["Review suggestions"]),
    preparing: true, reviewButtons: [{ text: "Review suggestions", disabled: true }],
  });
  assert.equal(judgeLibraryButtons(state).ok, true);
  assert.equal(judgeLibraryButtons({ ...state, reviewButtons: [{ text: "Review suggestions", disabled: false }] }).ok, false);
  assert.equal(judgeLibraryButtons({ ...state, ...buttons(["Change folder", "Indexing…"], ["Review suggestions"]) }).ok, false);
});

test("a duplicate library shortcut or missing review action fails", () => {
  assert.equal(judgeLibraryButtons(preparedLibrary({ legacySuggestionRows: 1 })).ok, false);
  assert.equal(judgeLibraryButtons(preparedLibrary({ reviewButtons: [] })).ok, false);
});

test("a fourth button fails even when the row is not otherwise checked", () => {
  const verdict = judgeLibraryButtons(
    buttons(["Change folder", "Retry preparation", "Revoke", "Preview"]),
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /more than three/);
});

test("any button that says authorize fails, wherever in the section it sits", () => {
  const verdict = judgeLibraryButtons(
    buttons(["Change folder", "Retry preparation"], ["Review & authorize"]),
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /authorization button/);
});

test("the Manage menu coming back fails", () => {
  const verdict = judgeLibraryButtons(buttons(["Change folder", "Manage"]));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /Manage/);
});

const geometry = (overrides = {}) => ({
  row: { left: 0, right: 600 },
  info: { left: 0, right: 380 },
  control: { left: 400, right: 590, scrollWidth: 190, clientWidth: 190 },
  buttons: [
    { text: "Change folder", left: 400, right: 490, top: 100 },
    { text: "Retry preparation", left: 498, right: 590, top: 100 },
  ],
  ...overrides,
});

test("buttons on one line, flush right, inside their control, pass", () => {
  const verdict = judgeLibraryGeometry(geometry());
  assert.equal(verdict.ok, true, verdict.reason);
});

test("a wrapped button is caught by its top, which happy-dom cannot produce", () => {
  const verdict = judgeLibraryGeometry(geometry({
    buttons: [
      { text: "Change folder", left: 400, right: 490, top: 100 },
      { text: "Retry preparation", left: 498, right: 590, top: 140 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /different lines/);
});

test("a row whose buttons stop short of the right edge is not right-aligned", () => {
  const verdict = judgeLibraryGeometry(geometry({
    buttons: [
      { text: "Change folder", left: 400, right: 490, top: 100 },
      { text: "Retry preparation", left: 498, right: 540, top: 100 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /right edge/);
});

test("buttons spilling out of their control box, over the description, fail", () => {
  const verdict = judgeLibraryGeometry(geometry({
    buttons: [
      { text: "Change folder", left: 300, right: 490, top: 100 },
      { text: "Retry preparation", left: 498, right: 590, top: 100 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /overflow it by 100px on the left/);
  assert.match(verdict.reason, /cover the description/);
});

test("buttons leaving the row entirely fail", () => {
  const verdict = judgeLibraryGeometry(geometry({
    control: { left: 400, right: 700, scrollWidth: 300, clientWidth: 300 },
    buttons: [
      { text: "Change folder", left: 400, right: 600, top: 100 },
      { text: "Retry preparation", left: 608, right: 700, top: 100 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /outside the row/);
});

const description = (overrides = {}) => ({
  left: 0,
  right: 220,
  top: 110,
  bottom: 150,
  width: 220,
  height: 40,
  contentWidth: 220,
  characters: 78,
  lines: 2,
  widestLine: 218,
  text: "Choose a folder of PDFs and build the index to search it.",
  ...overrides,
});

const threeButtons = (overrides = {}) => ({
  row: { left: 0, right: 600 },
  info: { left: 0, right: 260 },
  control: { left: 280, right: 590, scrollWidth: 310, clientWidth: 310 },
  description: description(),
  buttons: [
    { text: "Change folder", left: 280, right: 390, top: 100, width: 110, height: 30 },
    { text: "Retry preparation", left: 398, right: 490, top: 100, width: 92, height: 30 },
    { text: "Revoke", left: 498, right: 590, top: 100, width: 92, height: 30 },
  ],
  ...overrides,
});

test("three buttons on one line, right-aligned, beside a readable description, pass", () => {
  const verdict = judgeLibraryWrappedGeometry(threeButtons());
  assert.equal(verdict.ok, true, verdict.reason);
});

test("three buttons wrapped onto two right-aligned lines still pass — wrapping is the point", () => {
  const verdict = judgeLibraryWrappedGeometry(threeButtons({
    control: { left: 400, right: 590, scrollWidth: 190, clientWidth: 190 },
    buttons: [
      { text: "Change folder", left: 480, right: 590, top: 100, width: 110, height: 30 },
      { text: "Retry preparation", left: 400, right: 492, top: 140, width: 92, height: 30 },
      { text: "Revoke", left: 500, right: 590, top: 140, width: 90, height: 30 },
    ],
  }));
  assert.equal(verdict.ok, true, verdict.reason);
});

test("the failing state — nothing overlaps, the description is a column of letters", () => {
  const verdict = judgeLibraryWrappedGeometry(threeButtons({
    info: { left: 0, right: 8 },
    control: { left: 28, right: 330, scrollWidth: 302, clientWidth: 302 },
    description: description({ left: 0, right: 6, width: 6, contentWidth: 6, lines: 57, widestLine: 6 }),
    buttons: [
      { text: "Change folder", left: 28, right: 138, top: 100, width: 110, height: 30 },
      { text: "Retry preparation", left: 146, right: 238, top: 100, width: 92, height: 30 },
      { text: "Revoke", left: 246, right: 330, top: 100, width: 84, height: 30 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /description column is 6px wide/);
  assert.match(verdict.reason, /characters a line/);
});

test("a wide description column that fits one word a line is still not readable", () => {
  const verdict = judgeDescriptionReadable(description({ lines: 12 }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /6.5 characters a line/);
});

test("a sliver of a column fails on width even when its two lines look fine", () => {
  const verdict = judgeDescriptionReadable(description({ contentWidth: 90, characters: 40, lines: 2 }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /90px wide/);
  assert.doesNotMatch(verdict.reason, /characters a line/);
});

test("a wrapped line that stops short of the right edge is not right-aligned", () => {
  const verdict = judgeLibraryWrappedGeometry(threeButtons({
    control: { left: 400, right: 590, scrollWidth: 190, clientWidth: 190 },
    buttons: [
      { text: "Change folder", left: 440, right: 550, top: 100, width: 110, height: 30 },
      { text: "Retry preparation", left: 400, right: 492, top: 140, width: 92, height: 30 },
      { text: "Revoke", left: 500, right: 590, top: 140, width: 90, height: 30 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /do not end at the control's right edge/);
});

test("the main call to action collapsing to nothing fails even with everything else in place", () => {
  const verdict = judgeLibraryWrappedGeometry(threeButtons({
    buttons: [
      { text: "Change folder", left: 280, right: 390, top: 100, width: 110, height: 30 },
      { text: "Retry preparation", left: 490, right: 490, top: 100, width: 0, height: 30 },
      { text: "Revoke", left: 498, right: 590, top: 100, width: 92, height: 30 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /Retry preparation is laid out 0x30/);
});

test("wrapped buttons that spill over the description still fail", () => {
  const verdict = judgeLibraryWrappedGeometry(threeButtons({
    buttons: [
      { text: "Change folder", left: 180, right: 390, top: 100, width: 210, height: 30 },
      { text: "Retry preparation", left: 398, right: 490, top: 100, width: 92, height: 30 },
      { text: "Revoke", left: 498, right: 590, top: 100, width: 92, height: 30 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /cover the description/);
});

test("a declined disclosure has to leave the mode and the grant exactly as they were", () => {
  const before = { embedding: { mode: "local", baseUrl: "x" }, status: { kind: "authorization-required" } };
  assert.equal(judgeUnchanged(before, before).ok, true);
  assert.match(
    judgeUnchanged(before, { ...before, embedding: { mode: "remote", baseUrl: "x" } }).reason,
    /local → remote/,
  );
  assert.match(
    judgeUnchanged(before, { ...before, status: { kind: "authorized" } }).reason,
    /authorization-required → authorized/,
  );
  assert.match(
    judgeUnchanged(before, { ...before, embedding: { mode: "local", baseUrl: "y" } }).reason,
    /embedding settings changed/,
  );
});

test("the settings-page interaction expressions address elements by their visible text", () => {
  assert.match(selectEmbeddingModeExpression("remote"), /select.value = "remote"/);
  assert.match(selectEmbeddingModeExpression("remote"), /dispatchEvent\(new Event\("change"\)\)/);
  assert.match(clickLibraryRowButtonExpression("Retry preparation"), /"Retry preparation"/);
});

/*
 * The class is spelled out here, not imported from the module under test, so
 * that renaming it is caught rather than agreed with. Its other end — the
 * plugin actually putting it on the dialog root — is guarded by
 * `plugin/tests/library-modal.test.ts`.
 */
test("the disclosure dialog is located by a stable class, never by its heading", () => {
  assert.equal(DISCLOSURE_MODAL_CLASS, "arxiv-daily-library-authorization-modal");
  for (const expression of [
    modalExpression(),
    clickModalButtonExpression(DISCLOSURE_CANCEL_BUTTON_CLASS),
  ]) {
    assert.match(expression, /\.modal-container \.modal/);
    assert.match(expression, /"arxiv-daily-library-authorization-modal"/);
    // A heading match in the lookup is what made a reworded title read as a
    // missing dialog; the heading may only be read out, never selected on.
    assert.doesNotMatch(expression, /modal-title[^\n]*===/);
  }
});

/*
 * Same rule, one level down. The class literals are spelled out here rather
 * than imported from the module under test, so renaming one is caught instead
 * of agreed with; the plugin actually marking the buttons is guarded by
 * `plugin/tests/library-modal.test.ts`.
 */
test("the disclosure's answers are clicked by a stable mark, never by their label", () => {
  assert.equal(DISCLOSURE_CONFIRM_BUTTON_CLASS, "arxiv-daily-library-authorization-confirm");
  assert.equal(DISCLOSURE_CANCEL_BUTTON_CLASS, "arxiv-daily-library-authorization-cancel");

  const confirmClick = clickModalButtonExpression(DISCLOSURE_CONFIRM_BUTTON_CLASS);
  assert.match(confirmClick, /querySelector\("button\." \+ "arxiv-daily-library-authorization-confirm"\)/);
  // Selecting on the label is what would make a reworded button read as a
  // missing one — the failure this whole arrangement exists to prevent.
  assert.doesNotMatch(confirmClick, /textContent[^\n]*===/);
  assert.doesNotMatch(confirmClick, /"Authorize"|"Send full text"/);

  const cancelClick = clickModalButtonExpression(DISCLOSURE_CANCEL_BUTTON_CLASS);
  assert.match(cancelClick, /querySelector\("button\." \+ "arxiv-daily-library-authorization-cancel"\)/);
  assert.doesNotMatch(cancelClick, /textContent[^\n]*===/);
});

test("the reader reports the heading and both labels as data, to be judged separately", () => {
  const expression = modalExpression();
  assert.match(expression, /title: \(modal\.querySelector\("\.modal-title"\)/);
  assert.match(expression, /confirm: marked\("arxiv-daily-library-authorization-confirm"\)/);
  assert.match(expression, /cancel: marked\("arxiv-daily-library-authorization-cancel"\)/);
});

test("both legacy processing depths describe the current title-and-abstract scope", () => {
  for (const depth of ["full-text", "metadata-and-abstracts"]) {
    assert.equal(DISCLOSURE_TITLES[depth], "Send titles and abstracts off this device?");
    assert.equal(judgeDisclosureTitle({ present: true, title: "Send titles and abstracts off this device?" }, depth).ok, true);
    assert.equal(judgeDisclosureButtons({ present: true, confirm: "Send titles and abstracts", cancel: "Cancel" }, depth).ok, true);
  }
});

test("the retired full-text heading fails at either legacy depth as a wording problem", () => {
  const fullText = { present: true, title: "Send full text off this device?" };
  for (const depth of ["full-text", "metadata-and-abstracts"]) {
    const verdict = judgeDisclosureTitle(fullText, depth);
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /titled "Send full text off this device\?", expected/);
    assert.doesNotMatch(verdict.reason, /no .*dialog|not found|no dialog/i);
  }
});

test("the retired heading fails the title judge, quoting both headings", () => {
  const verdict = judgeDisclosureTitle(
    { present: true, title: "Authorize personal library" },
    "full-text",
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /titled "Authorize personal library"/);
  assert.match(verdict.reason, /expected "Send titles and abstracts off this device\?"/);
});

test("a depth with no specified heading is a failure rather than a silent pass", () => {
  const verdict = judgeDisclosureTitle({ present: true, title: "anything" }, "invented-depth");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no heading is specified/);
});

const dialog = (depth) => ({
  present: true,
  title: DISCLOSURE_TITLES[depth],
  confirm: DISCLOSURE_CONFIRM_LABELS[depth],
  cancel: DISCLOSURE_CANCEL_LABEL,
});

test("the confirm button answers the heading in the heading's own words", () => {
  for (const depth of Object.keys(DISCLOSURE_TITLES)) {
    assert.equal(
      DISCLOSURE_TITLES[depth],
      `${DISCLOSURE_CONFIRM_LABELS[depth]} off this device?`,
      `the ${depth} heading and confirm button do not speak of the same scope`,
    );
    assert.equal(judgeDisclosureButtons(dialog(depth), depth).ok, true);
  }
  assert.doesNotMatch(DISCLOSURE_CONFIRM_LABELS["metadata-and-abstracts"], /full text/i);
  for (const label of Object.values(DISCLOSURE_CONFIRM_LABELS)) {
    assert.doesNotMatch(label, /authoriz/i);
  }
});

test("the retired Authorize label fails as wording, not as a missing button", () => {
  const verdict = judgeDisclosureButtons(
    { ...dialog("full-text"), confirm: "Authorize" },
    "full-text",
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /the confirm button reads "Authorize", expected "Send titles and abstracts"/);
  assert.doesNotMatch(verdict.reason, /has no|not found|missing/i);
});

test("the retired full-text confirmation label fails at either legacy depth", () => {
  for (const depth of ["full-text", "metadata-and-abstracts"]) {
    const verdict = judgeDisclosureButtons({ ...dialog(depth), confirm: "Send full text" }, depth);
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /reads "Send full text", expected "Send titles and abstracts"/);
  }
});

test("a genuinely missing button says so, distinctly from a reworded one", () => {
  const verdict = judgeDisclosureButtons({ ...dialog("full-text"), confirm: null }, "full-text");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /has no \.arxiv-daily-library-authorization-confirm button/);
  assert.doesNotMatch(verdict.reason, /reads/);
});

test("the way out keeps its own label, and its absence is reported", () => {
  const reworded = judgeDisclosureButtons(
    { ...dialog("full-text"), cancel: "Nope" },
    "full-text",
  );
  assert.equal(reworded.ok, false);
  assert.match(reworded.reason, /the cancel button reads "Nope", expected "Cancel"/);

  const missing = judgeDisclosureButtons({ ...dialog("full-text"), cancel: null }, "full-text");
  assert.equal(missing.ok, false);
  assert.match(missing.reason, /has no \.arxiv-daily-library-authorization-cancel button/);
});

test("a depth with no specified confirm label is a failure rather than a silent pass", () => {
  const verdict = judgeDisclosureButtons(dialog("full-text"), "invented-depth");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no confirm label is specified/);
});

// ── the row while a run is in flight ────────────────────────────────────────

const RUNNING_LABEL = "Indexing… (12/40)";

const runningRow = (overrides = {}) => ({
  rowButtons: ["Change folder", RUNNING_LABEL, "Cancel"],
  rowButtonStates: [
    { text: "Change folder", disabled: true, mark: "mark-0" },
    { text: RUNNING_LABEL, disabled: true, mark: "mark-1" },
    { text: "Cancel", disabled: false, mark: "mark-2" },
  ],
  rowDescription: "Indexing papers — extracting and embedding PDF text. Nothing is saved until the run finishes, so cancelling discards it.",
  groupButtons: ["Change folder", RUNNING_LABEL, "Cancel"],
  ...overrides,
});

const runningOptions = {
  expectedLabel: RUNNING_LABEL,
  phase: "extracting and embedding PDF text",
};

test("the row carrying a run passes", () => {
  const verdict = judgeIndexingRow(runningRow(), runningOptions);
  assert.equal(verdict.ok, true);
});

test("a run whose main button can still be pressed fails", () => {
  const states = runningRow().rowButtonStates.map((button) =>
    button.text === RUNNING_LABEL ? { ...button, disabled: false } : button);
  const verdict = judgeIndexingRow(runningRow({ rowButtonStates: states }), runningOptions);
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /offers to start a second run/);
});

test("a run that still lets the folder move fails", () => {
  const states = runningRow().rowButtonStates.map((button) =>
    button.text === "Change folder" ? { ...button, disabled: false } : button);
  const verdict = judgeIndexingRow(runningRow({ rowButtonStates: states }), runningOptions);
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /out from under the run/);
});

test("a run with no way to stop it fails", () => {
  const states = runningRow().rowButtonStates.filter((button) => button.text !== "Cancel");
  const verdict = judgeIndexingRow(
    runningRow({ rowButtonStates: states, rowButtons: states.map((b) => b.text) }),
    runningOptions,
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no way to stop the run/);
});

test("a run the description never names fails", () => {
  const verdict = judgeIndexingRow(
    runningRow({ rowDescription: "Selected: papers. Build the search index." }),
    runningOptions,
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /does not say what the run is doing/);
});

test("a progress step that kept the same buttons and changed their text passes", () => {
  const before = runningRow();
  const after = runningRow({
    rowButtonStates: before.rowButtonStates.map((button) =>
      button.text === RUNNING_LABEL ? { ...button, text: "Indexing… (13/40)" } : button),
  });
  const verdict = judgeInPlaceProgressUpdate(before, after, { expectedLabel: "Indexing… (13/40)" });
  assert.equal(verdict.ok, true);
});

test("a progress step that replaced the buttons is a re-render, and fails", () => {
  const before = runningRow();
  const after = runningRow({
    rowButtonStates: before.rowButtonStates.map((button) => ({
      ...button,
      mark: null,
      ...(button.text === RUNNING_LABEL ? { text: "Indexing… (13/40)" } : {}),
    })),
  });
  const verdict = judgeInPlaceProgressUpdate(before, after, { expectedLabel: "Indexing… (13/40)" });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /re-rendered instead of updating the row/);
});

test("buttons that survived but never advanced their count fail", () => {
  const before = runningRow();
  const verdict = judgeInPlaceProgressUpdate(before, before, { expectedLabel: "Indexing… (13/40)" });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /expected "Indexing… \(13\/40\)"/);
});

const stoppingRow = (overrides = {}) => runningRow({
  rowButtonStates: [
    { text: "Change folder", disabled: true, mark: "mark-0" },
    { text: RUNNING_LABEL, disabled: true, mark: "mark-1" },
    { text: "Cancelling…", disabled: true, mark: "mark-2" },
  ],
  ...overrides,
});

const cancelled = [{ id: "personal-library-fulltext-index:1", cancellationRequested: true }];
const stillRunning = [{ id: "personal-library-fulltext-index:1", cancellationRequested: false }];

test("a pressed Cancel that says it is stopping, and stopped something, passes", () => {
  const verdict = judgeCancellingRow(stoppingRow(), cancelled);
  assert.equal(verdict.ok, true);
});

test("a Cancel that changed the row but asked no run to stop fails", () => {
  const verdict = judgeCancellingRow(stoppingRow(), stillRunning);
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /asked no operation to stop/);
});

test("a Cancel that stopped the run but still invites a second press fails", () => {
  const verdict = judgeCancellingRow(runningRow(), cancelled);
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /still reads "Cancel"/);
});

// ── what a finished run leaves on the row ───────────────────────────────────

const idleRow = (rowDescription) => ({
  rowButtons: ["Change folder", "Retry preparation"],
  rowButtonStates: [
    { text: "Change folder", disabled: false, mark: null },
    { text: "Retry preparation", disabled: false, mark: null },
  ],
  rowDescription,
  groupButtons: ["Change folder", "Retry preparation"],
});

test("a row naming both when and how much passes", () => {
  const verdict = judgeIndexTraceSentence(
    idleRow("Selected: papers. Local embedding stays on this device. Last indexed 2026-08-30 21:15 · 128 papers searchable."),
    { papers: 128 },
  );
  assert.equal(verdict.ok, true);
});

test("a count with no time fails: it does not answer whether last night's run finished", () => {
  const verdict = judgeIndexTraceSentence(idleRow("Selected: papers. 128 papers searchable."), { papers: 128 });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /does not say when/);
});

test("a time with no count fails: it does not say what can be searched", () => {
  const verdict = judgeIndexTraceSentence(
    idleRow("Selected: papers. Last indexed 2026-08-30 21:15."),
    { papers: 128 },
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /128 papers searchable/);
});

test("a row that says nothing about past runs fails once one has happened", () => {
  const verdict = judgeIndexTraceSentence(
    idleRow("Selected: papers. Build the search index to search these PDFs."),
    { papers: 3 },
  );
  assert.equal(verdict.ok, false);
});
