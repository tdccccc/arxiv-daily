import assert from "node:assert/strict";
import test from "node:test";
import {
  DISCLOSURE_MODAL_CLASS,
  DISCLOSURE_TITLES,
  EXPECTED_GROUP_ORDER,
  judgeDisclosureTitle,
  judgeGroupOrder,
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
  const verdict = judgeLibraryButtons(buttons(["Change folder", "Build index"]), {
    expected: ["Change folder", "Build index"],
  });
  assert.equal(verdict.ok, true);
});

test("a fourth button fails even when the row is not otherwise checked", () => {
  const verdict = judgeLibraryButtons(
    buttons(["Change folder", "Build index", "Revoke", "Preview"]),
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /more than three/);
});

test("any button that says authorize fails, wherever in the section it sits", () => {
  const verdict = judgeLibraryButtons(
    buttons(["Change folder", "Build index"], ["Review & authorize"]),
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
    { text: "Build index", left: 498, right: 590, top: 100 },
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
      { text: "Build index", left: 498, right: 590, top: 140 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /different lines/);
});

test("a row whose buttons stop short of the right edge is not right-aligned", () => {
  const verdict = judgeLibraryGeometry(geometry({
    buttons: [
      { text: "Change folder", left: 400, right: 490, top: 100 },
      { text: "Build index", left: 498, right: 540, top: 100 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /right edge/);
});

test("buttons spilling out of their control box, over the description, fail", () => {
  const verdict = judgeLibraryGeometry(geometry({
    buttons: [
      { text: "Change folder", left: 300, right: 490, top: 100 },
      { text: "Build index", left: 498, right: 590, top: 100 },
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
      { text: "Build index", left: 608, right: 700, top: 100 },
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
    { text: "Build index", left: 398, right: 490, top: 100, width: 92, height: 30 },
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
      { text: "Build index", left: 400, right: 492, top: 140, width: 92, height: 30 },
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
      { text: "Build index", left: 146, right: 238, top: 100, width: 92, height: 30 },
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
      { text: "Build index", left: 400, right: 492, top: 140, width: 92, height: 30 },
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
      { text: "Build index", left: 490, right: 490, top: 100, width: 0, height: 30 },
      { text: "Revoke", left: 498, right: 590, top: 100, width: 92, height: 30 },
    ],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /Build index is laid out 0x30/);
});

test("wrapped buttons that spill over the description still fail", () => {
  const verdict = judgeLibraryWrappedGeometry(threeButtons({
    buttons: [
      { text: "Change folder", left: 180, right: 390, top: 100, width: 210, height: 30 },
      { text: "Build index", left: 398, right: 490, top: 100, width: 92, height: 30 },
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

test("the interaction expressions address elements by their visible text", () => {
  assert.match(selectEmbeddingModeExpression("remote"), /select.value = "remote"/);
  assert.match(selectEmbeddingModeExpression("remote"), /dispatchEvent\(new Event\("change"\)\)/);
  assert.match(clickLibraryRowButtonExpression("Build index"), /"Build index"/);
  assert.match(clickModalButtonExpression("Cancel"), /"Cancel"/);
});

/*
 * The class is spelled out here, not imported from the module under test, so
 * that renaming it is caught rather than agreed with. Its other end — the
 * plugin actually putting it on the dialog root — is guarded by
 * `plugin/tests/library-modal.test.ts`.
 */
test("the disclosure dialog is located by a stable class, never by its heading", () => {
  assert.equal(DISCLOSURE_MODAL_CLASS, "arxiv-daily-library-authorization-modal");
  for (const expression of [modalExpression(), clickModalButtonExpression("Cancel")]) {
    assert.match(expression, /\.modal-container \.modal/);
    assert.match(expression, /"arxiv-daily-library-authorization-modal"/);
    // A heading match in the lookup is what made a reworded title read as a
    // missing dialog; the heading may only be read out, never selected on.
    assert.doesNotMatch(expression, /modal-title[^\n]*===/);
  }
});

test("the reader reports the heading as data so it can be judged separately", () => {
  assert.match(modalExpression(), /title: \(modal\.querySelector\("\.modal-title"\)/);
});

test("each processing depth has its own heading, and neither claims the other's", () => {
  assert.equal(DISCLOSURE_TITLES["full-text"], "Send full text off this device?");
  assert.equal(
    DISCLOSURE_TITLES["metadata-and-abstracts"],
    "Send titles and abstracts off this device?",
  );
  assert.doesNotMatch(DISCLOSURE_TITLES["metadata-and-abstracts"], /full text/i);
});

test("a heading judged at the wrong depth fails as a wording problem, not a missing dialog", () => {
  const fullText = { present: true, title: "Send full text off this device?" };
  assert.equal(judgeDisclosureTitle(fullText, "full-text").ok, true);

  const verdict = judgeDisclosureTitle(fullText, "metadata-and-abstracts");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /titled "Send full text off this device\?", expected/);
  assert.doesNotMatch(verdict.reason, /no .*dialog|not found|no dialog/i);
});

test("the retired heading fails the title judge, quoting both headings", () => {
  const verdict = judgeDisclosureTitle(
    { present: true, title: "Authorize personal library" },
    "full-text",
  );
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /titled "Authorize personal library"/);
  assert.match(verdict.reason, /expected "Send full text off this device\?"/);
});

test("a depth with no specified heading is a failure rather than a silent pass", () => {
  const verdict = judgeDisclosureTitle({ present: true, title: "anything" }, "invented-depth");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no heading is specified/);
});
