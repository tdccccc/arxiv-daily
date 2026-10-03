import assert from "node:assert/strict";
import test from "node:test";
import {
  LONG_DIRECTION,
  judgeAcceptBarBelowList,
  judgeDirectionNotBold,
  judgeDirectionVisible,
  judgeDirectionWraps,
  judgeLibraryOverviewRoundTrip,
  judgeNoTabBar,
  judgeNothingNewDone,
  judgeReadFailurePrimary,
  judgeStaleCoveragePrimary,
} from "../desktop-acceptance/direction-review.mjs";

const page = (overrides = {}) => ({
  present: true,
  heading: "Topics from your library",
  stateTitle: "",
  stateMessage: "",
  hasBackToReview: false,
  optionsButtons: [],
  stateButtons: [],
  generateButtons: [],
  topicCount: 0,
  hasAcceptBar: false,
  acceptBarAfterTopics: null,
  ...overrides,
});

// ── (a) stale coverage ──────────────────────────────────────────────────────

test("stale coverage: Update suggestions as the only action passes", () => {
  const verdict = judgeStaleCoveragePrimary(page({
    stateTitle: "This review needs an update",
    stateButtons: [{ text: "Update suggestions", disabled: false, modCta: true }],
  }));
  assert.equal(verdict.ok, true);
});

test("stale coverage: a disabled primary action fails", () => {
  const verdict = judgeStaleCoveragePrimary(page({
    stateTitle: "This review needs an update",
    stateButtons: [{ text: "Update suggestions", disabled: true, modCta: true }],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /disabled/);
});

test("stale coverage: a Back to review control means this is not the single default page", () => {
  const verdict = judgeStaleCoveragePrimary(page({
    stateTitle: "This review needs an update",
    stateButtons: [{ text: "Update suggestions", disabled: false, modCta: true }],
    hasBackToReview: true,
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /single default page/);
});

test("stale coverage: an accept bar with nothing to review fails", () => {
  const verdict = judgeStaleCoveragePrimary(page({
    stateTitle: "This review needs an update",
    stateButtons: [{ text: "Update suggestions", disabled: false, modCta: true }],
    hasAcceptBar: true,
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /accept topics bar|Add to research topics bar/);
});

test("the review modal not opening is reported, not crashed on", () => {
  const verdict = judgeStaleCoveragePrimary({ present: false });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /did not open/);
});

// ── (b) read failure ────────────────────────────────────────────────────────

test("read failure: Try again primary plus a disabled, explained fallback passes", () => {
  const verdict = judgeReadFailurePrimary(page({
    stateTitle: "Couldn't open your suggestions",
    stateButtons: [{ text: "Try again", disabled: false, modCta: true }],
    generateButtons: [{
      text: "Generate topics", disabled: true, title: "The saved suggestions could not be loaded.", insideOptions: true, insideState: false,
    }],
  }));
  assert.equal(verdict.ok, true);
});

test("read failure: a generation fallback that is not disabled fails", () => {
  const verdict = judgeReadFailurePrimary(page({
    stateTitle: "Couldn't open your suggestions",
    stateButtons: [{ text: "Try again", disabled: false, modCta: true }],
    generateButtons: [{ text: "Generate topics", disabled: false, title: "", insideOptions: true, insideState: false }],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /not disabled/);
});

test("read failure: a disabled fallback with no reason fails", () => {
  const verdict = judgeReadFailurePrimary(page({
    stateTitle: "Couldn't open your suggestions",
    stateButtons: [{ text: "Try again", disabled: false, modCta: true }],
    generateButtons: [{ text: "Generate topics", disabled: true, title: "", insideOptions: true, insideState: false }],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no explanatory reason/);
});

test("read failure: no fallback at all in More options fails", () => {
  const verdict = judgeReadFailurePrimary(page({
    stateTitle: "Couldn't open your suggestions",
    stateButtons: [{ text: "Try again", disabled: false, modCta: true }],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no generation fallback/);
});

// ── (c) nothing new needed ──────────────────────────────────────────────────

test("nothing new: Done as the only action passes", () => {
  const verdict = judgeNothingNewDone(page({
    stateTitle: "No new directions to add",
    stateButtons: [{ text: "Done", disabled: false, modCta: true }],
  }));
  assert.equal(verdict.ok, true);
});

test("nothing new: a different primary label fails and names it", () => {
  const verdict = judgeNothingNewDone(page({
    stateTitle: "No new directions to add",
    stateButtons: [{ text: "Generate again", disabled: false, modCta: true }],
  }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /Generate again/);
});

// ── accept bar position and the absence of a tab bar ────────────────────────

test("an accept bar after the last topic passes", () => {
  const verdict = judgeAcceptBarBelowList(page({ topicCount: 2, hasAcceptBar: true, acceptBarAfterTopics: true }));
  assert.equal(verdict.ok, true);
});

test("an accept bar positioned before the topics fails", () => {
  const verdict = judgeAcceptBarBelowList(page({ topicCount: 2, hasAcceptBar: true, acceptBarAfterTopics: false }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /not positioned after/);
});

test("no direction topics on the page fails, rather than vacuously passing", () => {
  const verdict = judgeAcceptBarBelowList(page({ topicCount: 0, hasAcceptBar: false, acceptBarAfterTopics: null }));
  assert.equal(verdict.ok, false);
});

test("no Back to review control on the default page passes the single-page check", () => {
  const verdict = judgeNoTabBar(page({ hasBackToReview: false }));
  assert.equal(verdict.ok, true);
});

test("a visible Back to review control on the default page fails", () => {
  const verdict = judgeNoTabBar(page({ hasBackToReview: true }));
  assert.equal(verdict.ok, false);
});

// ── (d) More options → Library overview → Back to review ───────────────────

test("a full round trip through Library overview passes", () => {
  const before = page({ heading: "Topics from your library", hasBackToReview: false });
  const overview = page({ heading: "Library overview", hasBackToReview: true });
  const back = page({ heading: "Topics from your library", hasBackToReview: false });
  const verdict = judgeLibraryOverviewRoundTrip(before, overview, back);
  assert.equal(verdict.ok, true);
});

test("Library overview without its own Back to review control fails", () => {
  const before = page({ heading: "Topics from your library" });
  const overview = page({ heading: "Library overview", hasBackToReview: false });
  const back = page({ heading: "Topics from your library" });
  const verdict = judgeLibraryOverviewRoundTrip(before, overview, back);
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no Back to review/);
});

test("failing to return to the original heading fails", () => {
  const before = page({ heading: "Topics from your library" });
  const overview = page({ heading: "Library overview", hasBackToReview: true });
  const back = page({ heading: "Library overview" });
  const verdict = judgeLibraryOverviewRoundTrip(before, overview, back);
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /Back to review the heading reads/);
});

// ── (e) long direction text: unbold and wrapped ─────────────────────────────

test("a normal-weight heading passes", () => {
  const verdict = judgeDirectionNotBold({ fontWeight: "400" });
  assert.equal(verdict.ok, true);
});

test("a bold numeric font-weight fails", () => {
  const verdict = judgeDirectionNotBold({ fontWeight: "700" });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /bold/);
});

test("a bold keyword font-weight fails", () => {
  const verdict = judgeDirectionNotBold({ fontWeight: "bold" });
  assert.equal(verdict.ok, false);
});

test("a lookup failure is reported, not silently passed", () => {
  const verdict = judgeDirectionNotBold({ error: `no direction heading reads ${JSON.stringify(LONG_DIRECTION)}` });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no direction heading reads/);
});

// A realistic, passing measurement: two line boxes, comfortably inside the
// dialog's own content box. Each test below overrides exactly the one fact
// it means to break.
const heading = (overrides = {}) => ({
  fontWeight: "400",
  lineHeight: 20,
  width: 300,
  height: 44,
  right: 500,
  lineBoxCount: 2,
  contentRight: 600,
  contentScrollWidth: 600,
  contentClientWidth: 600,
  ...overrides,
});

// ── judgeDirectionVisible ────────────────────────────────────────────────────

test("a heading with a real size is visible", () => {
  const verdict = judgeDirectionVisible(heading());
  assert.equal(verdict.ok, true);
});

test("a zero-size heading is not visible — this is the vacuous-pass guard", () => {
  const verdict = judgeDirectionVisible(heading({ width: 0, height: 0 }));
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /hidden|collapsed/);
});

test("a zero-height heading (e.g. display: none) is not visible", () => {
  const verdict = judgeDirectionVisible(heading({ height: 0 }));
  assert.equal(verdict.ok, false);
});

test("a lookup failure is reported, not silently passed as visible", () => {
  const verdict = judgeDirectionVisible({ error: `no direction heading reads ${JSON.stringify(LONG_DIRECTION)}` });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no direction heading reads/);
});

// ── judgeDirectionWraps ──────────────────────────────────────────────────────

test("text with two line boxes, inside the dialog, passes", () => {
  const verdict = judgeDirectionWraps(heading(), "wide");
  assert.equal(verdict.ok, true);
});

test("a hidden or collapsed heading fails — the check never passes vacuously", () => {
  // This is exactly the bug a closed <details> produced: an inline element
  // measured as 0x0 used to satisfy `scrollWidth > clientWidth` trivially.
  const verdict = judgeDirectionWraps(heading({ width: 0, height: 0, lineBoxCount: 0 }), "wide");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /hidden|collapsed/);
});

test("a single line box fails, even though nothing overflows", () => {
  const verdict = judgeDirectionWraps(heading({ lineBoxCount: 1, height: 20 }), "wide");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /single line/);
});

test("a single rendered line box but a height of 1.5+ line-heights still counts as wrapped", () => {
  // A fallback for themes that ever report one giant line box per wrapped
  // field instead of one per visual line.
  const verdict = judgeDirectionWraps(heading({ lineBoxCount: 1, height: 35, lineHeight: 20 }), "wide");
  assert.equal(verdict.ok, true);
});

test("the heading's right edge past the dialog's content edge fails and names the widths", () => {
  const verdict = judgeDirectionWraps(heading({ right: 650, contentRight: 600 }), "narrow");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /narrow/);
  assert.match(verdict.reason, /right edge/);
});

test("the dialog's own content overflowing horizontally fails", () => {
  const verdict = judgeDirectionWraps(heading({ contentScrollWidth: 650, contentClientWidth: 600 }), "wide");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /overflows horizontally/);
});

test("a lookup failure is reported, not silently passed as wrapped", () => {
  const verdict = judgeDirectionWraps({ error: `no direction heading reads ${JSON.stringify(LONG_DIRECTION)}` }, "wide");
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no direction heading reads/);
});
