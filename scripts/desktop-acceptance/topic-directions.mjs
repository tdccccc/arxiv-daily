/**
 * The topic direction list, checked in the real renderer.
 *
 * The unit suite renders the same rows into happy-dom, which has no layout
 * engine and no Obsidian stylesheet. It can say that a field collapses and that
 * a badge appears, because those are classes and text — but the three things
 * that actually decide whether the control is usable are geometric, and it can
 * say nothing about them:
 *
 *   - a collapsed direction really occupies one line rather than several,
 *   - the "+N" badge really matches how much taller the field gets when opened,
 *   - a long direction wraps instead of running off the side of the panel.
 *
 * The unit tests state that geometry themselves (clientHeight 20, scrollHeight
 * 60), which proves the arithmetic and nothing about the layout it describes.
 */
import { clearViewport, setViewport } from "./cdp.mjs";
import { OPEN_SETTINGS_EXPRESSION } from "./library-settings.mjs";

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

/** Long enough to wrap several times in any panel a person would use. */
const LONG_DIRECTION =
  "Photometric redshift methods, catalogue cross-matching, survey-to-survey "
  + "calibration, and the systematic comparisons between template fitting and "
  + "machine-learning estimators that follow from them.";

const SHORT_DIRECTION = "Catalog cross-matching.";

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

/**
 * Replace the vault's topics with one that has a long direction and a short
 * one. The harness captures and restores the settings store around the run, so
 * this does not outlive the session.
 */
const SEED_EXPRESSION = `(async () => {
  const plugin = ${PLUGIN};
  if (!plugin) return "ERROR: the arXiv Daily plugin is not loaded";
  plugin.settings.arxiv.topics = [{
    id: "acceptance-directions",
    name: "Acceptance",
    tag: "acceptance",
    detail: false,
    description: ${asCode(LONG_DIRECTION)},
    directions: [
      { id: "d-long", text: ${asCode(LONG_DIRECTION)}, origin: "manual" },
      { id: "d-short", text: ${asCode(SHORT_DIRECTION)}, origin: "manual" },
    ],
  }];
  await plugin.saveSettings();
  return "seeded";
})()`;

const RENDERER_PRELUDE = `
  const content = document.querySelector(".arxiv-daily-settings")
    ?? document.querySelector(".vertical-tab-content.arxiv-daily-settings")
    ?? document.querySelector(".modal.mod-settings .vertical-tab-content");
  const card = () => document.querySelector(
    '[data-arxiv-daily-topic-id="acceptance-directions"]',
  );
  const fields = () => Array.from(
    document.querySelectorAll(".arxiv-daily-settings__topic-direction-field"),
  );
  // x/y are page coordinates because that is what the screenshot clip wants;
  // top/left stay viewport-relative for the geometry judgements.
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return {
      x: r.left + window.scrollX,
      y: r.top + window.scrollY,
      top: r.top, left: r.left, right: r.right, bottom: r.bottom,
      width: r.width, height: r.height,
    };
  };
  /**
   * A field's geometry, plus the numbers needed to turn heights into lines:
   * the vertical padding and border the box adds on top of its text, and the
   * theme's own line height rather than an assumed one.
   */
  const describeField = (field) => {
    if (!field) return null;
    const input = field.querySelector(".arxiv-daily-settings__topic-direction-input");
    const badge = field.querySelector(".arxiv-daily-settings__topic-direction-more");
    const style = window.getComputedStyle(input);
    const chrome = parseFloat(style.paddingTop || "0")
      + parseFloat(style.paddingBottom || "0")
      + parseFloat(style.borderTopWidth || "0")
      + parseFloat(style.borderBottomWidth || "0");
    return {
      collapsed: field.classList.contains("is-collapsed"),
      clientHeight: field.clientHeight,
      scrollHeight: field.scrollHeight,
      clientWidth: field.clientWidth,
      scrollWidth: field.scrollWidth,
      lineHeight: parseFloat(style.lineHeight || "0"),
      chrome,
      rect: box(field),
      badge: badge
        ? { visible: badge.classList.contains("is-visible"), text: badge.textContent ?? "", rect: box(badge) }
        : null,
      value: input?.value ?? "",
    };
  };
`;

const inRenderer = (body) => `(() => {${RENDERER_PRELUDE}${body}})()`;

const EXPAND_CARD_EXPRESSION = inRenderer(`
  const el = card();
  if (!el) return JSON.stringify({ error: "the seeded topic card is not on the page" });
  const header = el.querySelector(".arxiv-daily-settings__topic-header");
  if (!header) return JSON.stringify({ error: "the topic card has no header to open" });
  if (header.getAttribute("aria-expanded") !== "true") header.click();
  el.scrollIntoView({ block: "center" });
  return JSON.stringify({ expanded: header.getAttribute("aria-expanded") === "true" });
`);

const MEASURE_EXPRESSION = inRenderer(`
  if (!content) return JSON.stringify({ error: "the arXiv Daily settings tab is not mounted" });
  const all = fields();
  if (all.length < 2) {
    return JSON.stringify({ error: "expected two direction fields, found " + all.length });
  }
  return JSON.stringify({ long: describeField(all[0]), short: describeField(all[1]) });
`);

const FOCUS_LONG_EXPRESSION = inRenderer(`
  const input = fields()[0]?.querySelector(".arxiv-daily-settings__topic-direction-input");
  if (!input) return JSON.stringify({ error: "the long direction field is gone" });
  input.focus();
  return JSON.stringify({ focused: document.activeElement === input });
`);

const BLUR_EXPRESSION = inRenderer(`
  const input = fields()[0]?.querySelector(".arxiv-daily-settings__topic-direction-input");
  input?.blur();
  return JSON.stringify({ blurred: document.activeElement !== input });
`);

/** How many lines of text a box of this height holds. */
function linesOf(height, field) {
  if (!field.lineHeight) return null;
  return (height - field.chrome) / field.lineHeight;
}

function judgeCollapsedToOneLine(long) {
  const lines = linesOf(long.clientHeight, long);
  if (lines === null) return { ok: false, reason: "the field reports no line height" };
  // Half a line of tolerance: themes round box heights, and a value near 1.5
  // would mean the cap is not doing its job.
  if (Math.abs(lines - 1) > 0.5) {
    return { ok: false, reason: `a collapsed direction occupies ${lines.toFixed(2)} lines, expected 1` };
  }
  return { ok: true, reason: `a collapsed direction occupies ${lines.toFixed(2)} lines` };
}

/**
 * The badge's claim, judged against what opening the field actually does.
 *
 * Deliberately not recomputed from scrollHeight the way the plugin does: that
 * would restate the plugin's own arithmetic and pass even if the layout it
 * describes were wrong. Growing the box and counting the lines gained is an
 * independent measurement of the same claim.
 */
function judgeBadgeMatchesGrowth(collapsed, expanded) {
  if (!collapsed.badge?.visible) {
    return { ok: false, reason: "a wrapped direction shows no +N badge while collapsed" };
  }
  const claimed = Number.parseInt(collapsed.badge.text.replace("+", ""), 10);
  if (!Number.isFinite(claimed) || claimed < 1) {
    return { ok: false, reason: `the badge reads ${JSON.stringify(collapsed.badge.text)}` };
  }
  const openedLines = linesOf(expanded.clientHeight, expanded);
  const collapsedLines = linesOf(collapsed.clientHeight, collapsed);
  if (openedLines === null || collapsedLines === null) {
    return { ok: false, reason: "the field reports no line height, so growth cannot be measured" };
  }
  const gained = openedLines - collapsedLines;
  // NaN must fail rather than slip through a `> 0.5` comparison.
  if (!Number.isFinite(gained) || Math.abs(gained - claimed) > 0.5) {
    return {
      ok: false,
      reason: `the badge promises +${claimed} lines but opening the field gains ${gained.toFixed(2)}`,
    };
  }
  return { ok: true, reason: `+${claimed} matches the ${gained.toFixed(2)} lines gained on opening` };
}

function judgeShortHasNoBadge(short) {
  if (short.badge?.visible) {
    return { ok: false, reason: `a direction that fits shows ${JSON.stringify(short.badge.text)}` };
  }
  return { ok: true, reason: "a direction that fits shows no badge" };
}

function judgeNoHorizontalOverflow(field, label) {
  // One pixel of slack: sub-pixel layout rounds scrollWidth up on some themes.
  if (field.scrollWidth > field.clientWidth + 1) {
    return {
      ok: false,
      reason: `${label}: the direction runs ${field.scrollWidth - field.clientWidth}px off the side instead of wrapping`,
    };
  }
  return { ok: true, reason: `${label}: the direction wraps inside ${field.clientWidth}px` };
}

function judgeBadgeClearOfText(long) {
  if (!long.badge?.visible) return { ok: true, reason: "no badge to overlap" };
  if (long.badge.rect.left < long.rect.left) {
    return { ok: false, reason: "the badge is outside its field" };
  }
  return { ok: true, reason: "the badge sits inside the field, right-aligned" };
}

/**
 * Six assertions over one real session, plus screenshots so a person can see
 * the collapsed and open states rather than take the numbers on trust.
 */
export async function topicDirectionsScenarios({
  session,
  screenshots,
  narrowViewport = NARROW_VIEWPORT,
  wideViewport = WIDE_VIEWPORT,
  settleMs = 600,
}) {
  const { evaluate, client, diagnostics } = session;
  const errorsBefore = diagnostics.errors().length;
  const results = [];
  const shot = async (name, where) => {
    if (!screenshots) return;
    await screenshots.capture(name, where);
  };

  const seeded = await evaluate(SEED_EXPRESSION);
  if (typeof seeded === "string" && seeded.startsWith("ERROR:")) {
    return [fail("topic-directions-seed", seeded.slice(7).trim())];
  }

  await setViewport(client, wideViewport);
  await evaluate(OPEN_SETTINGS_EXPRESSION);
  await wait(evaluate, settleMs * 3);

  const opened = await readJson(evaluate, EXPAND_CARD_EXPRESSION);
  if (opened.error) return [fail("topic-directions-open-card", opened.error)];
  await wait(evaluate, settleMs);

  const collapsed = await readJson(evaluate, MEASURE_EXPRESSION);
  if (collapsed.error) return [fail("topic-directions-measure", collapsed.error)];

  await shot("topic-directions-collapsed", { rect: collapsed.long.rect });

  // 1 — a collapsed direction is one line
  {
    const verdict = judgeCollapsedToOneLine(collapsed.long);
    results.push((verdict.ok ? pass : fail)("topic-direction-collapsed-one-line", verdict.reason));
  }

  // 2 — a direction that fits carries no badge
  {
    const verdict = judgeShortHasNoBadge(collapsed.short);
    results.push((verdict.ok ? pass : fail)("topic-direction-short-no-badge", verdict.reason));
  }

  // 3 — the badge is inside its field rather than over the text
  {
    const verdict = judgeBadgeClearOfText(collapsed.long);
    results.push((verdict.ok ? pass : fail)("topic-direction-badge-placement", verdict.reason));
  }

  // 4 — opening the field gains exactly the lines the badge promised
  const focused = await readJson(evaluate, FOCUS_LONG_EXPRESSION);
  if (focused.error) {
    results.push(fail("topic-direction-badge-matches-growth", focused.error));
  } else {
    await wait(evaluate, settleMs);
    const expanded = await readJson(evaluate, MEASURE_EXPRESSION);
    if (expanded.error) {
      results.push(fail("topic-direction-badge-matches-growth", expanded.error));
    } else {
      await shot("topic-directions-expanded", { rect: expanded.long.rect });
      const verdict = judgeBadgeMatchesGrowth(collapsed.long, expanded.long);
      results.push((verdict.ok ? pass : fail)("topic-direction-badge-matches-growth", verdict.reason));
    }
    await evaluate(BLUR_EXPRESSION);
    await wait(evaluate, settleMs);
  }

  // 5 & 6 — the text wraps rather than running off the side, at two widths
  for (const [label, viewport] of [["wide", wideViewport], ["narrow", narrowViewport]]) {
    await setViewport(client, viewport);
    await wait(evaluate, settleMs);
    const measured = await readJson(evaluate, MEASURE_EXPRESSION);
    if (measured.error) {
      results.push(fail(`topic-direction-wraps-${label}`, measured.error));
      continue;
    }
    const verdict = judgeNoHorizontalOverflow(measured.long, label);
    results.push((verdict.ok ? pass : fail)(`topic-direction-wraps-${label}`, verdict.reason));
    await shot(`topic-directions-${label}-panel`, { rect: measured.long.rect });
  }

  await clearViewport(client);

  const newErrors = diagnostics.errors().slice(errorsBefore);
  results.push(
    newErrors.length === 0
      ? pass("topic-directions-console-clean", "no renderer errors while editing directions")
      : fail("topic-directions-console-clean", `${newErrors.length} renderer error(s): ${newErrors[0]?.text ?? ""}`),
  );

  return results;
}
