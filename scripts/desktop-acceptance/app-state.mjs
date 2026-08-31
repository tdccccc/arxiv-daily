/**
 * Is the application in a state where walking it means anything?
 *
 * On 2026-08-31 it was not. The machine's file-watch quota was spent, Obsidian
 * came up on its own error page with the vault never opened, and the walk that
 * followed reported seventeen passes and wrote ten screenshots of that page.
 * The only red was "the renderer logged no error", so the run exited 1 —
 * *failed*, as though the product were at fault — instead of 2, *blocked*,
 * which is what an environment that cannot host the acceptance deserves.
 *
 * The guard here is deliberately not a matcher for that page. Matching the
 * ENOSPC sentence would have caught exactly one of the ways an environment can
 * take Obsidian down, and would have gone on believing every other one. What is
 * checked instead is the positive capability the whole acceptance rests on: a
 * vault window is mounted and reachable — Obsidian's own object graph is built,
 * the workspace is on screen with leaves in it, and the settings entry point
 * the walk drives actually exists. Whatever the page says when that is not
 * true is *reported*, so the operator sees the real cause, and is never what
 * the verdict is made of.
 */

/**
 * Read the capabilities as data. Nothing here decides anything: the page's own
 * words and buttons come back so a blocker can quote them, and the judgement is
 * a pure function below.
 */
export const APP_USABILITY_EXPRESSION = `(() => {
  const host = typeof app === "undefined" ? null : app;
  const found = (selector) => Boolean(document.querySelector(selector));
  const text = (document.body ? (document.body.innerText || document.body.textContent || "") : "")
    .replace(/\\s+/g, " ")
    .trim();
  return JSON.stringify({
    url: location.href,
    app: Boolean(host),
    workspace: Boolean(host && host.workspace),
    settings: Boolean(host && host.setting),
    workspaceContainer: found(".workspace") || found(".workspace-split") || found(".app-container"),
    leaves: document.querySelectorAll(".workspace-leaf").length,
    buttons: Array.from(document.querySelectorAll("button"))
      .map((button) => (button.textContent || "").trim())
      .filter((label) => label.length > 0)
      .slice(0, 8),
    text: text.slice(0, 300),
  });
})()`;

/**
 * Every capability the acceptance needs before any of its assertions can mean
 * what they say. Each is a thing the walk goes on to use, not a proxy for one.
 */
const REQUIRED = [
  { key: "app", missing: "Obsidian's app object" },
  { key: "workspace", missing: "app.workspace" },
  { key: "settings", missing: "app.setting, which is how the walk opens the settings page" },
  { key: "workspaceContainer", missing: "a workspace element in the document" },
];

export function judgeAppUsability(state) {
  if (!state || typeof state !== "object") {
    return { ok: false, reason: `the renderer described no state, received ${JSON.stringify(state)}` };
  }
  const missing = REQUIRED.filter((requirement) => !state[requirement.key]).map((r) => r.missing);
  if (!(Number(state.leaves) > 0)) missing.push("any open workspace leaf, so no vault is showing");

  if (missing.length === 0) {
    return { ok: true, reason: `a vault window with ${state.leaves} workspace leaf/leaves and a settings entry point` };
  }
  const shows = state.text ? ` The page reads: "${state.text}".` : "";
  const offers = state.buttons?.length > 0 ? ` It offers ${state.buttons.join(" / ")}.` : "";
  return {
    ok: false,
    reason:
      `the renderer is not showing a usable vault window — it has no ${missing.join(", no ")}.`
      + `${shows}${offers}`,
  };
}

/**
 * A blocked run, not a failed one. Carries blockers in the shape `preflight`
 * produces so the entry point can print and exit them the same way.
 */
export class AppErrorStateError extends Error {
  constructor(message, blockers) {
    super(message);
    this.name = "AppErrorStateError";
    this.blockers = blockers;
  }
}

/** `null` for an ordinary failure: the two exit codes must not blur together. */
export function blockersFromError(error) {
  return Array.isArray(error?.blockers) && error.blockers.length > 0 ? error.blockers : null;
}

const REMEDY =
  "fix what the page above reports and rerun — it names the environment failure Obsidian hit, not a product "
  + "defect; if it names a file-watch limit, raise fs.inotify.max_user_watches as the preflight blocker describes";

/**
 * Check the application once, and refuse to continue if it is not usable.
 *
 * `phase` is part of the message because the two moments mean different things
 * to whoever reads it: broken *before* the walk means nothing was ever driven,
 * broken *after* means results exist and are being discarded.
 */
export async function assertAppUsable({ evaluate, phase = "before the walk" }) {
  const raw = await evaluate(APP_USABILITY_EXPRESSION);
  const state = typeof raw === "string" ? JSON.parse(raw) : raw;
  const verdict = judgeAppUsability(state);
  if (verdict.ok) return state;
  throw new AppErrorStateError(
    `Obsidian was not in a usable state ${phase}: ${verdict.reason}`,
    [
      {
        message: `Obsidian was not in a usable state ${phase}: ${verdict.reason}`,
        remedy: REMEDY,
      },
    ],
  );
}

/**
 * Run `run` only against a usable application, and let its results out only if
 * the application is still usable afterwards.
 *
 * The second check is the point. An application that collapses part-way through
 * still returns a result set, and that set is worthless — every assertion in it
 * was made against whatever replaced the vault window. Throwing here discards
 * it entirely rather than reporting it, so a broken host can never produce a
 * single PASS.
 */
export async function withAppUsable({ evaluate, run }) {
  await assertAppUsable({ evaluate, phase: "before the walk" });
  const outcome = await run();
  await assertAppUsable({ evaluate, phase: "after the walk, so its results are discarded" });
  return outcome;
}
