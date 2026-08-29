/**
 * A vault opened for the first time under a fresh configuration is gated by
 * Obsidian's trust prompt, and community plugins do not load until it is
 * accepted. That gate is also what makes diagnostics complete: the harness
 * attaches before accepting, so the whole plugin startup is observed.
 */
const TRUST_BUTTON_LABEL = "Trust author";

const TRUST_EXPRESSION = `(() => {
  const button = Array.from(document.querySelectorAll(".modal button"))
    .find((candidate) => candidate.textContent.includes(${JSON.stringify(TRUST_BUTTON_LABEL)}));
  if (!button) return "absent";
  button.click();
  return "clicked";
})()`;

/**
 * The harness attaches as soon as the renderer exposes a debugging target,
 * which can be before Obsidian has built its `app` object, so the query must
 * survive a renderer that is not ready rather than throw a ReferenceError.
 */
function pluginVersionExpression(pluginId) {
  return `typeof app === "undefined" ? null : (app.plugins?.plugins?.[${JSON.stringify(pluginId)}]?.manifest?.version ?? null)`;
}

export async function acceptVaultTrust({ evaluate }) {
  const outcome = await evaluate(TRUST_EXPRESSION);
  return { accepted: outcome === "clicked" };
}

/**
 * Wait on the observable condition — the plugin appearing in Obsidian's own
 * registry — rather than on a fixed sleep that is either flaky or slow.
 */
export async function waitForPluginLoaded({
  evaluate,
  pluginId,
  attempts = 60,
  intervalMs = 250,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
}) {
  const expression = pluginVersionExpression(pluginId);
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const version = await evaluate(expression);
    if (typeof version === "string" && version.length > 0) return version;
    await sleep(intervalMs);
  }
  throw new Error(
    `plugin ${pluginId} never loaded: still absent from app.plugins.plugins after ${attempts} attempts`,
  );
}

/**
 * Drive the renderer from "just attached" to "plugin running": the trust prompt
 * may not be mounted yet on the first poll, so acceptance is retried alongside
 * the readiness check instead of being attempted once and assumed handled.
 */
export async function waitForPluginReady({
  evaluate,
  pluginId,
  attempts = 120,
  intervalMs = 250,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
}) {
  const expression = pluginVersionExpression(pluginId);
  let trustPromptAccepted = false;

  // Probing before touching the trust prompt tells us whether the plugin was
  // already running when the harness attached. If it was, its startup output
  // happened before diagnostics were enabled and the collected log is
  // incomplete — the caller needs to know rather than trust a clean result.
  const preexisting = await evaluate(expression);
  const loadedBeforeAttach = typeof preexisting === "string" && preexisting.length > 0;
  if (loadedBeforeAttach) {
    return { version: preexisting, trustPromptAccepted, loadedBeforeAttach };
  }

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (!trustPromptAccepted) {
      const { accepted } = await acceptVaultTrust({ evaluate });
      trustPromptAccepted ||= accepted;
    }
    const version = await evaluate(expression);
    if (typeof version === "string" && version.length > 0) {
      return { version, trustPromptAccepted, loadedBeforeAttach };
    }
    await sleep(intervalMs);
  }
  throw new Error(
    `plugin ${pluginId} never loaded after ${attempts} attempts (trust prompt ${trustPromptAccepted ? "was" : "was never"} accepted)`,
  );
}
