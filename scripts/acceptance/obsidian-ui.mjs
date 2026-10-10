import fs from "node:fs/promises";
import path from "node:path";
import { createScreenshotWriter } from "../desktop-acceptance/screenshots.mjs";
import { installHttpProxyExpression } from "./obsidian.mjs";

export const PLUGIN = 'app.plugins.plugins["arxiv-daily"]';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

/** Hit testing and consecutive geometry observations are the readiness signal, not elapsed sleep. */
export async function waitForStableTarget(observe, label, { stableSamples = 3, ...options } = {}) {
  let previous, stable = 0;
  return poll(observe, value => {
    const geometry = [value.x, value.y, value.width, value.height];
    if (!value.ready || !value.hitTarget || !geometry.every(Number.isFinite)) {
      previous = undefined;
      stable = 0;
      return false;
    }
    stable = previous && geometry.every((coordinate, index) => coordinate === previous[index]) ? stable + 1 : 1;
    previous = geometry;
    return stable >= stableSamples;
  }, label, { timeoutMs: 5000, intervalMs: 50, ...options });
}

export function dateSubmitTargetExpression(date) {
  return `(() => {
    const input = document.querySelector(".modal input[type=date]");
    const dialog = input?.closest(".modal");
    const button = dialog?.querySelector("button.mod-cta");
    if (!input || !button) return { ready: false, reason: "Date input or submit button is absent" };
    const rect = button.getBoundingClientRect();
    const style = button.ownerDocument.defaultView.getComputedStyle(button);
    const visible = rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
    const x = rect.x + rect.width / 2, y = rect.y + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    return {
      ready: visible && !button.disabled && input.value === ${JSON.stringify(date)} && input.getAttribute("aria-invalid") !== "true",
      value: input.value, disabled: button.disabled, label: button.textContent,
      x, y, width: rect.width, height: rect.height,
      hitTarget: hit === button || button.contains(hit),
      hit: hit ? { tag: hit.tagName, cls: hit.className, text: (hit.textContent ?? "").trim().slice(0, 100) } : null,
    };
  })()`;
}

/** Plugin saveData may briefly expose an absent or partial data.json during a save. */
export async function pollJsonFile(readJson, accepts, label, options) {
  const snapshot = await poll(async () => {
    try {
      return { value: await readJson() };
    } catch (error) {
      if (!(error instanceof SyntaxError) && error.code !== "ENOENT") throw error;
      return { readError: { name: error.name, code: error.code, message: error.message } };
    }
  }, sample => !sample.readError && accepts(sample.value), label, options);
  return snapshot.value;
}

export async function poll(observe, accepts, label, { timeoutMs = 30000, intervalMs = 100 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let value;
  while (Date.now() < deadline) {
    value = await observe();
    if (accepts(value)) return value;
    await pause(intervalMs);
  }
  throw new Error(`Timed out waiting for ${label}; last observation: ${JSON.stringify(value)}`);
}

export async function createObsidianDriver({ session, fixture, artifactDir }) {
  const screenshots = await createScreenshotWriter({ client: session.client, evaluate: session.evaluate, outputDir: artifactDir });
  const dateSubmissions = [];
  const interactionTracePath = path.join(artifactDir, "obsidian-date-interactions.json");
  const saveInteractions = () => fs.writeFile(interactionTracePath, JSON.stringify(dateSubmissions, null, 2) + "\n");
  await saveInteractions();
  const evaluate = session.evaluate;
  const read = expression => evaluate(`JSON.stringify(${expression})`).then(raw => raw === undefined ? undefined : JSON.parse(raw));
  const wait = (expression, accepts, label, options) => poll(() => read(expression), accepts, label, options);
  const command = async id => {
    const accepted = await evaluate(`app.commands.executeCommandById(${JSON.stringify(`arxiv-daily:${id}`)})`);
    if (accepted !== true) throw new Error(`Obsidian command was not registered: ${id}`);
  };
  const click = async selector => evaluate(`(() => {
    const button = document.querySelector(${JSON.stringify(selector)});
    if (!button || !button.isConnected || button.getClientRects().length === 0) throw new Error("Control not visible: " + ${JSON.stringify(selector)});
    if (button.disabled) throw new Error("Control disabled: " + ${JSON.stringify(selector)});
    button.scrollIntoView({ block: "center" }); button.click(); return true;
  })()`);
  const fill = async (selector, value) => evaluate(`(() => {
    const input = document.querySelector(${JSON.stringify(selector)});
    if (!input || input.disabled) throw new Error("Editable control not available: " + ${JSON.stringify(selector)});
    input.scrollIntoView({ block: "center" }); input.focus(); input.value = ${JSON.stringify(value)};
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true })); input.blur(); return true;
  })()`);
  const openSettings = async () => {
    await evaluate('(() => { app.setting.open(); app.setting.openTabById("arxiv-daily"); return true; })()');
    await wait('document.querySelectorAll(".arxiv-daily-settings__model-input").length', value => value > 0, "plugin settings");
  };
  const closeSettings = () => evaluate("(() => { app.setting.close(); return true; })()");
  const openDashboard = async () => {
    await closeSettings();
    await command("open-reading-dashboard");
    await wait('document.querySelectorAll(".arxiv-daily-dashboard__tab").length', value => value > 0, "reading dashboard");
  };
  const runDate = async (date, { force = false } = {}) => {
    const trace = { date, force, samples: [] };
    dateSubmissions.push(trace);
    const targetExpression = dateSubmitTargetExpression(date);
    try {
      await command(force ? "force-run-for-date" : "run-for-date");
      await wait('document.querySelectorAll(".modal input[type=date]").length', value => value === 1, "date dialog");
      await fill(".modal input[type=date]", date);
      const target = await waitForStableTarget(async () => {
        const observed = await read(targetExpression);
        trace.samples.push(observed);
        if (trace.samples.length > 12) trace.samples.shift();
        return observed;
      }, "stable date submit geometry and matching elementFromPoint");
      trace.before = target;
      await saveInteractions();
      await session.client.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: target.x, y: target.y });
      await session.client.send("Input.dispatchMouseEvent", { type: "mousePressed", x: target.x, y: target.y, button: "left", clickCount: 1 });
      await session.client.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: target.x, y: target.y, button: "left", clickCount: 1 });
      trace.after = await read(targetExpression);
      await wait('document.querySelectorAll(".modal input[type=date]").length', value => value === 0, "date dialog to close after submission", { timeoutMs: 5000 });
      trace.closed = true;
    } catch (error) {
      trace.after = await read(targetExpression).catch(() => undefined);
      trace.error = error.message;
      throw new Error(`${error.message}; date submission before=${JSON.stringify(trace.before)} after=${JSON.stringify(trace.after)}`, { cause: error });
    } finally {
      await saveInteractions();
    }
  };
  const reloadPlugin = async () => {
    await closeSettings();
    await evaluate(`(async () => {
      await app.plugins.disablePlugin("arxiv-daily");
      await app.plugins.enablePlugin("arxiv-daily");
      return Boolean(${PLUGIN});
    })()`);
    await evaluate(installHttpProxyExpression(fixture.server.url));
  };
  const screenshot = async (name, t) => {
    await screenshots.capture(name, { selector: ".workspace" });
    const target = path.join(artifactDir, `${name}.png`);
    t?.artifact(target);
    return target;
  };
  const file = async relative => fs.readFile(path.join(fixture.vaultRoot, relative), "utf8");
  const jsonFile = async relative => JSON.parse(await file(relative));
  const exists = async relative => {
    try { await fs.access(path.join(fixture.vaultRoot, relative)); return true; }
    catch (error) { if (error.code === "ENOENT") return false; throw error; }
  };
  const idle = () => wait(`${PLUGIN}.operations.snapshot()`, operations => operations.length === 0, "all tasks to finish", { timeoutMs: 90000 });
  await evaluate(installHttpProxyExpression(fixture.server.url));
  return { session, fixture, artifactDir, interactionTracePath, evaluate, read, wait, command, click, fill, openSettings, closeSettings, openDashboard, runDate, reloadPlugin, screenshot, file, jsonFile, exists, idle, screenshots };
}
