import { mkdir, readFile, realpath, stat } from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { createExplorationRedactor, ExplorationActionError, parseExplorationAction } from "./exploration.mjs";

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

function contained(root, file) {
  const relative = path.relative(root, file);
  return relative !== "" && !relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative);
}

async function assertFixture(fixture) {
  if (!fixture?.root || !fixture.vaultRoot || !fixture.configPath) throw new Error("Exploration needs an owned temporary fixture");
  const root = await realpath(fixture.root);
  for (const file of [fixture.vaultRoot, fixture.configPath]) if (!contained(root, await realpath(file))) throw new Error("Exploration fixture paths must remain under its owned temporary root");
}

async function settings(fixture) {
  const { parse } = cliRequire("smol-toml");
  const config = parse(await readFile(fixture.configPath, "utf8"));
  if (path.resolve(config.vault_root) !== path.resolve(fixture.vaultRoot)) throw new Error("Fixture vault location changed outside the exploration contract");
  return config;
}

async function preferences(fixture) {
  try { return JSON.parse(await readFile(path.join(path.dirname(fixture.configPath), "workbench-ui.json"), "utf8")); }
  catch (error) { if (error.code === "ENOENT") return { sidebarCollapsed: false }; throw error; }
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
    const label = node.getAttribute("aria-label") || labelIds.map(labelId => document.getElementById(labelId)?.textContent || "").join(" ") ||
      Array.from(node.labels || []).map(item => item.textContent).join(" ") || node.closest("[data-setting-name]")?.getAttribute("data-setting-name") || node.getAttribute("title") || field || node.textContent;
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

export async function createExplorationDriver({ page, context, url, fixture, artifactDir, diagnostics, redact = createExplorationRedactor(), signal }) {
  await assertFixture(fixture);
  await mkdir(artifactDir, { recursive: true, mode: 0o700 });
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
          const live = await target.evaluate(node => ({ tag: node.tagName.toLowerCase(), type: node.getAttribute("type") || "", field: node.getAttribute("name") || "",
            name: node.getAttribute("aria-label") || node.textContent || "", href: node.getAttribute("href") || "",
            dataAction: node.getAttribute("data-settings-action") || node.getAttribute("data-settings") || node.getAttribute("data-action") || "", disabled: node.disabled === true,
            readOnly: node.readOnly === true, outsideDialog: Boolean(document.querySelector("dialog[open]") && !node.closest("dialog[open]")) }));
          const restriction = classifyControlRestriction(live, url);
          if (restriction || live.disabled || live.outsideDialog) throw new ExplorationActionError(restriction || "The control is no longer interactive");
          parseExplorationAction(JSON.stringify(action), { controls: [{ ...latest.controls.find(control => control.ref === action.ref), ...live }] });
        }
        if (action.type === "click") await target.click({ timeout: 6000 });
        else if (action.type === "fill") await target.fill(action.value, { timeout: 6000 });
        else if (action.type === "select") await target.selectOption(action.value, { timeout: 6000 });
        else if (action.type === "reload") { await page.reload({ waitUntil: "domcontentloaded", timeout: 8000 }); await page.locator(".workbench").waitFor({ state: "visible", timeout: 6000 }); }
        else if (action.type === "back") await page.goBack({ waitUntil: "domcontentloaded", timeout: 8000 });
        else if (action.type === "wait") await new Promise(resolve => setTimeout(resolve, action.milliseconds));
        else if (action.type === "scroll") {
          if (target) await target.hover({ timeout: 6000 });
          else { const region = page.locator("dialog[open] .settings-content, .reading-pane").last(); if (await region.isVisible()) await region.hover(); }
          await page.mouse.wheel(0, action.direction === "down" ? action.pixels : -action.pixels);
        }
        options.signal?.throwIfAborted(); assertScope();
        return { ok: true };
      } finally { options.signal?.removeEventListener("abort", onAbort); }
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
    async dispose() { page.off("pageerror", onError); context.off("page", onPage); await context.unroute("**/*", guard); },
  };
}

const check = (label, passed, actual) => ({ label, passed: Boolean(passed), ...(actual === undefined ? {} : { actual }) });
const result = checks => ({ passed: checks.every(value => value.passed), checks });

export async function createDefaultExplorationTasks(fixture) {
  await assertFixture(fixture);
  const initial = await settings(fixture);
  const target = initial.output?.max_daily_papers === 13 ? 14 : 13;
  let collapsedAfterReload = false, collapseReloadStep = 0;
  return [
    {
      id: "settings-persistence", title: "设置自动保存并刷新复查",
      goal: `打开设置，把每日论文数量上限 maxDailyPapers 改为 ${target}。移开焦点并等待保存完成，关闭设置，刷新页面，再打开设置，确认上限仍为 ${target}。不要修改路径、模型连接或密钥。`,
      async verify({ driver, observation, actions }) {
        const changed = actions.find(action => action.type === "fill" && (action.control?.field || action.control?.name) === "maxDailyPapers" && action.value === String(target));
        const refreshed = changed && actions.some(action => action.type === "reload" && action.step > changed.step);
        const state = await driver.inspect();
        return result([
          check("通过界面修改每日论文数量上限", changed),
          check("修改后刷新了页面", refreshed),
          check("隔离配置文件保存了新上限", state.maxDailyPapers === target, state.maxDailyPapers),
          check("刷新后重新打开设置仍显示新上限", refreshed && observation.controls.some(control => control.field === "maxDailyPapers" && control.value === String(target))),
        ]);
      },
    },
    {
      id: "sidebar-persistence", title: "侧栏折叠和展开状态持久化",
      goal: "关闭设置（如果仍打开）。收起侧栏，刷新页面确认侧栏仍收起；再展开侧栏，再次刷新，确认侧栏仍展开。等待偏好保存后再刷新。",
      async verify({ driver, observation, actions }) {
        const state = await driver.inspect();
        const last = actions.findLast(action => action.type === "reload");
        const lastToggle = actions.findLast(action => action.type === "click" && action.control?.purpose === "sidebar-toggle");
        const clicked = actions.some(action => action.type === "click" && action.control?.purpose === "sidebar-toggle");
        if (clicked && last && lastToggle && last.step > lastToggle.step && state.preferences.sidebarCollapsed === true && observation.layout.sidebarCollapsed) {
          collapsedAfterReload = true; collapseReloadStep = last.step;
        }
        const expandedAfterReload = collapsedAfterReload && last && lastToggle && last.step > lastToggle.step && last.step > collapseReloadStep && state.preferences.sidebarCollapsed === false && !observation.layout.sidebarCollapsed;
        return result([check("通过界面切换侧栏", clicked), check("刷新后持久化收起状态与页面一致", collapsedAfterReload), check("再展开并刷新后持久化状态与页面一致", expandedAfterReload)]);
      },
    },
    {
      id: "generate-and-read", title: "生成固定日期日报并打开论文",
      goal: "通过生成按钮生成 2026-10-01 日报（选择指定日期，不要生成今天）。等待成功，再从列表打开 arXiv 2610.10001 论文，确认阅读内容出现。无需判断论文内容质量。",
      async verify({ driver, observation, actions, startedAt }) {
        const state = await driver.dailyEvidence("2026-10-01");
        const generated = actions.some(action => action.type === "click" && action.control?.dataAction === "generate");
        const run = state.run;
        const checks = [
          check("通过生成入口启动任务", generated),
          check("本任务的指定日期运行已完成", run?.status === "completed" && run.date === "2026-10-01" && Date.parse(run.startedAt) >= startedAt, run ? { date: run.date, status: run.status } : null),
          check("真实日报文件包含夹具论文", state.report),
          check("论文阅读视图已打开", observation.layout.view === "reading" && observation.text.includes("2610.10001")),
        ];
        if (run?.date === "2026-10-01" && Date.parse(run.startedAt) >= startedAt && (run.status === "failed" || (run.status === "completed" && !state.report))) {
          return { passed: false, failed: true, checks, error: run.status === "failed" ? "The requested fixture generation reported failure" : "Generation reported completion without the expected report file" };
        }
        return result(checks);
      },
    },
  ];
}
