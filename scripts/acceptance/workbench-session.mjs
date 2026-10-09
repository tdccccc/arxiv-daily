import { spawn } from "node:child_process";
import { access, mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const delay = ms => new Promise(resolve => { const timer = setTimeout(resolve, ms); timer.unref?.(); });

export class WorkbenchBlockedError extends Error {
  constructor(message, cause) {
    super(message, cause ? { cause } : undefined);
    this.name = "WorkbenchBlockedError";
    this.code = "WORKBENCH_BLOCKED";
    this.status = "blocked";
  }
}

function launchUrl(value) {
  let url;
  try { url = new URL(value); } catch { throw new Error("Invalid workbench launch URL"); }
  if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || !url.port
    || url.username || url.password || url.search || url.hash
    || !/^\/[a-f0-9]{48}\/$/.test(url.pathname)) {
    throw new Error("Workbench launch URL must be the capability link on loopback");
  }
  return url.href;
}

async function defaultBrowser(options) {
  let chromium;
  try { ({ chromium } = await import("playwright")); }
  catch (error) { throw new WorkbenchBlockedError("Install workspace dependencies before launching the acceptance browser", error); }
  let executablePath = options.executablePath || process.env.ARXIV_DAILY_ACCEPTANCE_BROWSER;
  if (!executablePath) {
    for (const candidate of ["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]) {
      try { await access(candidate); executablePath = candidate; break; } catch { /* Try cache if no system browser exists. */ }
    }
  }
  return chromium.launch({
    headless: options.headless ?? true,
    ...(executablePath ? { executablePath } : {}),
    args: ["--disable-dev-shm-usage"], timeout: 20000,
  });
}

/**
 * Start the built CLI, including its real generation child processes, in a fixture.
 * launchBrowser and cliPath are seams for lifecycle contracts; ordinary callers
 * use the current build and the installed browser. The caller owns fixture disposal.
 */
export async function startWorkbenchSession({
  fixture, artifactDir, executablePath, headless = true, signal, launchBrowser = defaultBrowser,
  cliPath = join(repositoryRoot, "apps/cli/dist/arxiv-daily-cli.cjs"), startupTimeoutMs = 20000,
} = {}) {
  signal?.throwIfAborted();
  if (!fixture?.root || !fixture.configHome || fixture.env?.XDG_CONFIG_HOME !== fixture.configHome) {
    throw new WorkbenchBlockedError("A fixture with an isolated XDG configuration is required");
  }
  const configRelative = relative(fixture.root, fixture.configHome);
  if (!configRelative || configRelative.startsWith("..") || isAbsolute(configRelative)) {
    throw new WorkbenchBlockedError("The fixture configuration must remain inside its temporary root");
  }
  await mkdir(artifactDir, { recursive: true });
  try { await access(cliPath); }
  catch (error) { throw new WorkbenchBlockedError("CLI build is missing; run npm run build --workspace apps/cli", error); }

  const artifacts = [
    join(artifactDir, "workbench.stdout.log"),
    join(artifactDir, "workbench.stderr.log"),
    join(artifactDir, "workbench.browser.json"),
    join(artifactDir, "workbench.trace.zip"),
  ];
  const diagnostics = { pageErrors: [], consoleErrors: [], failedRequests: [], blockedRequests: [], responses: [], cleanupErrors: [] };
  let stdout = "", stderr = "", browser, context, page, stopPromise, initialized = false;
  const child = spawn(process.execPath, [cliPath, "ui", "--no-open"], {
    cwd: repositoryRoot, env: { ...fixture.env }, stdio: ["ignore", "pipe", "pipe"],
    detached: process.platform !== "win32", windowsHide: true,
  });
  let closed = false;
  const childClosed = new Promise(resolve => child.once("close", (code, signal) => { closed = true; resolve({ code, signal }); }));
  const capped = (previous, chunk) => (previous + String(chunk)).slice(-2 * 1024 * 1024);
  child.stdout.setEncoding("utf8").on("data", chunk => { stdout = capped(stdout, chunk); });
  child.stderr.setEncoding("utf8").on("data", chunk => { stderr = capped(stderr, chunk); });

  const signalChild = signal => {
    if (closed) return;
    try {
      if (process.platform !== "win32" && child.pid) process.kill(-child.pid, signal);
      else child.kill(signal);
    } catch (error) { if (error.code !== "ESRCH") throw error; }
  };
  const stopChild = async () => {
    if (closed) return;
    signalChild("SIGINT");
    await Promise.race([childClosed, delay(3500)]);
    if (!closed) { signalChild("SIGTERM"); await Promise.race([childClosed, delay(1500)]); }
    if (!closed) { signalChild("SIGKILL"); await Promise.race([childClosed, delay(1500)]); }
    if (!closed) throw new Error("Workbench process did not stop after SIGKILL");
  };
  async function stop() {
    if (stopPromise) return stopPromise;
    stopPromise = (async () => {
      signal?.removeEventListener("abort", onAbort);
      const clean = async action => {
        try { await action(); }
        catch (error) { diagnostics.cleanupErrors.push(error instanceof Error ? error.message : String(error)); }
      };
      if (context) await clean(() => context.tracing.stop({ path: artifacts[3] }));
      if (context) await clean(() => context.close());
      if (browser) await clean(() => browser.close());
      await clean(stopChild);
      await Promise.all([
        writeFile(artifacts[0], stdout),
        writeFile(artifacts[1], stderr),
        writeFile(artifacts[2], JSON.stringify(diagnostics, null, 2) + "\n"),
      ]);
      if (diagnostics.cleanupErrors.length) throw new Error("Workbench cleanup failed: " + diagnostics.cleanupErrors.join("; "));
    })();
    return stopPromise;
  }

  const onAbort = () => {
    if (initialized) void stop().catch(() => {});
    else signalChild("SIGINT");
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    signal?.throwIfAborted();
    const url = await new Promise((resolve, reject) => {
      let ready = "";
      const cleanup = () => { clearTimeout(timer); child.stdout.off("data", onData); child.off("error", onError); child.off("close", onClose); };
      const finish = (error, value) => { cleanup(); error ? reject(error) : resolve(value); };
      const onData = chunk => {
        ready = capped(ready, chunk);
        const match = /^Workbench:\s*(\S+)[ \t]*\r?$/m.exec(ready);
        if (match) {
          try { finish(null, launchUrl(match[1])); }
          catch (error) { finish(error); }
        }
      };
      const onError = error => finish(new WorkbenchBlockedError("Could not launch CLI: " + error.message, error));
      const onClose = code => finish(new Error("Workbench exited before readiness (" + code + "): " + stderr.slice(-2000)));
      const timer = setTimeout(() => finish(new Error("Workbench startup timed out after " + startupTimeoutMs + "ms: " + stderr.slice(-2000))), startupTimeoutMs);
      child.stdout.on("data", onData); child.once("error", onError); child.once("close", onClose);
    });
    signal?.throwIfAborted();
    try { browser = await launchBrowser({ executablePath, headless }); }
    catch (error) {
      throw error instanceof WorkbenchBlockedError ? error : new WorkbenchBlockedError("Acceptance browser could not start: " + error.message, error);
    }
    signal?.throwIfAborted();
    context = await browser.newContext({
      viewport: { width: 1440, height: 1000 }, locale: "zh-CN", timezoneId: "UTC",
      acceptDownloads: false, serviceWorkers: "block",
    });
    signal?.throwIfAborted();
    await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
    const origin = new URL(url).origin;
    await context.route("**/*", async route => {
      const target = route.request().url();
      if (new URL(target).origin === origin) return route.continue();
      diagnostics.blockedRequests.push(target);
      return route.abort("blockedbyclient");
    });
    const attached = new WeakSet();
    const observe = target => {
      if (attached.has(target)) return;
      attached.add(target);
      target.on("pageerror", error => diagnostics.pageErrors.push(error.stack || error.message));
      target.on("console", message => { if (message.type() === "error") diagnostics.consoleErrors.push(message.text()); });
      target.on("requestfailed", request => diagnostics.failedRequests.push({ url: request.url(), error: request.failure()?.errorText }));
      target.on("response", response => diagnostics.responses.push({ url: response.url(), status: response.status() }));
    };
    context.on("page", observe);
    page = await context.newPage();
    observe(page);
    page.setDefaultTimeout(12000);
    page.setDefaultNavigationTimeout(20000);
    await page.goto(url, { waitUntil: "domcontentloaded" });
    initialized = true;
    signal?.throwIfAborted();
    return { page, context, browser, url, stop, diagnostics, artifacts, pid: child.pid };
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error(String(cause));
    try { await stop(); } catch (cleanupError) { error.cleanupError = cleanupError.message; }
    error.artifacts = [];
    for (const path of artifacts) {
      try { await access(path); error.artifacts.push(path); } catch { /* A failed launch may not create a trace. */ }
    }
    throw error;
  }
}
