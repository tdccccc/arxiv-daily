import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { blockersFromError } from "../desktop-acceptance/app-state.mjs";
import { preflight } from "../desktop-acceptance/preflight.mjs";
import { runDesktopSession } from "../desktop-acceptance/session.mjs";
import { installSettingsFixture } from "../desktop-acceptance/settings-fixture.mjs";

export const OBSIDIAN_SCENARIOS = [
  ["obsidian.settings.persistence", "设置修改、保存与插件重新加载"],
  ["obsidian.run.cancel-repeat", "运行进度、重复启动与取消恢复"],
  ["obsidian.run.authorization-error", "模型认证失败后的错误提示与状态"],
  ["obsidian.run.retry-complete", "修复外部错误后重新生成完整日报"],
  ["obsidian.run.completed-idempotency", "再次运行已完成日期不重复生成"],
  ["obsidian.reading.star-reload", "打开日报、收藏论文与重新加载恢复"],
  ["obsidian.pdf.location", "真实 PDF 阅读器定位到指定页"],
];

/** Plugin data is independent of the CLI TOML. Only owned fixture endpoints are used. */
export function createObsidianSettings({ endpoint }) {
  return {
    settings: {
      llm: { provider: "openai", baseUrl: endpoint, apiKey: "acceptance-stub-key", model: "fixture-model", thinkingMode: false, reasoningEffort: "medium" },
      arxiv: { category: "cs.AI", categories: ["cs.AI"], timezone: "UTC", topics: [{
        id: "acceptance-research", name: "Research", tag: "research", description: "Reliable research methods", detail: false,
        directions: [{ id: "acceptance-reliable", text: "Reliable research methods", origin: "manual" }],
      }] },
      output: { dailyDir: "arxiv-daily/daily", papersDir: "arxiv-daily/papers", maxDailyPapers: 20, linkStyle: "wikilink", summaryLanguage: "en" },
      detailSelection: { profile: "conservative", normalThreshold: 85, exceptionalThreshold: 95, softLimit: 1 },
      schedule: { enabled: false, runAtLocal: "09:00", runUntilLocal: "18:00", tickIntervalMin: 20 },
      advanced: { requestDelayMs: 0, cacheExpiryDays: 7, sectionCharLimit: 16000, paperCharLimit: 100000, dailyCharLimit: 400000, logLevel: "info" },
      email: { enabled: false, mode: "self", to: "", fromEmail: "", fromName: "Acceptance", apiKey: "", hostedToken: "", hostedBaseUrl: "" },
      embedding: { mode: "local", provider: "", baseUrl: "", apiKey: "", model: "", dimension: 384, initialChoiceDone: false },
      pdfParserSidecar: { enabled: false, capabilitiesUrl: "http://127.0.0.1:5001/v1/capabilities", parseUrl: "http://127.0.0.1:5001/v1/parse" },
      onboarding: { guideCompleted: false },
    },
  };
}

/** Wrap the transport, not the pipeline: real requestUrl cancellation, storage and UI remain in use. */
export function installHttpProxyExpression(endpoint) {
  const url = new URL(endpoint);
  if (url.protocol !== "http:" || !["127.0.0.1", "[::1]"].includes(url.hostname) || url.username || url.password) {
    throw new TypeError("The Obsidian fixture proxy must be an owned HTTP loopback endpoint");
  }
  return `(() => {
    const http = app.plugins.plugins["arxiv-daily"].getHttpClient();
    if (!http.__acceptanceOriginalRequest) http.__acceptanceOriginalRequest = http.request.bind(http);
    http.request = (request) => {
      const target = new URL("/proxy", ${JSON.stringify(url.origin)});
      target.searchParams.set("url", request.url);
      return http.__acceptanceOriginalRequest({ ...request, url: target.href });
    };
    return "fixture transport installed";
  })()`;
}

/** Only an explicitly injected 401 is expected; an uncaught exception always fails. */
export function classifyObsidianDiagnostics(entries, { authorizationFailureExpected = false, fixtureRequests = [] } = {}) {
  const expected = [], unexpected = [];
  const observedFixture401 = fixtureRequests.some(request => request.method === "POST" && request.kind === "model" && request.status === 401);
  for (const entry of entries) {
    if (entry.source !== "pageerror" && !["error", "assert"].includes(entry.level)) continue;
    const controlled401 = entry.source === "console" && authorizationFailureExpected
      && observedFixture401
      && entry.text.includes("Controlled acceptance authentication failure");
    (controlled401 ? expected : unexpected).push(entry);
  }
  return { expected, unexpected };
}

function stoppedSuite(status, error) {
  return {
    suite: "obsidian", status, errors: [error],
    scenarios: OBSIDIAN_SCENARIOS.map(([id, title]) => ({ id, title, status, durationMs: 0, error, assertions: [], steps: [], artifacts: [] })),
  };
}

/** Run only against the caller's disposable fixture; never consult a user's vault list. */
export async function runObsidianAcceptance({ fixture, artifactDir }, runtime = {}) {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
  const obsidianPath = process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian";
  const sourceDir = path.join(repoRoot, "plugin");
  const check = runtime.preflight ?? preflight;
  let environment;
  try {
    environment = await check({ vaultPath: fixture.vaultRoot, obsidianPath, sourceDir });
  } catch (error) {
    return stoppedSuite("blocked", `Obsidian preflight could not complete: ${error.message}`);
  }
  if (!environment.ok) {
    return stoppedSuite("blocked", environment.blockers.map(({ message, remedy }) => `${message}; ${remedy}`).join("\n"));
  }
  const launch = runtime.runDesktopSession ?? runDesktopSession;
  try {
    await fs.mkdir(artifactDir, { recursive: true });
    const { runObsidianJourneys } = await import("./obsidian-scenarios.mjs");
    return await launch({
      vaultPath: fixture.vaultRoot, pluginId: "arxiv-daily", sourceDir, obsidianPath,
      beforeLaunch: ({ vaultPath, pluginId, fs: storage }) => installSettingsFixture({
        vaultPath, pluginId, fs: storage, data: createObsidianSettings({ endpoint: `${fixture.server.url}/v1` }),
      }),
      body: ({ session }) => runObsidianJourneys({ session, fixture, artifactDir }),
    });
  } catch (error) {
    const blockers = blockersFromError(error);
    return stoppedSuite(blockers ? "blocked" : "failed", blockers
      ? blockers.map(({ message }) => message).join("\n") : error.stack ?? error.message);
  } finally {
    fixture.server?.releaseHeld();
    fixture.server?.setMode({ llm: "normal", arxiv: "normal" });
  }
}
