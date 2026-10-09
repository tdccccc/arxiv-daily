import { mkdir, writeFile } from "node:fs/promises";
import { isAbsolute, join, relative } from "node:path";

export class BlockedError extends Error { code = "ACCEPTANCE_BLOCKED"; }
const STATES = new Set(["passed", "failed", "blocked", "not-run"]);

export function redact(value, key = "") {
  if (/^(?:api_?key|api-key|authorization|password|secret|cookie|access_token|refresh_token)$/i.test(key)) return "[redacted]";
  if (typeof value === "string") return value.replace(/Bearer\s+[^\s"'<>]+/gi, "Bearer [redacted]").replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[redacted]");
  if (Array.isArray(value)) return value.map(item => redact(item));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([name, item]) => [name, redact(item, name)]));
  return value;
}

/** A callback returning normally is insufficient: at least one assertion must pass. */
export async function runScenario({ id, title, artifactDir, captureFailure }, action) {
  const started = Date.now();
  const result = { id, title, status: "passed", durationMs: 0, steps: [], assertions: [], artifacts: [] };
  try {
    await action({
      step(label) { result.steps.push(String(label)); },
      check(passed, label, actual) {
        result.assertions.push({ label, passed: passed === true, ...(actual === undefined ? {} : { actual: redact(actual) }) });
        if (passed !== true) throw new Error(`Assertion failed: ${label}`);
      },
      artifact(path) { if (path) result.artifacts.push(String(path)); },
    });
    if (result.assertions.length === 0) throw new Error("Scenario completed without an assertion");
  } catch (error) {
    result.status = error?.code === "ACCEPTANCE_BLOCKED" || error?.name === "BlockedError" ? "blocked" : "failed";
    result.error = redact(error?.message ?? String(error));
    if (captureFailure) {
      try {
        if (artifactDir) await mkdir(artifactDir, { recursive: true });
        const captured = await captureFailure(artifactDir ? join(artifactDir, `${id.replace(/[^a-zA-Z0-9_.-]/g, "_")}-failure.png`) : undefined);
        if (captured) result.artifacts.push(captured);
      } catch (captureError) { result.captureError = redact(captureError.message); }
    }
  }
  result.durationMs = Date.now() - started;
  return redact(result);
}

function normalizeSuite(suite) {
  const scenarios = (suite.scenarios ?? []).map(scenario => {
    const result = { ...scenario };
    if (!STATES.has(result.status)) { result.status = "failed"; result.error = "Scenario returned an invalid status"; }
    if (result.status === "passed" && (!Array.isArray(result.assertions) || result.assertions.length === 0 || result.assertions.some(check => check.passed !== true))) {
      result.status = "failed"; result.error = "Passing scenario has missing or unsuccessful assertions";
    }
    return result;
  });
  let status = suite.status;
  if (!STATES.has(status)) status = "failed";
  if (scenarios.some(s => s.status === "failed")) status = "failed";
  else if (status === "passed" && (scenarios.length === 0 || scenarios.some(s => s.status !== "passed"))) status = "blocked";
  return { ...suite, status, scenarios };
}

export function buildReport({ suites = [], selectedSuites = [], metadata = {} }) {
  const seen = new Set();
  const normalized = suites.map(suite => {
    if (seen.has(suite.suite)) throw new Error(`Duplicate suite evidence: ${suite.suite}`);
    seen.add(suite.suite); return normalizeSuite(suite);
  });
  for (const suite of selectedSuites) if (!seen.has(suite)) normalized.push({ suite, status: "not-run", scenarios: [], error: "Requested suite produced no evidence" });
  const status = normalized.some(s => s.status === "failed") ? "failed" : normalized.length === 0 || normalized.some(s => s.status !== "passed") ? "blocked" : "passed";
  const counts = { passed: 0, failed: 0, blocked: 0, "not-run": 0 };
  for (const suite of normalized) for (const scenario of suite.scenarios) counts[scenario.status]++;
  return redact({ schemaVersion: 1, generatedAt: new Date().toISOString(), status, exitCode: status === "passed" ? 0 : status === "failed" ? 1 : 2, selectedSuites, counts, metadata, suites: normalized });
}

const escape = value => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");

function artifactLink(path, directory) {
  const local = isAbsolute(path) ? relative(directory, path) : path;
  // Evidence links are relative filesystem artifacts, never executable URL schemes.
  if (/^[a-z][a-z\d+.-]*:/i.test(local)) return `<code>${escape(local)}</code>`;
  return `<a href="${escape(local.split("/").map(encodeURIComponent).join("/"))}">${escape(local)}</a>`;
}

export async function writeReport(directory, input) {
  const report = redact(input);
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "report.json"), JSON.stringify(report, null, 2) + "\n");
  const suites = report.suites.map(suite => `<section><h2>${escape(suite.suite)} <span class="${escape(suite.status)}">${escape(suite.status)}</span></h2>${suite.error ? `<pre>${escape(suite.error)}</pre>` : ""}${suite.scenarios.map(s => `<details ${s.status !== "passed" ? "open" : ""}><summary><strong class="${escape(s.status)}">${escape(s.status)}</strong> ${escape(s.title)} <small>${escape(s.id)} · ${Number(s.durationMs ?? 0)} ms</small></summary>${s.error ? `<pre>${escape(s.error)}</pre>` : ""}<ol>${(s.steps ?? []).map(step => `<li>${escape(typeof step === "string" ? step : JSON.stringify(step))}</li>`).join("")}</ol><ul>${(s.assertions ?? []).map(check => `<li class="${check.passed ? "passed" : "failed"}">${check.passed ? "✓" : "✗"} ${escape(check.label ?? check.description)}${check.actual === undefined ? "" : `<pre>${escape(JSON.stringify(check.actual, null, 2))}</pre>`}</li>`).join("")}</ul><ul>${(s.artifacts ?? []).map(path => `<li>${artifactLink(path, directory)}</li>`).join("")}</ul>${s.findings?.length ? `<pre>${escape(JSON.stringify(s.findings, null, 2))}</pre>` : ""}</details>`).join("")}<ul>${(suite.artifacts ?? []).map(path => `<li>${artifactLink(path, directory)}</li>`).join("")}</ul></section>`).join("\n");
  const html = `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>arXiv Daily · 交互验收</title><style>body{max-width:1040px;margin:40px auto;padding:0 24px;background:#faf9f6;color:#252a2e;font:16px/1.6 system-ui,sans-serif}h1,h2{line-height:1.2}h2{margin-top:36px}small{color:#65717a;font-weight:400}summary{cursor:pointer;padding:12px 0}details{border-bottom:1px solid #d9ddde;padding:6px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0efea;padding:12px;font-size:13px}.passed{color:#226542}.failed{color:#a22c2c}.blocked,.not-run{color:#8d5d00}a{color:#245a83}li{margin:5px 0}</style><h1>arXiv Daily · 交互验收</h1><p><strong class="${report.status}">${escape(report.status)}</strong> · ${escape(report.generatedAt)}</p><p>只报告本次实际执行的套件与断言；论文内容质量不在本次评测范围。</p><pre>${escape(JSON.stringify(report.counts, null, 2))}</pre>${suites}<p><a href="report.json">完整机器可读报告</a></p></html>`;
  await writeFile(join(directory, "report.html"), html);
  return { json: join(directory, "report.json"), html: join(directory, "report.html") };
}
