#!/usr/bin/env node
// Read-only cutover preflight for the email relay v2 production cutover.
// This script never deploys (only --dry-run), never mutates KV, never
// calls verification or delivery endpoints, and never reads secret values.
// All production mutations (deploy, credential revocation, cutover actions)
// happen manually per the runbook, not from this script.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const RELAY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export const REQUIRED_TOML_NAMES = [
  "DELIVER_GATE",
  "STORE",
  "PUBLIC_BASE_URL",
  "FROM_EMAIL",
  "FROM_NAME",
  "DAILY_QUOTA",
];

export const REQUIRED_SECRET_NAMES = [
  "RESEND_API_KEY",
  "TOKEN_SECRET",
  "IDENTITY_SECRET",
  "DELIVERY_V2_CUTOVER_TOKEN",
];

// Tokens this script must never contain outside this single definition line.
const FORBIDDEN_TOKENS = ["v1/verify/start", "v1/deliver", "kv:key put", "kv:key delete", "internal/delivery-v2/cutover"];

/**
 * Static self-check: the source must never contain a non-dry-run deploy,
 * a KV mutation, or a verification/delivery/cutover endpoint reference.
 * Lines that define this check itself are skipped.
 */
export function checkSourceReadonly(source) {
  const lines = source.split("\n");
  const scanned = lines
    .map((line) => (line.includes("FORBIDDEN") || line.includes("checkSourceReadonly") ? "" : line))
    .join("\n");
  const issues = [];
  const deployRefs = (scanned.match(/"deploy"/g) ?? []).length;
  const dryRunRefs = (scanned.match(/"--dry-run"/g) ?? []).length;
  if (deployRefs > dryRunRefs) {
    issues.push(`${deployRefs} deploy reference(s) without a matching --dry-run`);
  }
  for (const token of FORBIDDEN_TOKENS) {
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (line.includes("FORBIDDEN") || line.includes("checkSourceReadonly")) continue;
      if (line.includes(token)) issues.push(`line ${i + 1}: forbidden token "${token}"`);
    }
  }
  return issues;
}

function defaultRunCommand(cmd, args, opts = {}) {
  try {
    const stdout = execFileSync(cmd, args, {
      cwd: opts.cwd ?? RELAY_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return { status: 0, stdout, stderr: "" };
  } catch (err) {
    const e = err;
    return {
      status: typeof e.status === "number" ? e.status : 1,
      stdout: String(e.stdout ?? ""),
      stderr: String(e.stderr ?? ""),
    };
  }
}

async function defaultHttpGet(url, timeoutMs = 10_000) {
  const res = await fetch(url, {
    signal: AbortSignal.timeout(timeoutMs),
    redirect: "follow",
  });
  return { status: res.status, text: await res.text() };
}

export function createPreflight(deps) {
  const { runCommand, readTextFile, httpGet } = deps;
  const tomlPath = deps.tomlPath ?? join(RELAY_ROOT, "wrangler.toml");

  return async function runPreflight(options = {}) {
    const results = [];

    const git = runCommand("git", ["rev-parse", "HEAD"]);
    const buildSha = git.status === 0 && /^[0-9a-f]{40}$/.test(git.stdout.trim())
      ? git.stdout.trim()
      : null;
    results.push(buildSha
      ? { name: "gitHead", status: "pass", detail: buildSha }
      : { name: "gitHead", status: "fail", detail: "cannot resolve a 40-hex HEAD SHA" });

    let toml = null;
    try {
      toml = readTextFile(tomlPath);
    } catch {
      results.push({ name: "wranglerToml", status: "fail", detail: `cannot read ${tomlPath}` });
    }
    if (toml !== null) {
      const missing = REQUIRED_TOML_NAMES.filter((name) => !toml.includes(name));
      results.push(missing.length === 0
        ? { name: "wranglerToml", status: "pass", detail: `${REQUIRED_TOML_NAMES.length} required names present` }
        : { name: "wranglerToml", status: "fail", detail: `missing: ${missing.join(", ")}` });
    }

    const tomlBase = toml !== null
      ? (/PUBLIC_BASE_URL\s*=\s*"([^"]+)"/.exec(toml)?.[1] ?? "").replace(/\/+$/, "")
      : "";
    const baseUrl = (options.remoteBaseUrl ?? tomlBase).replace(/\/+$/, "");

    if (options.skipLogin) {
      results.push({ name: "wranglerLogin", status: "skip", detail: "skipped by --skip-login" });
    } else {
      const whoami = runCommand("npx", ["--no-install", "wrangler", "whoami"]);
      results.push(whoami.status === 0
        ? { name: "wranglerLogin", status: "pass", detail: "authenticated" }
        : { name: "wranglerLogin", status: "fail", detail: "not logged in; run `npx wrangler login`" });
    }

    const secrets = runCommand("npx", ["--no-install", "wrangler", "secret", "list"]);
    if (secrets.status === 0) {
      const missing = REQUIRED_SECRET_NAMES.filter((name) => !secrets.stdout.includes(name));
      results.push(missing.length === 0
        ? { name: "wranglerSecrets", status: "pass", detail: `${REQUIRED_SECRET_NAMES.length} required secret names present` }
        : { name: "wranglerSecrets", status: "fail", detail: `missing secret names: ${missing.join(", ")}` });
    } else {
      results.push({ name: "wranglerSecrets", status: "fail", detail: "`wrangler secret list` failed; check login and account" });
    }

    if (buildSha !== null) {
      const tmp = mkdtempSync(join(tmpdir(), "relay-dryrun-"));
      const dryRun = runCommand("npx", [
        "--no-install",
        "wrangler",
        "deploy",
        "src/index.ts",
        "--dry-run",
        "--config",
        "wrangler.toml",
        "--var",
        `BUILD_IDENTITY:email-relay-v2-${buildSha}`,
        "--outdir",
        tmp,
      ]);
      rmSync(tmp, { recursive: true, force: true });
      results.push(dryRun.status === 0
        ? { name: "wranglerDryRun", status: "pass", detail: `bundle for email-relay-v2-${buildSha}` }
        : { name: "wranglerDryRun", status: "fail", detail: "dry-run failed; see wrangler output" });
    } else {
      results.push({ name: "wranglerDryRun", status: "fail", detail: "requires a resolvable HEAD SHA" });
    }

    if (!baseUrl) {
      results.push({ name: "remoteReadonly", status: "skip", detail: "no base URL (set PUBLIC_BASE_URL or pass --remote)" });
    } else {
      let health;
      try {
        health = await httpGet(`${baseUrl}/health`);
      } catch {
        health = { status: 0, text: "" };
      }
      results.push(health.status === 200
        ? { name: "remoteHealth", status: "pass", detail: `${baseUrl}/health` }
        : { name: "remoteHealth", status: "fail", detail: `${baseUrl}/health unreachable or unexpected (${health.status})` });

      let ready;
      try {
        ready = await httpGet(`${baseUrl}/ready`);
      } catch {
        ready = { status: 0, text: "" };
      }
      if (ready.status === 200) {
        results.push({ name: "remoteReady", status: "pass", detail: `ready: ${ready.text.trim().slice(0, 200)}` });
      } else if (ready.status === 503) {
        results.push({ name: "remoteReady", status: "pass", detail: "locked (503) — expected before cutover completes; manual checks continue" });
      } else {
        results.push({ name: "remoteReady", status: "fail", detail: `${baseUrl}/ready unexpected or unreachable (${ready.status})` });
      }
    }

    const ok = results.every((r) => r.status !== "fail");
    return { results, ok };
  };
}

export async function main() {
  const args = process.argv.slice(2);

  if (args.includes("--check-readonly")) {
    const source = readFileSync(fileURLToPath(import.meta.url), "utf8");
    const issues = checkSourceReadonly(source);
    for (const issue of issues) console.error(`READONLY: ${issue}`);
    if (issues.length > 0) process.exit(1);
    console.log("cutover-preflight is read-only: PASS");
    process.exit(0);
  }

  const remoteIndex = args.indexOf("--remote");
  const remoteBaseUrl = remoteIndex >= 0 ? args[remoteIndex + 1] ?? "" : undefined;
  const skipLogin = args.includes("--skip-login");

  const preflight = createPreflight({
    runCommand: defaultRunCommand,
    readTextFile: (path) => readFileSync(path, "utf8"),
    httpGet: defaultHttpGet,
  });
  const { results, ok } = await preflight({ remoteBaseUrl, skipLogin });
  for (const r of results) {
    console.log(`${r.status.toUpperCase().padEnd(4)} ${r.name}${r.detail ? ` — ${r.detail}` : ""}`);
  }
  if (!ok) {
    console.error("preflight FAILED: resolve failures before any cutover step");
    process.exit(1);
  }
  console.log("preflight OK: read-only checks passed");
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
