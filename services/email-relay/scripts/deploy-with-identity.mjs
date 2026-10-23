#!/usr/bin/env node
// Deploys the email relay together with the BUILD_IDENTITY variable it needs.
//
// `wrangler deploy` replaces every Worker variable with the ones in
// wrangler.toml plus the --var flags of that one call, so a bare deploy drops
// BUILD_IDENTITY and silently locks automatic delivery. This script is the one
// documented way to deploy: it checks that the tree is clean, derives
// email-relay-v2-<HEAD sha>, refuses an identity that would not match the one
// a finished cutover is bound to, and only then runs a single `wrangler deploy`.
// It reads /ready but never calls a verification, delivery or cutover endpoint
// and needs no secret.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const RELAY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export const BUILD_IDENTITY_PATTERN = /^email-relay-v2-[0-9a-f]{40}$/;

// Phases in which a permanent identity binding exists. "locked" is the state
// before the first inventory action, when any identity may still be chosen.
const BOUND_PHASES = new Set(["inventoried", "observing", "sealed", "ready"]);

export function parseArgs(argv) {
  const options = { dryRun: false, identity: undefined, remoteBaseUrl: undefined, errors: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--dry-run") options.dryRun = true;
    else if (arg === "--identity" || arg === "--remote") {
      const value = argv[i + 1];
      if (!value || value.startsWith("--")) options.errors.push(`${arg} needs a value`);
      else {
        if (arg === "--identity") options.identity = value;
        else options.remoteBaseUrl = value;
        i += 1;
      }
    } else options.errors.push(`unknown argument ${arg}`);
  }
  return options;
}

function defaultRunCommand(cmd, args, opts = {}) {
  try {
    const stdout = execFileSync(cmd, args, {
      cwd: opts.cwd ?? RELAY_ROOT,
      encoding: "utf8",
      stdio: opts.inherit ? "inherit" : ["ignore", "pipe", "pipe"],
      shell: process.platform === "win32",
    });
    return { status: 0, stdout: stdout ?? "", stderr: "" };
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

/** What /ready says about the identity the relay is bound to, if it says anything. */
function describeReadiness(response) {
  if (response.status !== 200 && response.status !== 503) return { kind: "unknown" };
  let body;
  try {
    body = JSON.parse(response.text);
  } catch {
    return { kind: "unknown" };
  }
  if (!body || typeof body !== "object") return { kind: "unknown" };
  if (
    BOUND_PHASES.has(body.phase) &&
    typeof body.buildIdentity === "string" &&
    BUILD_IDENTITY_PATTERN.test(body.buildIdentity)
  ) {
    return { kind: "bound", identity: body.buildIdentity };
  }
  if (body.phase === "locked" && !body.reason) return { kind: "unbound" };
  return {
    kind: "unknown",
    reason: typeof body.reason === "string" ? body.reason : undefined,
  };
}

export function createDeploy(deps) {
  const { runCommand, readTextFile, httpGet } = deps;
  const tomlPath = deps.tomlPath ?? join(RELAY_ROOT, "wrangler.toml");

  return async function runDeploy(options = {}) {
    const steps = [];
    const fail = (name, detail) => {
      steps.push({ name, status: "fail", detail });
      return { ok: false, steps };
    };

    const status = runCommand("git", ["status", "--porcelain", "--untracked-files=all", "--", "."]);
    if (status.status !== 0) {
      return fail("cleanTree", "cannot read the git status of services/email-relay");
    }
    if (status.stdout.trim() !== "") {
      return fail(
        "cleanTree",
        "services/email-relay has uncommitted changes; commit them first, because the identity names the commit that is deployed",
      );
    }
    steps.push({ name: "cleanTree", status: "pass" });

    const head = runCommand("git", ["rev-parse", "HEAD"]);
    const sha = head.status === 0 ? head.stdout.trim() : "";
    if (!/^[0-9a-f]{40}$/.test(sha)) {
      return fail("gitHead", "cannot resolve a 40-hex HEAD SHA");
    }
    steps.push({ name: "gitHead", status: "pass", detail: sha });

    let identity = `email-relay-v2-${sha}`;
    if (options.identity !== undefined) {
      if (!BUILD_IDENTITY_PATTERN.test(options.identity)) {
        return fail(
          "identity",
          "--identity must look like email-relay-v2-<40 lowercase hex characters>",
        );
      }
      identity = options.identity;
      steps.push({ name: "identity", status: "pass", detail: `${identity} (given)` });
    }

    let baseUrl = (options.remoteBaseUrl ?? "").replace(/\/+$/, "");
    if (!baseUrl) {
      try {
        baseUrl = (/PUBLIC_BASE_URL\s*=\s*"([^"]+)"/.exec(readTextFile(tomlPath))?.[1] ?? "")
          .replace(/\/+$/, "");
      } catch {
        baseUrl = "";
      }
    }
    if (!baseUrl) {
      const detail = "no base URL (PUBLIC_BASE_URL), so a cutover binding could not be checked. " +
        "Pass --identity <the identity recorded at cutover>; for an initial deployment, explicitly choose the identity to bind.";
      if (options.identity === undefined) return fail("remoteReady", detail);
      steps.push({ name: "remoteReady", status: "warn", detail });
    } else {
      let ready;
      try {
        ready = await httpGet(`${baseUrl}/ready`);
      } catch {
        ready = { status: 0, text: "" };
      }
      const seen = describeReadiness(ready);
      if (seen.kind === "bound" && seen.identity !== identity) {
        return fail(
          "remoteReady",
          `the relay is bound to ${seen.identity}; deploying ${identity} would lock automatic delivery. ` +
            `To release this commit under the bound identity run: npm run deploy -- --identity ${seen.identity} (docs/runbook-cutover.md, section 9)`,
        );
      }
      if (seen.kind === "bound") {
        steps.push({ name: "remoteReady", status: "pass", detail: `bound to ${seen.identity}` });
      } else if (seen.kind === "unbound") {
        steps.push({ name: "remoteReady", status: "pass", detail: "cutover not started; no identity bound yet" });
      } else {
        const detail = `could not tell whether ${baseUrl} is already bound to an identity` +
          `${seen.reason ? ` (/ready says: ${seen.reason})` : ready.status === 0 ? " (unreachable)" : ""}. ` +
          "Pass --identity <the identity recorded at cutover>; for an initial deployment, explicitly choose the identity to bind.";
        if (options.identity === undefined) return fail("remoteReady", detail);
        steps.push({ name: "remoteReady", status: "warn", detail });
      }
    }

    const args = ["--no-install", "wrangler", "deploy", "--var", `BUILD_IDENTITY:${identity}`];
    const command = `npx ${args.join(" ")}`;
    if (options.dryRun) {
      steps.push({ name: "deploy", status: "skip", detail: `dry run; would run: ${command}` });
      return { ok: true, steps, identity, command };
    }
    const deployed = runCommand("npx", args, { inherit: true });
    if (deployed.status !== 0) {
      return fail("deploy", `wrangler deploy failed (exit ${deployed.status})`);
    }
    steps.push({ name: "deploy", status: "pass", detail: identity });
    return { ok: true, steps, identity, command };
  };
}

export async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.errors.length > 0) {
    for (const error of options.errors) console.error(error);
    console.error("usage: npm run deploy -- [--dry-run] [--identity email-relay-v2-<sha>] [--remote <base-url>]");
    process.exit(2);
  }
  const deploy = createDeploy({
    runCommand: defaultRunCommand,
    readTextFile: (path) => readFileSync(path, "utf8"),
    httpGet: defaultHttpGet,
  });
  const { steps, ok } = await deploy(options);
  for (const step of steps) {
    console.log(`${step.status.toUpperCase().padEnd(4)} ${step.name}${step.detail ? ` - ${step.detail}` : ""}`);
  }
  if (!ok) {
    console.error("deploy refused or failed; nothing was changed unless wrangler reported success above");
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
