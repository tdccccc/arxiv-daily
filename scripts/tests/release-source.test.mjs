import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { parse } from "yaml";
import { evaluateReleaseSource, main, parseArguments } from "../check-release-source.mjs";
import { root } from "../release-utils.mjs";

function git(cwd, ...args) {
  // Ignore the developer's global hooks and signing settings: these are throwaway repositories.
  return execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", ...args], {
    cwd, encoding: "utf8",
    env: { ...process.env, GIT_AUTHOR_NAME: "t", GIT_AUTHOR_EMAIL: "t@example.test", GIT_COMMITTER_NAME: "t", GIT_COMMITTER_EMAIL: "t@example.test" },
  }).trim();
}

/** A bare origin plus a clone, the way the workflows see the repository. */
async function fixture() {
  const dir = await mkdtemp(join(tmpdir(), "release-source-"));
  const origin = join(dir, "origin.git");
  const work = join(dir, "work");
  execFileSync("git", ["init", "--bare", "-b", "main", origin]);
  execFileSync("git", ["clone", "-q", origin, work]);
  git(work, "checkout", "-q", "-b", "main");
  const commit = async (name) => {
    await writeFile(join(work, name), name);
    git(work, "add", name);
    git(work, "commit", "-q", "-m", name);
    return git(work, "rev-parse", "HEAD");
  };
  return { dir, work, commit, git: (...args) => git(work, ...args), cleanup: () => rm(dir, { recursive: true, force: true }) };
}

function run(fx, commit, extra = []) {
  const output = [];
  const log = console.log, error = console.error;
  console.log = (m) => output.push(m); console.error = (m) => output.push(m);
  try {
    const code = main(["--commit", commit, "--version", "1.2.3", ...extra], fx.work);
    return { code, output: output.join("\n") };
  } finally { console.log = log; console.error = error; }
}

test("a release branch tag that contains main is accepted before the branch merges", async () => {
  const fx = await fixture();
  try {
    await fx.commit("base");
    fx.git("push", "-q", "origin", "main");
    fx.git("checkout", "-q", "-b", "release/1.2.3");
    const tip = await fx.commit("release-prep");
    fx.git("push", "-q", "origin", "release/1.2.3");
    const result = run(fx, tip);
    assert.equal(result.code, 0, result.output);
    assert.match(result.output, /origin\/release\/1\.2\.3/);
  } finally { await fx.cleanup(); }
});

test("a tag at the tip of main is accepted (no release branch needed)", async () => {
  const fx = await fixture();
  try {
    await fx.commit("a");
    const tip = await fx.commit("b");
    fx.git("push", "-q", "origin", "main");
    assert.equal(run(fx, tip).code, 0);
  } finally { await fx.cleanup(); }
});

test("a tag on an unreviewed branch commit is refused", async () => {
  const fx = await fixture();
  try {
    await fx.commit("base");
    fx.git("push", "-q", "origin", "main");
    fx.git("checkout", "-q", "-b", "scratch");
    const rogue = await fx.commit("rogue");
    fx.git("push", "-q", "origin", "scratch");
    const result = run(fx, rogue);
    assert.equal(result.code, 1);
    assert.match(result.output, /not reachable from origin\/main or origin\/release\/1\.2\.3/);
  } finally { await fx.cleanup(); }
});

test("a release branch for a different version does not authorise the tag", async () => {
  const fx = await fixture();
  try {
    await fx.commit("base");
    fx.git("push", "-q", "origin", "main");
    fx.git("checkout", "-q", "-b", "release/9.9.9");
    const tip = await fx.commit("other");
    fx.git("push", "-q", "origin", "release/9.9.9");
    assert.equal(run(fx, tip).code, 1);
  } finally { await fx.cleanup(); }
});

test("a release branch that lacks commits already merged to main is refused in tag mode", async () => {
  const fx = await fixture();
  try {
    const base = await fx.commit("base");
    fx.git("push", "-q", "origin", "main");
    fx.git("checkout", "-q", "-b", "release/1.2.3", base);
    const tip = await fx.commit("release-prep");
    fx.git("push", "-q", "origin", "release/1.2.3");
    fx.git("checkout", "-q", "main");
    await fx.commit("security-fix");
    fx.git("push", "-q", "origin", "main");
    const result = run(fx, tip);
    assert.equal(result.code, 1);
    assert.match(result.output, /omit changes already merged/);
  } finally { await fx.cleanup(); }
});

test("republish mode accepts an older main commit and a release commit after main moved on", async () => {
  const fx = await fixture();
  try {
    const old = await fx.commit("old");
    fx.git("push", "-q", "origin", "main");
    fx.git("checkout", "-q", "-b", "release/1.2.3");
    const releaseTip = await fx.commit("release-prep");
    fx.git("push", "-q", "origin", "release/1.2.3");
    fx.git("checkout", "-q", "main");
    await fx.commit("later");
    fx.git("push", "-q", "origin", "main");
    assert.equal(run(fx, old, ["--mode", "republish"]).code, 0);
    assert.equal(run(fx, releaseTip, ["--mode", "republish"]).code, 0);
    assert.equal(run(fx, old).code, 1, "tag mode must still require current main");
  } finally { await fx.cleanup(); }
});

test("republish mode still refuses a commit that is on neither main nor the release branch", async () => {
  const fx = await fixture();
  try {
    await fx.commit("base");
    fx.git("push", "-q", "origin", "main");
    fx.git("checkout", "-q", "-b", "scratch");
    const rogue = await fx.commit("rogue");
    fx.git("push", "-q", "origin", "scratch");
    assert.equal(run(fx, rogue, ["--mode", "republish"]).code, 1);
  } finally { await fx.cleanup(); }
});

test("a missing origin/main is a failure, not a pass", async () => {
  const fx = await fixture();
  try {
    const tip = await fx.commit("base");
    const result = run(fx, tip, ["--no-fetch"]);
    assert.equal(result.code, 1);
    assert.match(result.output, /origin\/main|refs\/remotes\/origin\/main/);
  } finally { await fx.cleanup(); }
});

test("a custom branch prefix is honoured", async () => {
  const fx = await fixture();
  try {
    await fx.commit("base");
    fx.git("push", "-q", "origin", "main");
    fx.git("checkout", "-q", "-b", "release/dsh-1.2.3");
    const tip = await fx.commit("prep");
    fx.git("push", "-q", "origin", "release/dsh-1.2.3");
    assert.equal(run(fx, tip).code, 1);
    assert.equal(run(fx, tip, ["--prefix", "release/dsh-"]).code, 0);
  } finally { await fx.cleanup(); }
});

test("arguments are validated", () => {
  assert.throws(() => parseArguments([]), /required/);
  assert.throws(() => parseArguments(["--commit", "abc", "--version", "1.2.3"]), /40-character/);
  assert.throws(() => parseArguments(["--commit", "a".repeat(40), "--version", "latest"]), /stable X\.Y\.Z/);
  assert.throws(() => parseArguments(["--commit", "a".repeat(40), "--version", "1.2.3", "--bogus"]), /Unknown argument/);
  assert.throws(() => parseArguments(["--commit", "a".repeat(40), "--version", "1.2.3", "--prefix", "../x"]), /prefix/);
  const cli = spawnSync(process.execPath, [join(root, "scripts/check-release-source.mjs")], { encoding: "utf8" });
  assert.equal(cli.status, 2);
});

test("the pure decision rejects an unknown mode", () => {
  const git = { refExists: () => true, isAncestor: () => true };
  assert.equal(evaluateReleaseSource({ commit: "a".repeat(40), version: "1.2.3", mode: "yolo", git }).ok, false);
});

async function workflow(name) {
  const source = await readFile(join(root, ".github/workflows", name), "utf8");
  return { source, parsed: parse(source, { schema: "core", uniqueKeys: true }) };
}

function runSteps(parsed, job) {
  return parsed.jobs[job].steps.filter((step) => typeof step.run === "string");
}

test("release.yml runs the source check against the tag commit before building", async () => {
  const { parsed } = await workflow("release.yml");
  const steps = parsed.jobs.release.steps;
  const index = steps.findIndex((step) => /check-release-source\.mjs/.test(step.run ?? ""));
  assert.ok(index >= 0, "release.yml must call scripts/check-release-source.mjs");
  assert.match(steps[index].run, /--mode tag/);
  assert.match(steps[index].run, /--commit "\$EXPECTED_SHA"/);
  const install = steps.findIndex((step) => step.run === "npm ci");
  assert.ok(index < install, "source check must precede npm ci");
});

test("publish-cli.yml checks the source on both the dispatch and workflow_run paths", async () => {
  const { parsed } = await workflow("publish-cli.yml");
  const resolve = runSteps(parsed, "publish").find((step) => /check-release-source\.mjs/.test(step.run));
  assert.ok(resolve, "publish-cli.yml must call scripts/check-release-source.mjs");
  assert.match(resolve.run, /--mode republish/);
  assert.doesNotMatch(resolve.run, /--mode tag/);
});

test("publish-dsh-plugin.yml checks the source and runs the same gates as publish-cli", async () => {
  const { parsed, source } = await workflow("publish-dsh-plugin.yml");
  const steps = runSteps(parsed, "publish");
  const check = steps.find((step) => /check-release-source\.mjs/.test(step.run));
  assert.ok(check, "publish-dsh-plugin.yml must call scripts/check-release-source.mjs");
  assert.match(check.run, /--prefix release\/dsh-/);
  for (const gate of ["npm audit --audit-level=moderate", "npm run lint", "npm run test:release-tools", "test:workspaces", "npm run build", "npm run smoke:build"]) {
    assert.ok(source.includes(gate), `publish-dsh-plugin.yml must run ${gate}`);
  }
});
