#!/usr/bin/env node
// Guards where a release tag may come from. A tag pushed on an unreviewed branch
// (or on a commit older than a security fix already on main) would otherwise be
// released and published as long as its own metadata is self-consistent.
//
// docs/release.md tags the release branch BEFORE it merges into main, so "the tag
// commit is on main" cannot be required. Instead:
//   - reachable: the commit is reachable from origin/main or from the version's own
//     release branch (origin/<prefix><version>), and
//   - current:   (mode "tag" only) origin/main's tip is an ancestor of the commit,
//     so the release contains every fix already merged.
// Mode "republish" (manual re-publish of an already released version, or publishing
// a commit release.yml already accepted) checks reachability only, because main
// legitimately moves on after a release.

import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const MAIN_REF = "refs/remotes/origin/main";

export function releaseBranchRef(version, prefix = "release/") {
  return `refs/remotes/origin/${prefix}${version}`;
}

/**
 * Pure decision. `git` supplies `refExists(ref)` and `isAncestor(ancestor, descendant)`.
 * Returns { ok, message }.
 */
export function evaluateReleaseSource({ commit, version, mode = "tag", prefix = "release/", git }) {
  if (mode !== "tag" && mode !== "republish") {
    return { ok: false, message: `Unknown mode ${JSON.stringify(mode)}; expected "tag" or "republish".` };
  }
  if (!git.refExists(MAIN_REF)) {
    return { ok: false, message: `${MAIN_REF} is missing; cannot verify where ${version} was tagged. Fetch origin/main first.` };
  }
  const releaseRef = releaseBranchRef(version, prefix);
  const hasReleaseBranch = git.refExists(releaseRef);
  const onMain = git.isAncestor(commit, MAIN_REF);
  const onReleaseBranch = hasReleaseBranch && git.isAncestor(commit, releaseRef);
  if (!onMain && !onReleaseBranch) {
    return {
      ok: false,
      message:
        `Commit ${commit} is not reachable from origin/main or origin/${prefix}${version}.\n` +
        `Releases must be tagged on main or on the ${prefix}${version} branch that is pushed for review; ` +
        `a tag on any other commit is refused.`,
    };
  }
  if (mode === "tag" && !git.isAncestor(MAIN_REF, commit)) {
    return {
      ok: false,
      message:
        `origin/main has commits that ${commit} does not contain, so this release would omit changes already merged.\n` +
        `Merge or rebase main into ${prefix}${version}, re-run the release checks, and tag the new head ` +
        `(a tag that never produced a release can be deleted and recreated; a published one must not be moved).`,
    };
  }
  const where = onMain ? "origin/main" : `origin/${prefix}${version}`;
  return { ok: true, message: `Release source OK: ${commit} is reachable from ${where}${mode === "tag" ? " and contains origin/main" : ""}.` };
}

function gitRunner(cwd) {
  const run = (args) => execFileSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  return {
    refExists(ref) {
      try { run(["rev-parse", "--verify", "--quiet", `${ref}^{commit}`]); return true; } catch { return false; }
    },
    isAncestor(ancestor, descendant) {
      try { run(["merge-base", "--is-ancestor", ancestor, descendant]); return true; }
      catch (error) {
        if (error.status === 1) return false;
        throw error;
      }
    },
    fetch(refspec, { optional }) {
      try { run(["fetch", "--no-tags", "--quiet", "origin", refspec]); }
      catch (error) { if (!optional) throw error; }
    },
  };
}

export function parseArguments(argv) {
  const options = { mode: "tag", prefix: "release/", fetch: true };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--no-fetch") options.fetch = false;
    else if (["--commit", "--version", "--mode", "--prefix"].includes(arg)) {
      const value = argv[++index];
      if (!value || value.startsWith("--")) throw new Error(`${arg} requires a value`);
      options[arg.slice(2)] = value;
    } else throw new Error(`Unknown argument ${arg}`);
  }
  if (!options.commit || !options.version) throw new Error("--commit and --version are required");
  if (!/^[0-9a-f]{40}$/i.test(options.commit)) throw new Error("--commit must be a full 40-character SHA");
  // Self-contained on purpose: publish-cli runs a copy of this file taken from origin/main.
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(options.version)) throw new Error("--version must be a stable X.Y.Z version");
  if (!/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(options.prefix) || options.prefix.includes("..")) {
    throw new Error("--prefix must be a plain branch-name prefix such as release/ or release/dsh-");
  }
  return options;
}

export function main(argv = process.argv.slice(2), cwd = process.cwd()) {
  let options;
  try { options = parseArguments(argv); }
  catch (error) {
    console.error(error.message);
    console.error("Usage: node scripts/check-release-source.mjs --commit SHA --version X.Y.Z [--mode tag|republish] [--prefix release/] [--no-fetch]");
    return 2;
  }
  const git = gitRunner(cwd);
  if (options.fetch) {
    try {
      git.fetch(`+refs/heads/main:${MAIN_REF}`, { optional: false });
      git.fetch(`+refs/heads/${options.prefix}${options.version}:${releaseBranchRef(options.version, options.prefix)}`, { optional: true });
    } catch (error) {
      console.error(`Could not fetch origin/main to verify the release source: ${error.message}`);
      return 1;
    }
  }
  const result = evaluateReleaseSource({ ...options, git });
  (result.ok ? console.log : console.error)(result.message);
  return result.ok ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exitCode = main();
}
