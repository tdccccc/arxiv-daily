#!/usr/bin/env node
// Guards the rule that cost arXiv Daily its Community directory listing on
// 2026-10-03: Obsidian reads the root manifest.json on the default branch to
// learn the latest version, then looks for a GitHub release whose tag equals
// that version. If main ever names a version with no published (non-draft)
// matching release, Obsidian delists the plugin from in-app search within
// about 24h. This check fails CI before such a gap can land on main.

import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { readJson } from "./release-utils.mjs";

export const DEFAULT_REPO = "tdccccc/arxiv-daily";

/**
 * I/O: look up a GitHub release by exact tag via the REST API.
 * Returns a discriminated result instead of throwing, so callers can produce
 * a clear, specific message for every outcome (found/missing/error).
 */
export async function fetchReleaseByTag(repo, tag, { fetchImpl = fetch, token } = {}) {
  const headers = {
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetchImpl(
      `https://api.github.com/repos/${repo}/releases/tags/${encodeURIComponent(tag)}`,
      { headers, signal: AbortSignal.timeout(30_000) },
    );
  } catch (error) {
    return { kind: "error", message: `request to the GitHub API failed: ${error.message}` };
  }

  if (response.status === 404) return { kind: "missing" };

  if (!response.ok) {
    let detail = "";
    try {
      detail = await response.text();
    } catch {
      // Best-effort detail only; the status line is enough to fail clearly.
    }
    return {
      kind: "error",
      message: `GitHub API responded ${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`,
    };
  }

  let body;
  try {
    body = await response.json();
  } catch (error) {
    return { kind: "error", message: `GitHub API returned an unparsable response: ${error.message}` };
  }

  return { kind: "found", draft: Boolean(body?.draft), htmlUrl: body?.html_url };
}

/**
 * Pure: manifest version + release lookup result -> ok/error message.
 * No network, no filesystem; easy to exercise exhaustively in tests.
 */
export function evaluatePublishedManifest(version, result) {
  const prefix = `Root manifest.json names version ${version}`;
  const rule =
    "Obsidian's Community directory reads manifest.json from main and expects a " +
    "published GitHub release with the matching tag; a mismatch gets the plugin " +
    "delisted from in-app search within about 24h.";

  switch (result?.kind) {
    case "found":
      if (result.draft) {
        return {
          ok: false,
          message:
            `${prefix}, but GitHub release "${version}" exists only as a draft.\n${rule}\n` +
            `Publish the release (finish it from the release/${version} branch, where it should ` +
            `have been tagged and pushed before merging into main) before merging this into main, ` +
            "then re-run this check.",
        };
      }
      return { ok: true, message: `Published GitHub release ${version} found; manifest.json is safe to land on main.` };

    case "missing":
      return {
        ok: false,
        message:
          `${prefix}, but no GitHub release with tag "${version}" exists yet.\n${rule}\n` +
          `Publish the release from the release/${version} branch first (push the annotated tag ` +
          "so release.yml runs and creates the GitHub release), then re-run this check.",
      };

    case "error":
      return {
        ok: false,
        message:
          `${prefix}, but verifying its GitHub release failed: ${result.message}\n` +
          "Treating an unverifiable release as a failure rather than silently passing. Re-run once " +
          "the GitHub API is reachable (set GH_TOKEN or GITHUB_TOKEN if this was a rate limit).",
      };

    default:
      return { ok: false, message: `${prefix}, but the release lookup returned an unrecognized result.` };
  }
}

/** Orchestration: read the manifest, look up its release, evaluate. */
export async function checkPublishedManifest({ fetchImpl = fetch, env = process.env } = {}) {
  const manifest = await readJson("manifest.json");
  const version = manifest.version;
  const repo = env.GITHUB_REPOSITORY || DEFAULT_REPO;
  const token = env.GH_TOKEN || env.GITHUB_TOKEN;
  const result = await fetchReleaseByTag(repo, version, { fetchImpl, token });
  return evaluatePublishedManifest(version, result);
}

export async function main() {
  const evaluation = await checkPublishedManifest();
  if (!evaluation.ok) {
    console.error(evaluation.message);
    process.exit(1);
  }
  console.log(evaluation.message);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
