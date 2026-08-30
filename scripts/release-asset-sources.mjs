/**
 * Holds every other copy of the release asset list to the one in
 * `scripts/release-assets.mjs`.
 *
 * Each parser here fails loudly. A checklist whose wording moved, a workflow
 * whose steps were renamed, or a list that parsed to nothing must never read as
 * agreement: the only way to report "consistent" is to have actually read a
 * non-empty list out of every source.
 */
import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { parse as parseYaml } from "yaml";
import { RELEASE_ASSETS } from "./release-assets.mjs";
import { deployedArtifactPaths } from "./desktop-acceptance/build-deploy.mjs";

export const root = resolve(import.meta.dirname, "..");

export const RELEASE_DOC_PATH = "docs/release.md";
export const RELEASE_WORKFLOW_PATH = ".github/workflows/release.yml";

/** The sentence in `docs/release.md` that introduces the asset list. */
export const RELEASE_DOC_MARKER = "The Obsidian release assets remain exactly:";
const ATTEST_ACTION = "actions/attest-build-provenance";
const GH_RELEASE_CREATE = "gh release create";
const REPO_PATH_PATTERN = /^plugin\/([A-Za-z0-9._-]+)$/;

export const CANONICAL_LABEL = "RELEASE_ASSETS in scripts/release-assets.mjs";
export const DOC_LABEL = `${RELEASE_DOC_PATH} (the release checklist read by humans)`;
export const ATTEST_LABEL = `${RELEASE_WORKFLOW_PATH} attestation subject-path`;
export const UPLOAD_LABEL = `${RELEASE_WORKFLOW_PATH} \`${GH_RELEASE_CREATE}\` arguments`;
export const DEPLOY_LABEL =
  "scripts/desktop-acceptance/build-deploy.mjs deployedArtifactPaths()";

function assetFromRepoPath(value, source, where) {
  const match = REPO_PATH_PATTERN.exec(value);
  if (!match) {
    throw new Error(
      `${source}: ${where} names ${JSON.stringify(value)}, which is not a `
        + "`plugin/<file>` release asset; the asset list can no longer be read reliably",
    );
  }
  return match[1];
}

function rejectDuplicates(assets, source, where) {
  const seen = new Set();
  for (const asset of assets) {
    if (seen.has(asset)) {
      throw new Error(`${source}: ${where} lists ${asset} more than once`);
    }
    seen.add(asset);
  }
  return assets;
}

/**
 * Read the bullet list under `RELEASE_DOC_MARKER`.
 *
 * `docs/release.md` stays hand-written prose; only this one list is machine
 * read. Anything that is not exactly the expected shape — marker gone, marker
 * duplicated, bullets replaced by a table or a sentence, a bullet naming
 * something other than `plugin/<file>` — throws, so a reformatted document
 * cannot quietly parse as an empty and therefore "agreeing" list.
 */
export function parseReleaseDocAssets(markdown, { source = RELEASE_DOC_PATH } = {}) {
  if (typeof markdown !== "string") {
    throw new Error(`${source}: expected markdown text, received ${typeof markdown}`);
  }
  const lines = markdown.split(/\r?\n/);
  const markers = [];
  lines.forEach((line, index) => {
    if (line.trim() === RELEASE_DOC_MARKER) markers.push(index + 1);
  });
  if (markers.length === 0) {
    throw new Error(
      `${source}: no line reads exactly ${JSON.stringify(RELEASE_DOC_MARKER)}, so the `
        + "release asset list cannot be located; if the wording moved, update "
        + "RELEASE_DOC_MARKER in scripts/release-asset-sources.mjs alongside it",
    );
  }
  if (markers.length > 1) {
    throw new Error(
      `${source}: ${JSON.stringify(RELEASE_DOC_MARKER)} appears on lines `
        + `${markers.join(", ")}; the checked asset list must appear exactly once`,
    );
  }

  let index = markers[0]; // zero-based index of the line after the marker
  while (index < lines.length && lines[index].trim() === "") index += 1;

  const assets = [];
  for (; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.startsWith("- ")) break;
    const bullet = /^- `([^`]+)`\s*$/.exec(line);
    if (!bullet) {
      throw new Error(
        `${source}: line ${index + 1} is inside the release asset list but is not a `
          + `\`- \`plugin/<file>\`\` bullet: ${JSON.stringify(line)}`,
      );
    }
    assets.push(assetFromRepoPath(bullet[1], source, `line ${index + 1}`));
  }

  if (assets.length === 0) {
    throw new Error(
      `${source}: found ${JSON.stringify(RELEASE_DOC_MARKER)} on line ${markers[0]} but no `
        + "`- `plugin/<file>`` bullets under it; an empty list is a parse failure, "
        + "not an empty release",
    );
  }
  return rejectDuplicates(assets, source, "the release asset list");
}

function workflowSteps(workflow, source) {
  const jobs = workflow?.jobs;
  if (!jobs || typeof jobs !== "object") {
    throw new Error(`${source}: no \`jobs:\` mapping, so the release steps cannot be read`);
  }
  const steps = Object.values(jobs).flatMap((job) => (Array.isArray(job?.steps) ? job.steps : []));
  if (steps.length === 0) throw new Error(`${source}: no workflow steps found`);
  return steps;
}

function attestedAssets(steps, source) {
  const matching = steps.filter(
    (step) => typeof step?.uses === "string" && step.uses.startsWith(`${ATTEST_ACTION}@`),
  );
  if (matching.length !== 1) {
    throw new Error(
      `${source}: expected exactly one \`${ATTEST_ACTION}\` step to read the attested `
        + `asset list from, found ${matching.length}`,
    );
  }
  const subjectPath = matching[0]?.with?.["subject-path"];
  if (typeof subjectPath !== "string" || subjectPath.trim() === "") {
    throw new Error(
      `${source}: the \`${ATTEST_ACTION}\` step has no \`subject-path\` list, `
        + `received ${JSON.stringify(subjectPath)}`,
    );
  }
  const assets = subjectPath
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== "")
    .map((line) => assetFromRepoPath(line, source, "the attestation `subject-path`"));
  if (assets.length === 0) {
    throw new Error(`${source}: the attestation \`subject-path\` list is empty`);
  }
  return rejectDuplicates(assets, source, "the attestation `subject-path`");
}

function uploadedAssets(steps, source) {
  const matching = steps.filter(
    (step) => typeof step?.run === "string" && step.run.includes(GH_RELEASE_CREATE),
  );
  if (matching.length !== 1) {
    throw new Error(
      `${source}: expected exactly one step running \`${GH_RELEASE_CREATE}\` to read the `
        + `uploaded asset list from, found ${matching.length}`,
    );
  }
  const run = matching[0].run;
  const rest = run
    .slice(run.indexOf(GH_RELEASE_CREATE) + GH_RELEASE_CREATE.length)
    .replace(/\\\r?\n/g, " ");
  const positional = [];
  for (const token of rest.split(/\s+/).filter(Boolean)) {
    if (token.startsWith("-")) break;
    positional.push(token);
  }
  const [tag, ...uploads] = positional;
  if (tag === undefined) {
    throw new Error(`${source}: \`${GH_RELEASE_CREATE}\` is invoked without arguments`);
  }
  if (uploads.length === 0) {
    throw new Error(
      `${source}: \`${GH_RELEASE_CREATE} ${tag}\` uploads no assets; an empty upload list is a `
        + "parse failure, not a release without assets",
    );
  }
  return rejectDuplicates(
    uploads.map((upload) => assetFromRepoPath(upload, source, `\`${GH_RELEASE_CREATE}\``)),
    source,
    `\`${GH_RELEASE_CREATE}\``,
  );
}

/**
 * Read both asset lists out of the release workflow: the files whose provenance
 * is attested, and the files actually uploaded to the GitHub release.
 */
export function parseWorkflowReleaseAssets(
  workflowText,
  { source = RELEASE_WORKFLOW_PATH } = {},
) {
  if (typeof workflowText !== "string") {
    throw new Error(`${source}: expected YAML text, received ${typeof workflowText}`);
  }
  let workflow;
  try {
    workflow = parseYaml(workflowText);
  } catch (error) {
    throw new Error(`${source}: is not valid YAML: ${error.message}`);
  }
  const steps = workflowSteps(workflow, source);
  return { attested: attestedAssets(steps, source), uploaded: uploadedAssets(steps, source) };
}

/**
 * Describe how two asset lists differ, naming both sides and every file that is
 * only on one of them. Order is not part of the contract; membership is.
 */
export function compareAssetLists(expectedLabel, expected, actualLabel, actual) {
  const missing = expected.filter((asset) => !actual.includes(asset));
  const unexpected = actual.filter((asset) => !expected.includes(asset));
  if (missing.length === 0 && unexpected.length === 0) return null;
  const differences = [];
  if (missing.length > 0) differences.push(`${actualLabel} is missing ${missing.join(", ")}`);
  if (unexpected.length > 0) {
    differences.push(`${actualLabel} additionally lists ${unexpected.join(", ")}`);
  }
  return (
    `release asset lists disagree: ${expectedLabel} has [${expected.join(", ")}] but `
    + `${actualLabel} has [${actual.join(", ")}] — ${differences.join("; ")}`
  );
}

function deployedAssets() {
  return deployedArtifactPaths("/vault", { pluginId: "arxiv-daily" }).map((path) => basename(path));
}

/**
 * Compare every copy of the release asset list against the canonical one.
 * Returns the problems found; an empty array means, and only means, that every
 * source was read successfully and every one of them agreed.
 */
export async function verifyReleaseAssetSources({
  canonical = RELEASE_ASSETS,
  read = (relativePath) => readFile(resolve(root, relativePath), "utf8"),
} = {}) {
  const issues = [];
  const lists = [];

  try {
    lists.push([DOC_LABEL, parseReleaseDocAssets(await read(RELEASE_DOC_PATH))]);
  } catch (error) {
    issues.push(error.message);
  }
  try {
    const { attested, uploaded } = parseWorkflowReleaseAssets(await read(RELEASE_WORKFLOW_PATH));
    lists.push([ATTEST_LABEL, attested], [UPLOAD_LABEL, uploaded]);
  } catch (error) {
    issues.push(error.message);
  }
  try {
    lists.push([DEPLOY_LABEL, deployedAssets()]);
  } catch (error) {
    issues.push(`${DEPLOY_LABEL}: ${error.message}`);
  }

  if (!Array.isArray(canonical) || canonical.length === 0) {
    issues.push(
      `${CANONICAL_LABEL} is empty; a release ships at least one asset, so an empty canonical `
        + "list is a defect rather than something the other copies should agree with",
    );
    return issues;
  }

  for (const [label, list] of lists) {
    if (list.length === 0) {
      issues.push(`${label} yielded no assets; treating that as agreement would hide the drift`);
      continue;
    }
    const mismatch = compareAssetLists(CANONICAL_LABEL, [...canonical], label, list);
    if (mismatch) issues.push(mismatch);
  }
  return issues;
}
