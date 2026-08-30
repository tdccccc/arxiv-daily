import fsPromises from "node:fs/promises";
import path from "node:path";
import { RELEASE_ASSETS } from "../release-assets.mjs";

/**
 * The harness deploys exactly what a release ships, from the one list both read.
 *
 * `styles.css` is on that list because the settings-page geometry assertions
 * measure what the stylesheet lays out. While the harness kept its own copy of
 * the list, omitting `styles.css` made the run read the vault's leftover
 * stylesheet and report a layout no build on the branch produces — silently.
 * Sharing the list means a fourth release asset reaches the harness by the same
 * edit that adds it to the release.
 */
const ARTIFACTS = RELEASE_ASSETS;

/**
 * The files the harness overwrites in the test vault. The vault also holds
 * historical `main.js.bak-*` builds; those are the user's and are never read,
 * written, or removed.
 */
export function deployedArtifactPaths(vaultPath, { pluginId }) {
  const pluginDir = path.join(vaultPath, ".obsidian", "plugins", pluginId);
  return ARTIFACTS.map((name) => path.join(pluginDir, name));
}

function readManifestVersion(raw) {
  let manifest;
  try {
    manifest = JSON.parse(raw.toString());
  } catch {
    throw new Error("manifest.json in the build output is not valid JSON");
  }
  const { version } = manifest;
  if (typeof version !== "string" || version.length === 0) {
    throw new Error(
      `manifest.json has no usable version, received ${JSON.stringify(version)}`,
    );
  }
  return version;
}

/**
 * Copy the current branch build into the test vault so acceptance runs against
 * the code under test rather than whatever build the vault happened to hold.
 * Both artifacts are read and validated before anything is written, so an
 * unbuilt branch cannot leave the vault half-updated.
 */
export async function deployBuildUnderTest({ vaultPath, pluginId, sourceDir, fs = fsPromises }) {
  const sources = [];
  for (const name of ARTIFACTS) {
    const source = path.join(sourceDir, name);
    try {
      sources.push(await fs.readFile(source));
    } catch (error) {
      if (error?.code === "ENOENT") {
        throw new Error(`build output missing: ${source} — run the plugin build first`);
      }
      throw error;
    }
  }
  const version = readManifestVersion(sources[ARTIFACTS.indexOf("manifest.json")]);

  const targets = deployedArtifactPaths(vaultPath, { pluginId });
  await fs.mkdir(path.dirname(targets[0]), { recursive: true });
  for (const [index, target] of targets.entries()) {
    await fs.writeFile(target, sources[index]);
  }
  return { version, deployed: targets };
}

/**
 * Guard against accepting evidence produced by a stale build that Obsidian
 * loaded instead of the one just deployed.
 */
export function assertVersionUnderTest({ expected, reported }) {
  if (typeof reported !== "string" || reported.length === 0) {
    throw new Error(
      `the running plugin reported no version, received ${JSON.stringify(reported)}`,
    );
  }
  if (reported !== expected) {
    throw new Error(
      `build under test is ${expected} but Obsidian loaded ${reported}; acceptance evidence would describe the wrong build`,
    );
  }
  return reported;
}
