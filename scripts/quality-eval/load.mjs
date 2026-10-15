import { mkdtemp, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const moduleDirectory = path.dirname(fileURLToPath(import.meta.url));

/**
 * Compile the TypeScript tools against core and node-runtime on demand, the way
 * acceptance/exploration.mjs does, so developer tooling never enters a shipped bundle.
 */
export async function loadQualityEval() {
  const { build } = await import("esbuild");
  const directory = await mkdtemp(path.join(tmpdir(), "arxiv-quality-eval-"));
  try {
    const outfile = path.join(directory, "quality-eval.cjs");
    const cliRequire = createRequire(path.resolve(moduleDirectory, "../../apps/cli/package.json"));
    await build({
      entryPoints: [path.join(moduleDirectory, "index.ts")], outfile, bundle: true, platform: "node", format: "cjs",
      target: "node20", loader: { ".md": "text" }, alias: { "smol-toml": cliRequire.resolve("smol-toml") }, logLevel: "silent",
    });
    return createRequire(import.meta.url)(outfile);
  } finally { await rm(directory, { recursive: true, force: true }); }
}
