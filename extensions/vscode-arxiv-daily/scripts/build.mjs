import esbuild from "esbuild";
import { mkdir, rm, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Bundles the extension, together with @arxiv-daily/core and
// @arxiv-daily/node-runtime, into one CommonJS file. The workspace packages are
// TypeScript sources linked into the repository root's node_modules, so the
// root `npm ci` must have run first (esbuild itself is also resolved from
// there). Only `vscode` stays external: the host provides it at runtime.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outfile = path.join(root, "out", "extension.js");

await rm(path.join(root, "out"), { recursive: true, force: true });
await mkdir(path.dirname(outfile), { recursive: true });
await esbuild.build({
  entryPoints: [path.join(root, "src", "extension.js")],
  outfile,
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
  external: ["vscode"],
  minify: true,
  sourcemap: false,
  legalComments: "inline",
  logLevel: "warning",
  // The VS Code companion never loads the native private-storage binding; the
  // paper index is written through the plain atomic path.
  define: { __ARXIV_DAILY_NATIVE_ASSETS__: "{}" },
});

const { size } = await stat(outfile);
console.log(`Bundled out/extension.js (${(size / 1024).toFixed(0)} KiB)`);
