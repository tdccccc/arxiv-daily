import esbuild from "esbuild";
import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { noticeBanner, readPakoNotice } from "../../scripts/release-utils.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const thirdPartyBanner = noticeBanner(await readPakoNotice());
const root = resolve(here, "../..");
const outfile = resolve(here, "dist/arxiv-daily-cli.cjs");
await mkdir(dirname(outfile), { recursive: true });
await esbuild.build({
  entryPoints: [resolve(here, "src/main.ts")],
  outfile,
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
  minify: true,
  sourcemap: false,
  loader: { ".md": "text" },
  banner: { js: `#!/usr/bin/env node\n${thirdPartyBanner}` },
  legalComments: "inline",
});
await Promise.all([
  copyFile(outfile, resolve(root, "plugin/arxiv-daily-cli.cjs")),
  copyFile(resolve(root, "arxiv_daily.py"), resolve(here, "dist/arxiv_daily.py")),
]);
