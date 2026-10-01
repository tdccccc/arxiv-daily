import esbuild from "esbuild";
import { copyFile, mkdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { noticeBanner, readPakoNotice } from "../../scripts/release-utils.mjs";
import { nativeAssetsForBuild } from "../../scripts/native-assets.mjs";
import { buildWorkbenchAssets } from "./build-workbench.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const thirdPartyBanner = noticeBanner(await readPakoNotice());
const root = resolve(here, "../..");
const outfile = resolve(here, "dist/arxiv-daily-cli.cjs");
const pkg = JSON.parse(await readFile(resolve(here, "package.json"), "utf8"));
const workbench = await buildWorkbenchAssets(here);
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
  define: {
    __ARXIV_DAILY_NATIVE_ASSETS__: JSON.stringify(nativeAssetsForBuild()),
    __ARXIV_DAILY_VERSION__: JSON.stringify(pkg.version ?? "0.0.0"),
    __ARXIV_DAILY_WORKBENCH_ASSETS__: JSON.stringify(workbench.assets),
  },
  banner: { js: `#!/usr/bin/env node\n${thirdPartyBanner}\n${noticeBanner(workbench.notices)}` },
  legalComments: "inline",
});
await Promise.all([
  copyFile(outfile, resolve(root, "plugin/arxiv-daily-cli.cjs")),
  copyFile(resolve(root, "arxiv_daily.py"), resolve(here, "dist/arxiv_daily.py")),
]);
