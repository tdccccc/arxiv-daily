import esbuild from "esbuild";
import { resolve } from "node:path";
import { noticeBanner, readPakoNotice } from "../scripts/release-utils.mjs";

const prod = process.argv[2] === "production";
const thirdPartyBanner = noticeBanner(await readPakoNotice());
const options = {
  entryPoints: [resolve(import.meta.dirname, "main.ts")],
  outfile: resolve(import.meta.dirname, "main.js"),
  bundle: true,
  format: "cjs",
  target: "es2022",
  platform: "node",
  external: [
    "obsidian",
    "electron",
    "node:fs",
    "node:fs/promises",
    "node:path",
  ],
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  minify: prod,
  loader: { ".md": "text" },
  banner: { js: thirdPartyBanner },
  legalComments: "inline",
};

if (prod) await esbuild.build(options);
else {
  const context = await esbuild.context(options);
  await context.watch();
}
