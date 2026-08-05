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
  external: ["obsidian", "electron"],
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  minify: prod,
  loader: { ".md": "text" },
  banner: { js: thirdPartyBanner },
  legalComments: "inline",
  // transformers.js: bundle its web build (Electron/Chromium renderer), not
  // the Node build that its exports map would select under platform "node"
  // (the Node build requires onnxruntime-node + sharp native binaries and
  // cannot be bundled for Obsidian).
  alias: {
    // npm workspaces hoists dependencies to the repo-root node_modules.
    "@huggingface/transformers": resolve(
      import.meta.dirname,
      "../node_modules/@huggingface/transformers/dist/transformers.web.min.js",
    ),
  },
  // onnxruntime-web: resolve to the extern-wasm build so no wasm asset is
  // emitted into the plugin folder. The inlined-wasm "bundle" build is
  // unusable under CJS output — its `new URL("...wasm", import.meta.url)`
  // asset references resolve to an empty import.meta and throw at runtime.
  // With the extern build, transformers.js fetches the wasm binaries from
  // its CDN default on first use and caches them via the Cache API.
  conditions: ["onnxruntime-web-use-extern-wasm"],
};

if (prod) await esbuild.build(options);
else {
  const context = await esbuild.context(options);
  await context.watch();
}
