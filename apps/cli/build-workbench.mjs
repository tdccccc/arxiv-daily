import esbuild from "esbuild";
import { readFile, readdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";

/** Produce a self-contained client: no CDN, source checkout, or runtime node_modules required. */
export async function buildWorkbenchAssets(here) {
  const web = resolve(here, "src/workbench/web");
  const build = await esbuild.build({ entryPoints: [resolve(web, "app.ts")], bundle: true, platform: "browser", format: "iife", target: "es2022", minify: true, write: false });
  const require = createRequire(import.meta.url);
  const katexDir = dirname(require.resolve("katex"));
  const assets = {
    "index.html": { type: "text/html; charset=utf-8", body: await readFile(resolve(web, "index.html"), "utf8") },
    "app.js": { type: "text/javascript; charset=utf-8", body: build.outputFiles[0].text },
    "style.css": { type: "text/css; charset=utf-8", body: (await Promise.all(["style.css", "papers.css", "sidebar.css", "settings.css"].map(name => readFile(resolve(web, name), "utf8")))).join("\n") },
    "katex.css": { type: "text/css; charset=utf-8", body: await readFile(resolve(katexDir, "katex.min.css"), "utf8") },
  };
  for (const name of await readdir(resolve(katexDir, "fonts"))) {
    if (!/\.(woff2?|ttf)$/.test(name)) continue;
    assets[`fonts/${name}`] = { type: name.endsWith(".woff2") ? "font/woff2" : name.endsWith(".woff") ? "font/woff" : "font/ttf", body: (await readFile(resolve(katexDir, "fonts", name))).toString("base64"), encoding: "base64" };
  }
  const notices = await readFile(resolve(here, "../../THIRD_PARTY_NOTICES.md"), "utf8");
  assets["third-party-notices.txt"] = { type: "text/plain; charset=utf-8", body: notices };
  return { assets, notices };
}
