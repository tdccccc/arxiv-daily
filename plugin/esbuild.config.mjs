import esbuild from "esbuild";
import { resolve } from "node:path";

const prod = process.argv[2] === "production";
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
};

if (prod) await esbuild.build(options);
else {
  const context = await esbuild.context(options);
  await context.watch();
}
