import esbuild from "esbuild";
import process from "process";
import builtins from "builtin-modules";

const prod = process.argv[2] === "production";

const externalBuiltins = [...builtins, ...builtins.map((b) => `node:${b}`)];

const common = {
  bundle: true,
  format: "cjs",
  target: "es2020",
  platform: "node",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  minify: prod,
};

const contexts = await Promise.all([
  esbuild.context({
    ...common,
    entryPoints: ["main.ts"],
    external: ["obsidian", "electron", ...externalBuiltins],
    outfile: "main.js",
  }),
  esbuild.context({
    ...common,
    entryPoints: ["src/cli/main.ts"],
    external: [...externalBuiltins],
    outfile: "arxiv-daily-cli.cjs",
    banner: { js: "#!/usr/bin/env node" },
  }),
]);

if (prod) {
  for (const ctx of contexts) await ctx.rebuild();
  for (const ctx of contexts) await ctx.dispose();
} else {
  for (const ctx of contexts) await ctx.watch();
}
