import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFileSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
export default defineConfig({
  plugins: [{
    name: "markdown-as-text",
    enforce: "pre",
    load(id) {
      const path = id.split("?")[0];
      return path.endsWith(".md")
        ? `export default ${JSON.stringify(readFileSync(path, "utf8"))};`
        : null;
    },
  }],
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.ts"],
    isolate: true,
    restoreMocks: true,
    environmentOptions: { happyDOM: { settings: {
      disableJavaScriptEvaluation: true,
      disableJavaScriptFileLoading: true,
      disableCSSFileLoading: true,
    } } },
  },
  resolve: { alias: {
    obsidian: resolve(here, "tests/__mocks__/obsidian.ts"),
    "@arxiv-daily/core": resolve(here, "../packages/core/src/index.ts"),
  } },
});
