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
    globalSetup: ["../packages/node-runtime/native/test-setup.mjs"],
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
    "@arxiv-daily/node-runtime/private-storage": resolve(here, "../packages/node-runtime/src/native-private-storage.ts"),
    "@arxiv-daily/node-runtime/file-lock": resolve(here, "../packages/node-runtime/src/file-lock.ts"),
    "@arxiv-daily/node-runtime/scoped-library-source": resolve(
      here,
      "../packages/node-runtime/src/scoped-library-source.ts",
    ),
  } },
});
