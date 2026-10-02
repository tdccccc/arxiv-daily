import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { nativeAssetsForTests } from "../../scripts/native-assets.mjs";

const here = dirname(fileURLToPath(import.meta.url));
export default defineConfig({
  define: { __ARXIV_DAILY_NATIVE_ASSETS__: JSON.stringify(nativeAssetsForTests()) },
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
  test: { environment: "node", include: ["tests/**/*.test.ts"], globalSetup: ["./native/test-setup.mjs"] },
  resolve: {
    alias: {
      "@arxiv-daily/core": resolve(here, "../core/src/index.ts"),
    },
  },
});
