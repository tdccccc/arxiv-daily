import { readFileSync } from "node:fs";
import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

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
  resolve: {
    alias: { "@arxiv-daily/core": resolve(here, "../../packages/core/src/index.ts") },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    isolate: true,
    restoreMocks: true,
  },
});
