import { defineConfig } from "vitest/config";
import { readFileSync } from "node:fs";

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
    environment: "node",
    include: ["tests/**/*.test.ts"],
    isolate: true,
    restoreMocks: true,
  },
});
