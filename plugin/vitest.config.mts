import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFileSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));

const markdownAsText = {
  name: "markdown-as-text",
  enforce: "pre" as const,
  load(id: string) {
    const path = id.split("?")[0];
    if (path.endsWith(".md")) {
      return `export default ${JSON.stringify(readFileSync(path, "utf-8"))};`;
    }
    return null;
  },
};

export default defineConfig({
  plugins: [markdownAsText],
  test: {
    environment: "happy-dom",
    globals: false,
    include: ["tests/**/*.test.ts"],
    environmentOptions: {
      happyDOM: {
        settings: {
          disableJavaScriptEvaluation: true,
          disableJavaScriptFileLoading: true,
          disableCSSFileLoading: true,
        },
      },
    },
  },
  resolve: {
    alias: {
      "@": resolve(here, "src"),
      obsidian: resolve(here, "tests/__mocks__/obsidian.ts"),
    },
  },
});
