import js from "@eslint/js";
import json from "@eslint/json";
import { defineConfig } from "eslint/config";
import obsidianmd from "eslint-plugin-obsidianmd";
import { PlainTextParser } from "eslint-plugin-obsidianmd/dist/lib/plainTextParser.js";

const disabledJavaScriptRules = Object.fromEntries(
  Object.keys(js.configs.recommended.rules).map((rule) => [rule, "off"]),
);

export default defineConfig([
  {
    name: "arxiv-daily/non-production-files",
    ignores: [
      "apps/**",
      "extensions/**",
      "node_modules/**",
      "packages/**",
      "plugin/.vitest-cache/**",
      "plugin/*.cjs",
      "plugin/*.js",
      "plugin/tests/**",
      "scripts/**",
    ],
  },
  ...obsidianmd.configs.recommended,
  {
    name: "arxiv-daily/production-plugin",
    files: ["plugin/main.ts", "plugin/src/**/*.ts"],
    languageOptions: {
      parserOptions: {
        project: "./plugin/tsconfig.json",
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    name: "arxiv-daily/manifest",
    files: ["manifest.json"],
    language: "json/json",
    plugins: { json, obsidianmd },
    rules: {
      ...disabledJavaScriptRules,
      "obsidianmd/validate-manifest": "warn",
    },
  },
  {
    name: "arxiv-daily/license",
    files: ["LICENSE"],
    languageOptions: {
      parser: PlainTextParser,
    },
    plugins: { obsidianmd },
    rules: {
      ...disabledJavaScriptRules,
      "obsidianmd/validate-license": "warn",
    },
  },
]);
