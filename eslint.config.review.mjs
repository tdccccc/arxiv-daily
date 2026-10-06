// Scanner-parity lint: approximates the checks Obsidian's hosted community
// plugin review runs (eslint-plugin-obsidianmd + typescript-eslint
// type-checked rules) over every production TypeScript source tree, not just
// the plugin package that `npm run lint` covers. This is diagnostic only —
// every rule here is "warn" on purpose, matching the hosted review's own
// severity and leaving `npm run lint`'s error-on-obsidian-rules behavior
// untouched. See `npm run lint:review`.
//
// Deliberately NOT reused from eslint.config.mjs: that file applies
// `obsidianmd.configs.recommended`, whose JS/TS blocks use repo-wide file
// globs (`**/*.{ts,...}`) and would require an elaborate ignore list to keep
// scoped to only the six production trees below, plus it pulls in several
// obsidianmd/plugin-authoring rules (commands/*, settings-tab/*, vault/*,
// sample code, platform checks, ...) that are meaningless outside an
// Obsidian Plugin subclass and that the hosted report never raised against
// this repo. Hand-picking rules with explicit `files` globs keeps the scope
// exact and the rule list traceable to the hosted report categories.
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";
import obsidianmd from "eslint-plugin-obsidianmd";

// One block per package: each tsconfig's own "include" already matches
// exactly the glob linted here (src only, no tests), so type-aware parsing
// can reuse the existing tsconfig as-is — no lint-only tsconfig needed.
const packages = [
  { name: "core", files: ["packages/core/src/**/*.ts"], project: "./packages/core/tsconfig.json" },
  { name: "node-runtime", files: ["packages/node-runtime/src/**/*.ts"], project: "./packages/node-runtime/tsconfig.json" },
  { name: "cli", files: ["apps/cli/src/**/*.ts"], project: "./apps/cli/tsconfig.json" },
  { name: "email-relay", files: ["services/email-relay/src/**/*.ts"], project: "./services/email-relay/tsconfig.json" },
  { name: "plugin", files: ["plugin/main.ts", "plugin/src/**/*.ts"], project: "./plugin/tsconfig.json" },
];

// typescript-eslint rules the hosted report's "Source code warnings" /
// "Recommendations" categories map onto. These are all "error" by default in
// typescript-eslint's recommendedTypeChecked config; downgraded to "warn"
// here to mirror the hosted report (which lists every item as a warning, not
// a build-breaking error) and to keep this script non-blocking.
const typeCheckedRules = {
  "@typescript-eslint/no-unsafe-argument": "warn",
  "@typescript-eslint/no-unsafe-assignment": "warn",
  "@typescript-eslint/no-unsafe-call": "warn",
  "@typescript-eslint/no-unsafe-member-access": "warn",
  "@typescript-eslint/no-unsafe-return": "warn",
  "@typescript-eslint/no-explicit-any": "warn",
  "@typescript-eslint/no-unnecessary-type-assertion": "warn",
  "@typescript-eslint/prefer-promise-reject-errors": "warn",
  "@typescript-eslint/only-throw-error": "warn",
  "@typescript-eslint/unbound-method": "warn",
  "@typescript-eslint/no-redundant-type-constituents": "warn",
  "@typescript-eslint/no-empty-object-type": "warn",
  "@typescript-eslint/no-this-alias": "warn",
  "@typescript-eslint/no-unused-vars": ["warn", { args: "none", ignoreRestSiblings: true }],
  "@typescript-eslint/no-deprecated": "warn",
  "@typescript-eslint/triple-slash-reference": "warn",
  // The base JS rules duplicate the typescript-eslint variants above; turn
  // them off so each finding is only counted once.
  "no-unused-vars": "off",
};

// Core ESLint rules (no type information needed) the hosted report raised.
const coreEslintRules = {
  "no-control-regex": "warn",
  "no-useless-escape": "warn",
  "no-irregular-whitespace": "warn",
  "no-constant-condition": "warn",
};

// eslint-plugin-obsidianmd rules the hosted report raised. Left out on
// purpose: commands/*, settings-tab/*, vault/iterate, detach-leaves,
// no-sample-code, platform, no-tfile-tfolder-cast, object-assign,
// prefer-file-manager-trash-file, prefer-instanceof, prefer-get-language,
// prefer-abstract-input-suggest, regex-lookbehind, sample-names,
// no-unsupported-api, no-view-references-in-plugin, no-plugin-as-component,
// no-static-styles-assignment, no-nodejs-modules, ui/sentence-case — these
// either require an Obsidian Plugin/View/Vault type context that shared
// core/node-runtime/CLI/worker code never has, or the hosted report never
// raised them for this repo. `obsidianmd/validate-manifest` is intentionally
// left to the existing root eslint.config.mjs (manifest.json is not part of
// "production TypeScript source").
const obsidianRules = {
  "obsidianmd/prefer-window-timers": "warn",
  "obsidianmd/no-global-this": "warn",
  "obsidianmd/prefer-create-el": "warn",
  "obsidianmd/hardcoded-config-path": "warn",
  // Re-labels a subset of `no-console` violations with Obsidian's reviewer
  // wording ("Avoid unnecessary logging to console..."); `no-console` itself
  // is left off to avoid double-reporting the same call site.
  "no-console": "off",
  "obsidianmd/rule-custom-message": [
    "warn",
    {
      "no-console": {
        messages: {
          "Unexpected console statement. Only these console methods are allowed: warn, error, debug.":
            "Avoid unnecessary logging to console. See https://docs.obsidian.md/Plugins/Releasing/Plugin+guidelines#Avoid+unnecessary+logging+to+console",
        },
        options: [{ allow: ["warn", "error", "debug"] }],
      },
    },
  ],
};

export default defineConfig([
  {
    name: "arxiv-daily/review-ignores",
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "**/.vitest-cache/**",
      "packages/node-runtime/native/**",
      "output/**",
      "tmp/**",
    ],
  },
  ...packages.map(({ name, files, project }) => ({
    name: `arxiv-daily/review-${name}`,
    files,
    plugins: { "@typescript-eslint": tseslint.plugin, obsidianmd },
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        project,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      ...coreEslintRules,
      ...typeCheckedRules,
      ...obsidianRules,
    },
  })),
]);
