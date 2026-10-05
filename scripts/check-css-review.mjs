// Scanner-parity CSS check for npm run lint:review.
//
// Obsidian's hosted plugin review also lints the stylesheet for two patterns
// that have no ESLint equivalent (ESLint only parses JS/TS): `!important`
// (fights the theme/snippet cascade the user controls) and the `:has()`
// relational selector (unsupported in the Electron/Chromium version some
// supported Obsidian installs still ship). Rather than add a CSS-aware lint
// dependency for two literal-token checks, scan the stylesheet text directly.
// This is diagnostic only, like the rest of lint:review: it always exits 0.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const stylesheets = ["plugin/styles.css"];

const checks = [
  { pattern: /!important/g, rule: "css/no-important", message: "Avoid !important; it overrides the user's theme/snippet cascade." },
  { pattern: /:has\(/g, rule: "css/no-has", message: "Avoid the :has() relational selector; unsupported in some Obsidian installs' bundled Chromium." },
];

let total = 0;
for (const relativePath of stylesheets) {
  const path = resolve(root, relativePath);
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n");
  for (const { pattern, rule, message } of checks) {
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
      const line = lines[lineIndex];
      pattern.lastIndex = 0;
      let match;
      while ((match = pattern.exec(line))) {
        total += 1;
        console.log(`${relativePath}:${lineIndex + 1}:${match.index + 1}  warning  ${message}  ${rule}`);
      }
    }
  }
}

console.log(`\n${total} CSS warning${total === 1 ? "" : "s"} (${stylesheets.join(", ")})`);
