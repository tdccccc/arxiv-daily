import { readFile } from "node:fs/promises";
import { builtinModules } from "node:module";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packagePath = path.join(root, "package.json");
const pkg = JSON.parse(await readFile(packagePath, "utf8"));

const requiredCommands = [
  "arxivDaily.openDashboard",
  "arxivDaily.run",
  "arxivDaily.summarizeById",
];
const removedCommands = [
  "arxivDaily.runPending",
  "arxivDaily.configureApiKey",
];

assert(pkg.name === "arxiv-daily-vscode", "unexpected package name");
assert(pkg.main === "./out/extension.js", "main must point to the bundled out/extension.js");
assert(
  /^\d+\.\d+\.\d+$/.test(pkg.version),
  "version must use plain semver",
);
assert(
  pkg.engines?.vscode?.startsWith("^"),
  "engines.vscode must declare a compatible VS Code range",
);

const commands = pkg.contributes?.commands ?? [];
const commandIds = commands.map((command) => command.command);
assertSameMembers(commandIds, requiredCommands, "command contributions");
assertSameMembers(
  pkg.activationEvents,
  requiredCommands.map((command) => `onCommand:${command}`),
  "activation events",
);
for (const command of removedCommands) {
  assert(!commandIds.includes(command), `removed command contribution remains: ${command}`);
  assert(
    !pkg.activationEvents.includes(`onCommand:${command}`),
    `removed activation event remains: ${command}`,
  );
}
assertSameMembers(
  Object.keys(pkg.contributes?.configuration?.properties ?? {}),
  ["arxivDaily.cliPath"],
  "configuration properties",
);

const mainPath = path.join(root, pkg.main);
const readmePath = path.join(root, "README.md");
const ignorePath = path.join(root, ".vscodeignore");
assert(existsSync(mainPath), `missing extension entrypoint: ${pkg.main}`);
assert(existsSync(readmePath), "missing README.md");
assert(existsSync(ignorePath), "missing .vscodeignore");

const ignore = await readFile(ignorePath, "utf8");
const ignoreRules = ignore.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
// The VSIX must carry the bundle and not the sources or build tooling it came from.
for (const rule of ["src/**", "scripts/**", "node_modules/**"]) {
  assert(ignoreRules.includes(rule), `.vscodeignore must exclude ${rule}`);
}
assert(
  !ignoreRules.some((rule) => /^(?:\.\/)?out(?:\/\*\*?)?\/?$/.test(rule) || rule === "out/extension.js"),
  ".vscodeignore must not exclude the out/ bundle",
);
assert(
  pkg.scripts?.["vscode:prepublish"] === "npm run build",
  "vsce packaging must rebuild the bundle through vscode:prepublish",
);

const source = await readFile(mainPath, "utf8");
// Everything the extension needs at runtime is inside the bundle; only the
// host-provided `vscode` module may be required from outside it.
const externalRequires = [...source.matchAll(/\brequire\(\s*["']([^"']+)["']\s*\)/g)]
  .map((match) => match[1])
  .filter((name) => !name.startsWith("node:") && !builtinModules.includes(name));
assertSameMembers([...new Set(externalRequires)], ["vscode"], "non-builtin modules required by the bundle");
assert(
  !/(?:^|[^\w])(?:from\s*|require\()\s*["']@arxiv-daily\//.test(source),
  "bundle must inline @arxiv-daily packages",
);
for (const command of requiredCommands) {
  assert(
    source.includes(`registerCommand("${command}"`),
    `entrypoint does not register ${command}`,
  );
}
for (const command of removedCommands) {
  assert(
    !source.includes(`registerCommand("${command}"`),
    `entrypoint still registers ${command}`,
  );
}

console.log(`arXiv Daily VS Code scaffold OK (${requiredCommands.length} commands)`);

function assertSameMembers(actual, expected, label) {
  const actualSorted = [...actual].sort();
  const expectedSorted = [...expected].sort();
  assert(
    JSON.stringify(actualSorted) === JSON.stringify(expectedSorted),
    `${label} differ: expected ${expectedSorted.join(", ")}; received ${actualSorted.join(", ")}`,
  );
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}
