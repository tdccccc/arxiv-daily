import { readFile } from "node:fs/promises";
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
assert(pkg.main === "./src/extension.js", "main must point to src/extension.js");
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

const source = await readFile(mainPath, "utf8");
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
