import { readdir, readFile, stat } from "node:fs/promises";
import { builtinModules } from "node:module";
import { relative, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const layers = [
  { dir: "packages/core", workspace: new Set(), thirdParty: new Set(["pako"]) },
  { dir: "packages/node-runtime", workspace: new Set(["@arxiv-daily/core"]) },
  { dir: "apps/cli", workspace: new Set(["@arxiv-daily/core", "@arxiv-daily/node-runtime"]) },
  { dir: "plugin", workspace: new Set(["@arxiv-daily/core"]), thirdParty: new Set(["obsidian"]) },
];
const forbiddenOldPaths = [
  "plugin/src/core", "plugin/src/llm", "plugin/src/pipeline", "plugin/src/prompts",
  "plugin/src/utils", "plugin/src/cli", "plugin/src/hosts/node",
];
const builtins = new Set([...builtinModules, ...builtinModules.map((name) => `node:${name}`)]);
const errors = [];

for (const oldPath of forbiddenOldPaths) {
  if (await exists(resolve(root, oldPath))) errors.push(`legacy duplicate path exists: ${oldPath}`);
}
for (const layer of layers) {
  for (const file of await filesUnder(resolve(root, layer.dir))) {
    if (!/\.(?:ts|mts|mjs)$/.test(file)) continue;
    const source = await readFile(file, "utf8");
    const display = relative(root, file);
    if (layer.dir === "packages/core" && file.includes("/src/") && /\b(?:process|Buffer)\b/.test(source)) {
      errors.push(`${display}: core uses forbidden Node global`);
    }
    for (const specifier of moduleSpecifiers(source)) {
      if (specifier.startsWith(".") || specifier.startsWith("/")) continue;
      const owner = packageOwner(specifier);
      if (owner.startsWith("@arxiv-daily/")) {
        if (!layer.workspace.has(owner)) errors.push(`${display}: forbidden dependency ${specifier}`);
        if (specifier !== owner) errors.push(`${display}: deep workspace import ${specifier}`);
        continue;
      }
      if (layer.dir === "packages/core" && file.includes("/src/") && !layer.thirdParty.has(owner)) {
        errors.push(`${display}: core third-party dependency is not allowlisted: ${specifier}`);
      }
      if (layer.dir === "packages/core" && file.includes("/src/") && builtins.has(specifier)) {
        errors.push(`${display}: core imports Node builtin ${specifier}`);
      }
      if (layer.dir === "plugin" && file.includes("/src/") && builtins.has(specifier)) {
        errors.push(`${display}: plugin imports Node builtin ${specifier}`);
      }
    }
  }
}
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log("Workspace boundaries OK");

function moduleSpecifiers(source) {
  const specs = [];
  const pattern = /(?:\b(?:import|export)\s+(?:type\s+)?(?:[^"'()]*?\s+from\s+)?|\brequire\s*\(|\bimport\s*\()\s*["']([^"']+)["']/g;
  for (const match of source.matchAll(pattern)) if (match[1]) specs.push(match[1]);
  return specs;
}
function packageOwner(specifier) {
  return specifier.startsWith("@") ? specifier.split("/").slice(0, 2).join("/") : specifier.split("/")[0];
}
async function exists(path) { try { await stat(path); return true; } catch { return false; } }
async function filesUnder(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (["node_modules", "dist"].includes(entry.name)) continue;
    const path = resolve(dir, entry.name);
    if (entry.isDirectory()) out.push(...await filesUnder(path));
    else out.push(path);
  }
  return out;
}
