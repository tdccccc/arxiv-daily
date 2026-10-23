import { readdir, readFile, stat } from "node:fs/promises";
import { builtinModules } from "node:module";
import path, { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const defaultRoot = resolve(import.meta.dirname, "..");
const layers = [
  { dir: "packages/core", workspace: new Set(), thirdParty: new Set(["pako"]) },
  { dir: "packages/node-runtime", workspace: new Set(["@arxiv-daily/core"]) },
  { dir: "apps/cli", workspace: new Set(["@arxiv-daily/core", "@arxiv-daily/node-runtime"]) },
  {
    dir: "plugin",
    workspace: new Set(["@arxiv-daily/core", "@arxiv-daily/node-runtime"]),
    workspaceSubpaths: new Set([
      "@arxiv-daily/node-runtime/scoped-library-source",
      "@arxiv-daily/node-runtime/file-lock",
      "@arxiv-daily/node-runtime/private-storage",
    ]),
    thirdParty: new Set(["obsidian"]),
  },
];
const forbiddenOldPaths = [
  "plugin/src/core", "plugin/src/llm", "plugin/src/pipeline", "plugin/src/prompts",
  "plugin/src/utils", "plugin/src/cli", "plugin/src/hosts/node",
];
const builtins = new Set([...builtinModules, ...builtinModules.map((name) => `node:${name}`)]);

// The plugin must stay loadable on Obsidian mobile, so no production plugin file
// may import a Node builtin except these, each with the reason it is allowed.
// Keys are repository-relative paths with "/" separators.
export const PLUGIN_NODE_BUILTIN_ALLOWLIST = new Map([
  [
    "plugin/node-fs-exclusive.ts",
    {
      // Desktop-only exclusive-create storage guard; the plugin loads it behind a desktop check.
      specifiers: new Set(["node:fs", "node:fs/promises", "node:path"]),
    },
  ],
]);

/** Repository-relative path with "/" separators, so the same rules apply on Windows. */
export function displayPath(root, file, pathModule = path) {
  return pathModule.relative(root, file).split(pathModule.sep).join("/");
}

/** Production plugin TypeScript: everything under plugin/ except tests, tool configs and build output. */
export function isPluginProductionFile(display) {
  return display.startsWith("plugin/")
    && !display.startsWith("plugin/tests/")
    && !/(?:^|\/)[^/]+\.config\.[cm]?[jt]s$/.test(display)
    && !/\.d\.ts$/.test(display);
}

// These are deliberate non-product integrations or bundled runtime manifests, not general layer escapes.
const FILE_IMPORT_ALLOWLIST = new Map([
  // Build configuration and integration tests intentionally exercise other layers.
  ["packages/core/tests/real-corpus-migration.test.ts", new Set(["packages/node-runtime/src/index"])],
  ["packages/node-runtime/vitest.config.mts", new Set(["scripts/native-assets.mjs"])],
  ["apps/cli/esbuild.config.mjs", new Set(["scripts/release-utils.mjs", "scripts/native-assets.mjs"])],
  ["plugin/esbuild.config.mjs", new Set(["scripts/release-utils.mjs", "scripts/native-assets.mjs"])],
  ["plugin/vitest.config.mts", new Set(["scripts/native-assets.mjs"])],
  ["plugin/tests/obsidian-adapters.test.ts", new Set(["packages/core/src/settings/topics", "packages/node-runtime/src/native-private-storage"])],
  ["packages/node-runtime/src/library-runtime.ts", new Set([
    "tools/node-library-runtime/pdf/package.json", "tools/node-library-runtime/pdf/package-lock.json",
    "tools/node-library-runtime/embedding/package.json", "tools/node-library-runtime/embedding/package-lock.json",
  ])],
  ["packages/node-runtime/native/test-setup.mjs", new Set(["scripts/native-build.mjs"])],
  ["apps/cli/tests/workbench-business-settings.test.ts", new Set(["plugin/src/settings/change-service"])],
  ["apps/cli/tests/workbench-settings-ui.test.ts", new Set(["plugin/src/settings/definitions"])],
]);

function fileImportCrossesLayer(display, specifier, layer) {
  const portable = specifier.replaceAll("\\", "/");
  // Absolute imports have no relocatable workspace contract, even if this machine can resolve them.
  if (path.posix.isAbsolute(portable) || path.win32.isAbsolute(specifier)) return true;
  const target = path.posix.normalize(path.posix.join(path.posix.dirname(display), portable));
  if (target === layer.dir || target.startsWith(`${layer.dir}/`)) return false;
  return !FILE_IMPORT_ALLOWLIST.get(display)?.has(target);
}

export async function checkBoundaries(root = defaultRoot) {
const errors = [];

for (const oldPath of forbiddenOldPaths) {
  if (await exists(resolve(root, oldPath))) errors.push(`legacy duplicate path exists: ${oldPath}`);
}
for (const layer of layers) {
  for (const file of await filesUnder(resolve(root, layer.dir))) {
    if (!/\.(?:ts|mts|mjs)$/.test(file)) continue;
    const source = await readFile(file, "utf8");
    const display = displayPath(root, file);
    const coreSource = layer.dir === "packages/core" && display.startsWith("packages/core/src/");
    const pluginProduction = layer.dir === "plugin" && isPluginProductionFile(display);
    if (coreSource && /\b(?:process|Buffer)\b/.test(source)) {
      errors.push(`${display}: core uses forbidden Node global`);
    }
    for (const specifier of moduleSpecifiers(source)) {
      if (specifier.startsWith(".") || specifier.startsWith("/") || path.win32.isAbsolute(specifier)) {
        if (fileImportCrossesLayer(display, specifier, layer)) errors.push(`${display}: file import crosses workspace boundary: ${specifier}`);
        continue;
      }
      const owner = packageOwner(specifier);
      if (owner.startsWith("@arxiv-daily/")) {
        if (!layer.workspace.has(owner)) errors.push(`${display}: forbidden dependency ${specifier}`);
        if (specifier !== owner && !layer.workspaceSubpaths?.has(specifier)) {
          errors.push(`${display}: deep workspace import ${specifier}`);
        }
        if (
          layer.dir === "plugin"
          && owner === "@arxiv-daily/node-runtime"
          && specifier === owner
        ) {
          errors.push(`${display}: broad node-runtime import is forbidden`);
        }
        continue;
      }
      if (coreSource && !layer.thirdParty.has(owner)) {
        errors.push(`${display}: core third-party dependency is not allowlisted: ${specifier}`);
      }
      if (coreSource && builtins.has(specifier)) {
        errors.push(`${display}: core imports Node builtin ${specifier}`);
      }
      if (pluginProduction && builtins.has(specifier)) {
        const allowed = PLUGIN_NODE_BUILTIN_ALLOWLIST.get(display);
        if (!allowed?.specifiers.has(specifier)) errors.push(`${display}: plugin imports Node builtin ${specifier}`);
      }
    }
  }
}
return errors;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const errors = await checkBoundaries();
  if (errors.length) {
    console.error(errors.join("\n"));
    process.exit(1);
  }
  console.log("Workspace boundaries OK");
}

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
