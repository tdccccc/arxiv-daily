import Module, { createRequire } from "node:module";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const bundlePath = path.join(root, "out", "extension.js");

/**
 * Loads the built extension bundle (what the VSIX ships) with `vscode` stubbed:
 * the host module only exists inside VS Code, and the tests pass their own
 * mock API to the exported internals.
 */
export function loadExtensionBundle() {
  if (!existsSync(bundlePath)) {
    throw new Error("out/extension.js is missing; run `npm run build` first");
  }
  const originalLoad = Module._load;
  Module._load = function patchedLoad(request, ...rest) {
    if (request === "vscode") return {};
    return originalLoad.call(this, request, ...rest);
  };
  try {
    return createRequire(import.meta.url)(bundlePath);
  } finally {
    Module._load = originalLoad;
  }
}
