import { afterEach, expect, it, vi } from "vitest";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { prepareNodeLibraryRuntime } from "../src/library-runtime";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function root() { const path = await mkdtemp(join(tmpdir(), "library-runtime-")); roots.push(path); return path; }
async function installFixture(directory: string) {
  const spec = JSON.parse(await readFile(join(directory, "package.json"), "utf8"));
  for (const [name, version] of Object.entries(spec.dependencies)) {
    const target = join(directory, "node_modules", name);
    await mkdir(target, { recursive: true });
    await writeFile(join(target, "package.json"), JSON.stringify({ name, version, main: "index.js" }));
    await writeFile(join(target, "index.js"), "module.exports = {};\n");
  }
}

it("prepares isolated, versioned runtime dependencies and reuses verified installations", async () => {
  const cacheDir = await root();
  const install = vi.fn(installFixture);
  const result = await prepareNodeLibraryRuntime({ cacheDir, localEmbedding: true, install });
  expect(result.pdf.startsWith(cacheDir)).toBe(true);
  expect(result.embedding?.startsWith(cacheDir)).toBe(true);
  expect(install).toHaveBeenCalledTimes(2);
  expect(JSON.parse(await readFile(join(result.pdf, "package.json"), "utf8")).dependencies["pdfjs-dist"]).toBe("5.4.624");
  await prepareNodeLibraryRuntime({ cacheDir, localEmbedding: true, install });
  expect(install).toHaveBeenCalledTimes(2);
});

it("does not install the CPU embedding component for remote embedding", async () => {
  const install = vi.fn(installFixture);
  const result = await prepareNodeLibraryRuntime({ cacheDir: await root(), localEmbedding: false, install });
  expect(install).toHaveBeenCalledTimes(1);
  expect(result.embedding).toBeUndefined();
});

it("rejects incomplete installation rather than marking it ready", async () => {
  await expect(prepareNodeLibraryRuntime({ cacheDir: await root(), localEmbedding: false, install: async () => {} }))
    .rejects.toThrow(/incomplete|missing/i);
});
