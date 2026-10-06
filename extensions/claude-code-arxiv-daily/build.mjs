import { copyFile, cp, mkdir, rm } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
// Reuse the official build entry, including native storage, version, and notices.
// Do not import runCli into a second bundled entry: it already owns process startup.
await import(pathToFileURL(resolve(root, 'apps/cli/esbuild.config.mjs')).href);
const dist = resolve(here, 'dist');
await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
await copyFile(resolve(root, 'apps/cli/dist/arxiv-daily-cli.cjs'), resolve(dist, 'arxiv-daily-cli.cjs'));
const packed = resolve(dist, 'plugin');
await mkdir(resolve(packed, 'dist'), { recursive: true });
for (const item of ['.claude-plugin', 'skills', 'references', 'README.md']) {
  await cp(resolve(here, item), resolve(packed, item), { recursive: true });
}
await copyFile(resolve(dist, 'arxiv-daily-cli.cjs'), resolve(packed, 'dist/arxiv-daily-cli.cjs'));
for (const name of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) await copyFile(resolve(root, name), resolve(packed, name));
