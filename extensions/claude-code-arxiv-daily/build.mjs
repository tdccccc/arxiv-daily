import { build } from 'esbuild';
import { copyFile, cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { resolve, dirname, relative, isAbsolute, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
await mkdir(resolve(here, 'dist'), { recursive: true });
const result = await build({
  absWorkingDir: root,
  entryPoints: { agent: resolve(here, 'src/agent.ts'), 'arxiv-agent': resolve(here, 'src/main.ts') },
  outdir: resolve(here, 'dist'),
  outExtension: { '.js': '.cjs' },
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node20',
  loader: { '.md': 'text' },
  tsconfig: resolve(root, 'tsconfig.base.json'),
  legalComments: 'inline',
  metafile: true,
});
for (const input of Object.keys(result.metafile.inputs)) {
  const absolute = resolve(root, input);
  const local = relative(root, absolute);
  const outside = isAbsolute(local) || local === '..' || local.startsWith(`..${sep}`);
  if (outside && !/[\\/]node_modules[\\/]/.test(absolute)) {
    throw new Error(`Build escaped the worktree: ${absolute}`);
  }
}
await writeFile(resolve(here, 'dist/build-meta.json'), JSON.stringify(result.metafile, null, 2));
const packed = resolve(here, 'dist/plugin');
await rm(packed, { recursive: true, force: true });
await mkdir(resolve(packed, 'dist'), { recursive: true });
for (const item of ['.claude-plugin', 'skills', 'references', 'README.md']) {
  await cp(resolve(here, item), resolve(packed, item), { recursive: true });
}
await copyFile(resolve(here, 'dist/arxiv-agent.cjs'), resolve(packed, 'dist/arxiv-agent.cjs'));
await copyFile(resolve(root, 'LICENSE'), resolve(packed, 'LICENSE'));
