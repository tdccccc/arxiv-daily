import { build } from 'esbuild';
import { mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
await mkdir(resolve(here, 'dist'), { recursive: true });
await build({
  entryPoints: [resolve(here, 'src/agent.ts')],
  outfile: resolve(here, 'dist/agent.cjs'),
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node20',
  loader: { '.md': 'text' },
  tsconfig: resolve(root, 'tsconfig.base.json'),
  legalComments: 'inline',
});
