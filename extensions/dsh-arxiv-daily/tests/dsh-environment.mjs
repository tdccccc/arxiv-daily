import { execFileSync } from 'node:child_process';
import { realpathSync } from 'node:fs';
export function installedDsh() {
  try {
    const cli = process.env.DSH_CLI || execFileSync(process.platform === 'win32' ? 'where.exe' : 'which', ['dsh'], { encoding: 'utf8' }).trim().split(/\r?\n/)[0];
    const file = realpathSync(cli);
    return { cli: file };
  } catch { return null; }
}
