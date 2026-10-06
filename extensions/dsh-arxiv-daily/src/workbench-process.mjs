import { spawn as spawnProcess } from 'node:child_process';
import { isWorkbenchUrl } from './protocol.mjs';

export function failure(code, message) { return Object.assign(new Error(message), { code }); }

/** Product process boundary; no DSH APIs, prompt execution or second data store. */
export class WorkbenchProcess {
  constructor({ cliPath, env = process.env, executable = process.execPath, spawn = spawnProcess, startupTimeoutMs = 15000, stopTimeoutMs = 5000 }) {
    this.cliPath = cliPath; this.env = { ...env }; this.executable = executable; this.spawn = spawn;
    this.startupTimeoutMs = startupTimeoutMs; this.stopTimeoutMs = stopTimeoutMs;
    this.current = null; this.disposed = false;
  }
  async open(frameOrigin) {
    if (this.disposed) throw failure('stopped', '插件已卸载，请重新启用。');
    if (this.current) {
      if (this.current.frameOrigin !== frameOrigin) throw failure('origin-changed', '工作台已从另一个 DSH 地址打开，请使用原地址或重新启用插件。');
      return this.current.ready;
    }
    const args = [this.cliPath, 'ui', '--no-open', ...(frameOrigin ? ['--frame-origin', frameOrigin] : [])];
    const env = { ...this.env, ...(process.versions.electron ? { ELECTRON_RUN_AS_NODE: '1' } : {}) };
    let child;
    try { child = this.spawn(this.executable, args, { env, shell: false, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true, detached: process.platform !== 'win32' }); }
    catch { throw failure('start-failed', '无法启动文献工作台，请检查插件安装和 Node 运行环境。'); }
    const state = { child, frameOrigin, ready: null, exited: null, reject: null, timer: null, settled: false, output: '', diagnostic: '' };
    this.current = state;
    state.ready = new Promise((resolve, reject) => {
      state.reject = error => { if (!state.settled) { state.settled = true; clearTimeout(state.timer); reject(error); } };
      state.timer = setTimeout(() => {
        state.reject(failure('start-timeout', '文献工作台启动超时，请重试。'));
        void this.stop(state);
      }, this.startupTimeoutMs);
      child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
      child.stdout.on('data', chunk => {
        if (state.settled) return;
        state.output = (state.output + chunk).slice(-16384);
        const match = /(?:^|\n)Workbench: ([^\r\n]+)\r?\n/.exec(state.output);
        if (!match) return;
        if (!isWorkbenchUrl(match[1])) {
          state.reject(failure('invalid-address', '工作台返回了无效地址，请检查插件版本。')); void this.stop(state); return;
        }
        state.settled = true; clearTimeout(state.timer); state.output = ''; state.diagnostic = ''; resolve(match[1]);
      });
      child.stderr.on('data', chunk => { if (!state.settled) state.diagnostic = (state.diagnostic + chunk).slice(-16384); });
      child.once('error', () => { state.reject(failure('start-failed', '无法启动文献工作台，请检查插件安装和 Node 运行环境。')); });
    });
    state.exited = new Promise(resolve => child.once('close', () => {
      const missing = state.diagnostic.includes('CLI config not found');
      state.reject(failure(missing ? 'setup-required' : 'start-failed', missing ? '尚未完成 arXiv Daily 首次设置，请按插件说明运行 init 后重试。' : '文献工作台已退出，请检查 CLI 配置后重试。'));
      state.diagnostic = ''; state.output = ''; clearTimeout(state.timer);
      if (this.current === state) this.current = null;
      resolve();
    }));
    return state.ready;
  }
  signal(state, signal) {
    if (state.child.exitCode !== null || state.child.signalCode !== null) return;
    try {
      if (process.platform !== 'win32' && state.child.pid) process.kill(-state.child.pid, signal);
      else state.child.kill(signal);
    } catch (error) { if (error.code !== 'ESRCH') state.child.kill(signal); }
  }
  async stop(state) {
    if (state.stopping) return state.stopping;
    state.stopping = (async () => {
      this.signal(state, 'SIGTERM');
      const timer = setTimeout(() => this.signal(state, 'SIGKILL'), this.stopTimeoutMs);
      try { await state.exited; } finally { clearTimeout(timer); }
    })();
    return state.stopping;
  }
  async dispose() {
    this.disposed = true;
    const state = this.current;
    if (!state) return;
    state.reject(failure('stopped', '插件已卸载，工作台已停止。'));
    await this.stop(state);
  }
}
