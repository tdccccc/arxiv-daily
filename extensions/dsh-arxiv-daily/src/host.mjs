import { fileURLToPath } from 'node:url';
import { WorkbenchProcess } from './workbench-process.mjs';
import { ENDPOINT } from './protocol.mjs';
export const name = 'dsh-arxiv-daily';
export const inject = ['connection', 'webServer'];
const errorCodes = new Set(['setup-required', 'start-failed', 'start-timeout', 'invalid-address', 'origin-changed', 'stopped']);

/** Startup is reachable only through DSH's authenticated Connection registry. */
export function installHost(ctx, manager) {
  ctx.effect(() => {
    const remove = ctx.connection.rpc.intercept('/api', endpoint => endpoint === ENDPOINT, async (_endpoint, body, signal) => {
      const bad = () => ({ ok: false, error: { code: 'invalid-request', message: '请选择本机 DSH 工作区打开文献。', details: {} } });
      if (!body || typeof body !== 'object' || Array.isArray(body) || Object.keys(body).some(key => key !== 'frameOrigin') || signal.aborted) return bad();
      const frameOrigin = body.frameOrigin;
      if (frameOrigin !== undefined && !['127.0.0.1', 'localhost', '[::1]'].map(host => `http://${host}:${ctx.webServer.port}`).includes(frameOrigin)) return bad();
      try { return { ok: true, value: { url: await manager.open(frameOrigin) } }; }
      catch (error) {
        return { ok: false, error: { code: errorCodes.has(error.code) ? error.code : 'start-failed', message: errorCodes.has(error.code) ? error.message : '无法打开文献工作台，请检查配置并重试。', details: {} } };
      }
    });
    return async () => { try { await remove(); } finally { await manager.dispose(); } };
  });
}
export function apply(ctx) {
  // The build places the exact CLI beside this Host entry, independent of CWD.
  installHost(ctx, new WorkbenchProcess({ cliPath: fileURLToPath(new URL('./arxiv-daily-cli.cjs', import.meta.url)) }));
}
