import { PLUGIN, ENDPOINT, isWorkbenchUrl } from './protocol.mjs';

export function createOpener(ctx, location) {
  const lifetime = new AbortController();
  let pending;
  async function launch() {
    if (lifetime.signal.aborted) return;
    const desktop = ['dsh-app:', 'file:'].includes(location.protocol);
    if (!desktop && !ctx.connection.isLoopback) throw new Error('请在本机 DSH 中打开文献工作台。');
    const session = ctx.sidebarRight.mounted?.getSnapshot();
    const result = await ctx.connection.rpc.call('/api', ENDPOINT, desktop ? {} : { frameOrigin: location.origin }, lifetime.signal);
    if (lifetime.signal.aborted || (session !== undefined && ctx.sidebarRight.mounted?.getSnapshot() !== session)) return;
    if (!result.ok) throw new Error(result.error?.message || '无法打开文献工作台，请重试。');
    if (!isWorkbenchUrl(result.value?.url)) throw new Error('工作台地址无效，请检查插件版本。');
    ctx.sidebarRight.openTab('browser', { params: { url: result.value.url } });
  }
  return {
    open() { if (!pending) pending = launch().finally(() => { pending = undefined; }); return pending; },
    dispose() { lifetime.abort(); },
  };
}

export function installClient(ctx, React, location) {
  ctx.effect(() => ctx.locale.register(PLUGIN, {
    zh: { open: '文献', opening: '正在打开…', title: '打开 arXiv Daily 文献工作台', retry: '重试' },
    en: { open: 'Papers', opening: 'Opening…', title: 'Open arXiv Daily research workbench', retry: 'Retry' },
  }));
  const t = ctx.locale.bind(PLUGIN);
  const h = React.createElement;
  function LiteratureButton() {
    const opener = React.useRef(null);
    const [busy, setBusy] = React.useState(false), [error, setError] = React.useState('');
    const alive = React.useRef(true);
    React.useEffect(() => {
      alive.current = true; const current = createOpener(ctx, location); opener.current = current;
      return () => { alive.current = false; current.dispose(); };
    }, []);
    const open = async () => {
      setBusy(true); setError('');
      try { await opener.current.open(); } catch (error) { if (alive.current) setError(error.message); }
      finally { if (alive.current) setBusy(false); }
    };
    return h('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px', flexWrap: 'wrap', width: '100%', fontSize: '12px' } },
      error ? h('span', { role: 'alert', style: { maxWidth: '380px', overflowWrap: 'anywhere' } }, error) : null,
      h('button', { type: 'button', disabled: busy, onClick: open, title: t('title'), 'aria-label': t('title'), style: { color: 'inherit', font: 'inherit', cursor: busy ? 'wait' : 'pointer', padding: '5px 10px', borderRadius: '6px', border: '1px solid currentColor', background: 'transparent', opacity: busy ? .6 : 1 } }, busy ? t('opening') : error ? t('retry') : `▤ ${t('open')}`));
  }
  ctx.effect(() => ctx.slots.inject('conversation.composer.dock', () => ctx.slots.register({ name: 'conversation.composer.dock', id: PLUGIN, order: 30, locale: PLUGIN }, LiteratureButton)));
}
