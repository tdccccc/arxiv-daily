import { PLUGIN, ENDPOINT, isWorkbenchUrl } from './protocol.mjs';

/** Resolve the shared workbench independently of any mounted conversation. */
export function createOpener(ctx, location) {
  const lifetime = new AbortController(); let pending;
  async function launch() {
    if (lifetime.signal.aborted) return;
    const desktop = location.protocol === 'dsh-app:';
    if (!desktop && (!ctx.connection.isLoopback || !['http:', 'https:'].includes(location.protocol))) throw new Error('请在本机 DSH 中打开文献工作台。');
    const result = await ctx.connection.rpc.call('/api', ENDPOINT, { frameOrigin: desktop ? 'dsh-app://app' : location.origin }, lifetime.signal);
    if (lifetime.signal.aborted) return;
    if (!result.ok) throw new Error(result.error?.message || '无法打开文献工作台，请重试。');
    if (!isWorkbenchUrl(result.value?.url)) throw new Error('工作台地址无效，请检查插件版本。');
    return result.value.url;
  }
  return {
    open() { if (!pending) pending = launch().finally(() => { pending = undefined; }); return pending; },
    dispose() { lifetime.abort(); },
  };
}

export function installClient(ctx, React, location) {
  ctx.effect(() => ctx.locale.register(PLUGIN, {
    zh: { open: 'arxiv-daily', title: 'arxiv-daily', description: '筛选论文、阅读日报与详细总结', opening: '正在打开…', retry: '重试', refresh: '刷新', close: '关闭 arxiv-daily', external: '在浏览器打开', frame: 'arxiv-daily' },
    en: { open: 'arxiv-daily', title: 'arxiv-daily', description: 'Filter papers and read reports and paper notes', opening: 'Opening…', retry: 'Retry', refresh: 'Refresh', close: 'Close arxiv-daily', external: 'Open in browser', frame: 'arxiv-daily' },
  }));
  const t = ctx.locale.bind(PLUGIN), h = React.createElement;
  let shown = false;
  const listeners = new Set();
  const overlay = { get: () => shown, subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }, set(value) { shown = value; listeners.forEach(fn => fn()); } };
  const buttonStyle = { color: 'inherit', font: 'inherit', cursor: 'pointer', padding: '6px 10px', borderRadius: '6px', border: 'none', background: 'transparent' };
  function Icon({ size = 18, className } = {}) {
    return h('svg', { width: size, height: size, className, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, 'aria-hidden': true }, h('path', { d: 'M5 3h14v18H5zM8 7h8M8 11h8M8 15h5' }));
  }
  function WorkbenchFrame() {
    const [attempt, retry] = React.useState(0), [state, setState] = React.useState({ loading: true, url: '', error: '' });
    React.useEffect(() => {
      let active = true; const opener = createOpener(ctx, location);
      setState({ loading: true, url: '', error: '' });
      opener.open().then(url => { if (active && url) setState({ loading: false, url, error: '' }); }).catch(error => {
        if (active) setState({ loading: false, url: '', error: error instanceof Error ? error.message : 'Unable to open workbench' });
      });
      return () => { active = false; opener.dispose(); };
    }, [attempt]);
    return h('div', { style: { height: '100%', minHeight: 0, display: 'flex', flexDirection: 'column', color: 'var(--dsw-alias-label-primary)', background: 'var(--dsw-alias-bg-base, white)' } },
      h('div', { style: { display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '8px', padding: '4px 8px', flexShrink: 0 } },
        state.url ? h('a', { href: state.url, target: '_blank', rel: 'noopener noreferrer', style: { color: 'inherit', fontSize: '12px' } }, t('external')) : null,
        h('button', { type: 'button', disabled: state.loading, onClick: () => retry(value => value + 1), style: buttonStyle }, t(state.error ? 'retry' : 'refresh'))),
      state.loading ? h('p', { role: 'status', style: { padding: '16px' } }, t('opening')) : null,
      state.error ? h('p', { role: 'alert', style: { padding: '16px', overflowWrap: 'anywhere' } }, state.error) : null,
      state.url ? h('iframe', { title: t('frame'), src: state.url, referrerPolicy: 'no-referrer', sandbox: 'allow-scripts allow-forms allow-same-origin allow-popups allow-popups-to-escape-sandbox', style: { width: '100%', flex: 1, minHeight: 0, border: 0 } }) : null);
  }
  function Footer({ wide } = {}) {
    return h('button', { type: 'button', title: t('title'), 'aria-label': t('title'), onClick: () => overlay.set(true), style: { ...buttonStyle, width: '100%', display: 'flex', alignItems: 'center', justifyContent: wide ? 'flex-start' : 'center', gap: '10px', minHeight: '40px' } }, h(Icon), wide ? h('span', null, t('title')) : null);
  }
  function GlobalWorkbench() {
    const open = React.useSyncExternalStore(overlay.subscribe, overlay.get);
    const closeButton = React.useRef(null);
    React.useEffect(() => {
      if (!open) return;
      const previous = document.activeElement;
      const key = event => { if (event.key === 'Escape') { event.preventDefault(); overlay.set(false); } };
      window.addEventListener('keydown', key); closeButton.current?.focus();
      return () => { window.removeEventListener('keydown', key); if (previous?.isConnected) previous.focus?.(); };
    }, [open]);
    if (!open) return null;
    return h('section', { role: 'dialog', 'aria-label': t('title'), style: { position: 'fixed', inset: '16px', zIndex: 1000, display: 'flex', flexDirection: 'column', minHeight: 0, background: 'var(--dsw-alias-bg-base, white)', color: 'var(--dsw-alias-label-primary)', border: '1px solid var(--dsw-alias-border-l1, #ddd)', borderRadius: '10px', boxShadow: '0 12px 40px #0003', overflow: 'hidden' } },
      h('header', { style: { display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', flexShrink: 0 } }, h(Icon), h('strong', { style: { flex: 1, fontSize: '14px' } }, t('title')), h('button', { ref: closeButton, type: 'button', 'aria-label': t('close'), onClick: () => overlay.set(false), style: buttonStyle }, '×')),
      h('div', { style: { flex: 1, minHeight: 0 } }, h(WorkbenchFrame)));
  }
  const seat = (name, component, extra = {}) => ctx.effect(() => ctx.slots.inject(name, () => ctx.slots.register({ name, id: PLUGIN, locale: PLUGIN, order: 30, ...extra }, component)));
  seat('sidebar.footer.action', Footer);
  // Clearing listeners belongs to the overlay seat's lifecycle too.
  ctx.effect(() => {
    const remove = ctx.slots.inject('shell.overlay', () => ctx.slots.register({ name: 'shell.overlay', id: PLUGIN, locale: PLUGIN, order: 30 }, GlobalWorkbench));
    return () => { overlay.set(false); listeners.clear(); return remove?.(); };
  });
  ctx.effect(() => ctx.sidebarRightTabs.register({ id: PLUGIN, kind: PLUGIN, title: () => t('open'), keepMounted: true, guide: [{ id: 'workbench', order: 30, title: () => t('open'), description: () => t('description'), icon: Icon }] }));
  seat('sidebar.right.pane.tab', WorkbenchFrame, { key: PLUGIN });
  seat('sidebar.right.pane.tab.title', () => h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '5px' } }, h(Icon, { size: 14 }), t('open')), { key: PLUGIN });
}
