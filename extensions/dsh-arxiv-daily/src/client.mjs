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
  // The `arxiv daily` wordmark as outlined glyph paths (Inter Display, Black 900 / Light 300),
  // same geometry as the browser workbench header's mark, flattened to one `currentColor` fill
  // so it follows the dialog's own text color instead of the workbench's light/dark ink+accent
  // pair. No font dependency: it renders identically whether or not the font is installed.
  function Wordmark({ height = 18 } = {}) {
    return h('svg', { height, viewBox: '-21.56 -97.73 457.10 139.28', role: 'img', 'aria-label': t('title'), style: { flexShrink: 0 } },
      h('path', { fill: 'currentColor', d: 'M20.020 0.928C26.611 0.928 31.445-1.074 34.863-7.227L35.059-7.227L35.059 0L54.102 0L54.102-34.473C54.102-46.191 45.605-52.734 29.688-52.734C13.867-52.734 5.420-46.191 4.590-34.473L22.461-34.473C22.803-37.500 25.049-39.160 28.906-39.160C32.471-39.160 34.766-37.695 34.766-35.449L34.766-35.254C34.766-32.422 31.348-31.348 22.266-30.762C11.133-30.029 2.441-25.781 2.441-14.258C2.441-4.004 9.668 0.928 20.020 0.928M26.367-11.230C23.047-11.230 20.703-12.793 20.703-15.430C20.703-18.066 23.145-19.824 27.148-20.508C30.762-21.143 33.008-21.680 34.961-22.363L34.961-18.262C34.961-13.672 30.566-11.230 26.367-11.230M60.100 0L79.631 0L79.631-24.707C79.631-31.592 83.489-35.937 89.592-35.937L96.721-35.937L96.721-51.562L91.838-51.562C84.758-51.562 80.998-49.121 79.143-42.187L79.045-42.187L79.045-51.562L60.100-51.562M94.907 0L115.805 0L118.491-4.687C120.346-7.959 121.811-10.791 123.325-13.623C124.790-10.791 126.304-7.959 128.061-4.687L130.600 0L151.548 0L134.018-26.855L150.766-51.562L130.259-51.562L127.720-46.875C126.157-44.043 124.839-41.455 123.569-38.965C122.348-41.455 121.030-44.043 119.565-46.875L117.124-51.562L96.323-51.562L112.729-26.270M153.152 0L172.683 0L172.683-51.562L153.152-51.562M162.868-54.980C168.972-54.980 173.269-58.887 173.269-64.404C173.269-69.873 168.972-73.730 162.868-73.730C156.765-73.730 152.468-69.873 152.468-64.404C152.468-58.887 156.765-54.980 162.868-54.980M192.744 0L215.205 0L233.662-51.562L213.545-51.562L207.392-29.687C206.269-25.684 205.244-21.680 204.218-17.627C203.144-21.680 202.021-25.684 200.898-29.687L194.599-51.562L174.287-51.562M271.722 0.798C278.357 0.798 284.026-2.646 286.755-8.398L286.839-8.398L286.839 0L292.130 0L292.130-62.568L286.713-62.568L286.713-35.987L286.671-35.987C284.194-41.656 278.357-45.142 271.722-45.142C260.342-45.142 252.406-35.903 252.406-22.172C252.406-8.440 260.300 0.798 271.722 0.798M272.436-4.157C263.996-4.157 257.865-10.624 257.865-22.172C257.865-33.720 263.996-40.229 272.436-40.229C281.338-40.229 287.133-33.132 287.133-22.172C287.133-11.212 281.338-4.157 272.436-4.157M315.034 0.672C323.013 0.672 327.422-3.149 329.353-7.265L329.479-7.265L329.479 0L334.854 0L334.854-30.738C334.854-39.263 328.094-45.142 318.225-45.142C308.189-45.142 301.009-39.095 300.757-31.074L306.216-31.074C306.468-36.281 311.423-40.271 318.225-40.271C324.986-40.271 329.479-36.239 329.479-30.738L329.479-30.066C329.479-26.581 325.910-26.455 317.638-25.321C307.223-23.978 299.749-21.584 299.749-12.388C299.749-3.695 306.594 0.672 315.034 0.672M315.706-4.241C309.575-4.241 305.250-7.055 305.250-12.136C305.250-17.469 309.953-19.568 317.386-20.618C323.222-21.416 327.926-22.172 329.479-23.558L329.479-17.343C329.479-9.658 324.566-4.241 315.706-4.241M344.614 0L349.989 0L349.989-44.344L344.614-44.344M347.302-53.960C349.443-53.960 351.207-55.724 351.207-57.865C351.207-60.007 349.443-61.771 347.302-61.771C345.160-61.771 343.397-60.007 343.397-57.865C343.397-55.724 345.160-53.960 347.302-53.960M365.166-62.568L359.791-62.568L359.791 0L365.166 0M374.926 17.553L379.042 17.553C384.585 17.553 388.112 14.739 390.631 8.440L411.543-44.344L405.791-44.344L395.376-17.259C393.991-13.689 392.605-10.078 391.303-6.509C390.001-10.078 388.658-13.689 387.272-17.259L376.732-44.344L370.895-44.344L388.532-0.126L385.802 6.677C383.997 11.212 382.359 12.724 379.042 12.724L374.926 12.724' }));
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
      h('header', { style: { display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', flexShrink: 0 } }, h('div', { style: { flex: 1, display: 'flex', alignItems: 'center' } }, h(Wordmark, { height: 18 })), h('button', { ref: closeButton, type: 'button', 'aria-label': t('close'), onClick: () => overlay.set(false), style: buttonStyle }, '×')),
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
