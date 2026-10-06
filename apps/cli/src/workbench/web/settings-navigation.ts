import { t } from "./i18n";
/** Move existing nodes: navigation never recreates controls or submits their drafts. */
export function mountSettingsNavigation(form: HTMLFormElement): void {
  if (form.querySelector('.settings-layout')) return;
  const layout = document.createElement('div'); layout.className = 'settings-layout';
  const nav = document.createElement('nav'); nav.className = 'settings-navigation'; nav.setAttribute('aria-label', t('设置导航'));
  const content = document.createElement('div'); content.className = 'settings-content';
  for (const child of Array.from(form.children)) if (!child.classList.contains('dialog-footer')) content.append(child);
  layout.append(nav, content); form.prepend(layout);
  const entries: Array<{ target: HTMLElement; button: HTMLButtonElement }> = [];
  for (const target of Array.from(content.children) as HTMLElement[]) {
    const heading = target.querySelector<HTMLElement>('[data-settings-heading]');
    const name = target.classList.contains('settings-host-context') ? t('设置概览') : (heading ? t(heading.dataset.settingsKey || heading.textContent || '') : '') || (['Automatic detail notes', 'Timezone'].includes(target.dataset.settingName || '') ? t(target.dataset.settingName!) : undefined);
    if (!name) continue;
    target.id = `settings-section-${entries.length}`;
    target.classList.add('settings-nav-target');
    const focus = heading ?? target; focus.tabIndex = -1;
    const button = document.createElement('button'); button.type = 'button'; button.dataset.settingsNav = ''; button.textContent = name; button.setAttribute('aria-controls', target.id);
    button.addEventListener('click', () => {
      activate(button);
      target.scrollIntoView?.({ block: 'start', behavior: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
      focus.focus({ preventScroll: true });
    });
    nav.append(button); entries.push({ target, button });
  }
  function activate(current: HTMLButtonElement) {
    for (const { button } of entries) {
      if (button === current) button.setAttribute('aria-current', 'location'); else button.removeAttribute('aria-current');
    }
  }
  content.addEventListener('scroll', () => {
    const top = content.getBoundingClientRect().top + 32;
    let current = entries[0];
    for (const entry of entries) if (entry.target.getBoundingClientRect().top <= top) current = entry;
    if (content.scrollHeight > content.clientHeight && content.scrollTop + content.clientHeight >= content.scrollHeight - 2) current = entries.at(-1);
    if (current) activate(current.button);
  }, { passive: true });
  if (entries[0]) activate(entries[0].button);
}
