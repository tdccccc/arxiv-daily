// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from 'vitest';
import { mountSettingsNavigation } from '../src/workbench/web/settings-navigation';
afterEach(() => { document.body.innerHTML=''; vi.restoreAllMocks(); });
it('adds jump navigation without hiding settings, submitting changes or losing drafts', () => {
 const form=document.createElement('form');
 form.innerHTML='<div class="settings-host-context">Root</div><section class="settings-section"><h3 data-settings-heading>LLM</h3><input name="model" value="draft"></section><div class="setting-item" data-setting-name="Timezone">Timezone<select><option>UTC</option></select></div><section class="settings-section"><h3 data-settings-heading>Email delivery</h3><input name="email"></section><div class="dialog-footer"><button type="submit">Save</button></div>';
 document.body.append(form); const submit=vi.fn();form.addEventListener('submit',submit);
 const scroll=vi.fn();form.querySelectorAll<HTMLElement>('section')[1]!.scrollIntoView=scroll;
 mountSettingsNavigation(form);
 const nav=form.querySelector('nav[aria-label="设置导航"]');expect(nav).toBeTruthy();
 expect(Array.from(nav!.querySelectorAll('button')).map(b=>b.textContent)).toEqual(['设置概览','LLM','Timezone','Email delivery']);
 const email=Array.from(nav!.querySelectorAll('button')).find(b=>b.textContent==='Email delivery')!;email.click();
 expect(scroll).toHaveBeenCalled(); expect(email.getAttribute('aria-current')).toBe('location');
 expect(submit).not.toHaveBeenCalled();expect(form.querySelector<HTMLInputElement>('[name=model]')!.value).toBe('draft');
 expect(form.querySelectorAll('.settings-content .settings-section')).toHaveLength(2);
 expect(form.querySelector('.settings-content [hidden]')).toBeNull();
 expect(form.querySelector(':scope > .dialog-footer')).toBeTruthy();
});
it('tracks the visible section during scrolling and does not duplicate navigation',()=>{
 const form=document.createElement('form');form.innerHTML='<section class="settings-section"><h3 data-settings-heading>LLM</h3></section><section class="settings-section"><h3 data-settings-heading>Advanced</h3></section>';document.body.append(form);
 mountSettingsNavigation(form);mountSettingsNavigation(form);
 expect(form.querySelectorAll('nav')).toHaveLength(1);
 const sections=form.querySelectorAll<HTMLElement>('section');
 sections[0]!.getBoundingClientRect=()=>({top:-200} as DOMRect);sections[1]!.getBoundingClientRect=()=>({top:10} as DOMRect);
 form.querySelector('.settings-content')!.dispatchEvent(new Event('scroll'));
 expect(form.querySelector('[aria-current=location]')?.textContent).toBe('Advanced');
});
