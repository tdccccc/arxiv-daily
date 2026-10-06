import { t } from "./i18n";
const escape = (value: string) => value.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]!);
export function modelCombobox(value: string): string {
  return `<div class="model-combobox"><input name="model" aria-label= "${escape(t('Model'))}" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="settings-model-choices" autocomplete="off" value="${escape(value)}" placeholder= "${escape(t('Model name'))}"><button type="button" data-model-toggle aria-label= "${escape(t('展开模型选项'))}" disabled>▾</button><div id="settings-model-choices" class="model-choices" role="listbox" aria-label= "${escape(t('Available models'))}" hidden></div></div>`;
}
export function mountModelCombobox(root: HTMLElement) {
  const box = root.querySelector<HTMLElement>('.model-combobox')!;
  const input = box.querySelector<HTMLInputElement>('input')!;
  const toggle = box.querySelector<HTMLButtonElement>('[data-model-toggle]')!;
  const list = box.querySelector<HTMLElement>('[role=listbox]')!;
  let models: string[] = [], visible: string[] = [], active = -1;
  function close() { list.hidden = true; input.setAttribute('aria-expanded','false'); input.removeAttribute('aria-activedescendant'); active = -1; }
  function highlight() {
    for (const [index, option] of Array.from(list.querySelectorAll<HTMLElement>('[role=option]')).entries()) option.setAttribute('aria-selected',String(index===active));
    if (active >= 0) input.setAttribute('aria-activedescendant',`settings-model-option-${active}`); else input.removeAttribute('aria-activedescendant');
    list.querySelector<HTMLElement>('[aria-selected=true]')?.scrollIntoView?.({ block:'nearest' });
  }
  function open(filter = '') {
    if (!models.length || input.disabled) return;
    visible = models.filter(model => model.toLocaleLowerCase().includes(filter.toLocaleLowerCase())); active = -1;
    list.innerHTML = visible.length ? visible.map((model,i) => `<button type="button" role="option" id="settings-model-option-${i}" data-model-index="${i}" aria-selected="false" tabindex="-1">${escape(model)}</button>`).join('') : `<div class="model-empty">${escape(t('No matching models. You can type a model name.'))}</div>`;
    list.hidden = false; input.setAttribute('aria-expanded','true'); input.removeAttribute('aria-activedescendant');
  }
  function choose(index: number) {
    if (visible[index] === undefined) return;
    input.value = visible[index]; close(); input.focus({preventScroll:true}); input.dispatchEvent(new Event('change',{bubbles:true}));
  }
  toggle.addEventListener('click',()=>{ if (list.hidden) { input.focus({preventScroll:true}); open(); } else { close(); input.focus({preventScroll:true}); } });
  input.addEventListener('click',()=>open());
  input.addEventListener('input',()=>open(input.value));
  input.addEventListener('keydown',event=>{
    if (event.key==='ArrowDown'||event.key==='ArrowUp') {
      if (!models.length) return;
      event.preventDefault(); if (list.hidden) open();
      if (visible.length) { active = event.key==='ArrowDown' ? (active+1)%visible.length : (active<=0?visible.length-1:active-1); highlight(); }
    } else if (event.key==='Enter'&&!list.hidden&&active>=0) { event.preventDefault(); choose(active); }
    else if (event.key==='Escape'&&!list.hidden) { event.preventDefault(); event.stopPropagation(); close(); }
    else if (event.key==='Tab') close();
  });
  list.addEventListener('pointerdown',event=>event.preventDefault());
  list.addEventListener('click',event=>{const option=event.target instanceof Element?event.target.closest<HTMLElement>('[data-model-index]'):null;if(option)choose(Number(option.dataset.modelIndex));});
  box.addEventListener('focusout',event=>{if(!box.contains(event.relatedTarget as Node|null))close();});
  return { setOptions(next: string[]) { models=[...new Set(next)]; toggle.disabled=models.length===0; if(models.length){input.focus({preventScroll:true});open();}else close(); } };
}
