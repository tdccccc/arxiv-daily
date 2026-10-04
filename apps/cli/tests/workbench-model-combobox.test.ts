import { setUiLanguage } from "../src/workbench/web/i18n";
// @vitest-environment happy-dom
import { afterEach, expect, it } from 'vitest';
import { modelCombobox, mountModelCombobox } from '../src/workbench/web/model-combobox';
afterEach(()=>{document.body.innerHTML='';});
it('loads choices under the original input without adding another field or replacing its value',()=>{
 const form=document.createElement('form');form.innerHTML=modelCombobox('custom-model');document.body.append(form);
 const control=mountModelCombobox(form);const input=form.querySelector<HTMLInputElement>('input')!;
 control.setOptions(['model-a','model-b']);
 expect(form.querySelectorAll('input')).toHaveLength(1);expect(form.querySelector('select')).toBeNull();
 expect(input.value).toBe('custom-model');expect(input.getAttribute('aria-expanded')).toBe('true');
 const option=form.querySelectorAll<HTMLButtonElement>('[role=option]')[1]!;option.click();
 expect(input.value).toBe('model-b');expect(input.getAttribute('aria-expanded')).toBe('false');
 form.querySelector<HTMLButtonElement>('[data-model-toggle]')!.click();expect(input.getAttribute('aria-expanded')).toBe('true');
 input.value='model-a';input.dispatchEvent(new Event('input',{bubbles:true}));
 expect(form.querySelectorAll('[role=option]')).toHaveLength(1);
 input.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true,cancelable:true}));
 const enter=new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true});input.dispatchEvent(enter);
 expect(enter.defaultPrevented).toBe(true);expect(input.value).toBe('model-a');
});
it('allows free text and closes suggestions on Escape or leaving the field',()=>{
 const form=document.createElement('form');form.innerHTML=modelCombobox('typed')+'<button type="submit">Get models</button>';document.body.append(form);
 const control=mountModelCombobox(form);control.setOptions(['alpha']);const input=form.querySelector<HTMLInputElement>('input')!;
 input.value='unknown';input.dispatchEvent(new Event('input',{bubbles:true}));expect(input.value).toBe('unknown');
 input.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true}));expect(input.getAttribute('aria-expanded')).toBe('false');
 form.querySelector<HTMLButtonElement>('[data-model-toggle]')!.click();form.querySelector<HTMLButtonElement>('[type=submit]')!.focus();
 expect(input.getAttribute('aria-expanded')).toBe('false');
 control.setOptions([]);expect(form.querySelector<HTMLButtonElement>('[data-model-toggle]')!.disabled).toBe(true);expect(input.value).toBe('unknown');
});

it('localizes model picker controls without translating identifiers',()=>{
 setUiLanguage('zh');const form=document.createElement('form');form.innerHTML=modelCombobox('English');document.body.append(form);
 const control=mountModelCombobox(form);const input=form.querySelector<HTMLInputElement>('input')!;
 expect(input.value).toBe('English');expect(input.placeholder).toBe('模型名称');expect(form.querySelector('[data-model-toggle]')?.getAttribute('aria-label')).toBe('展开模型选项');
 control.setOptions(['Chinese']);expect(form.querySelector('[role=option]')?.textContent).toBe('Chinese');
 setUiLanguage('en');form.innerHTML=modelCombobox('中文模型');expect(form.querySelector('input')?.getAttribute('placeholder')).toBe('Model name');expect(form.querySelector('input')?.getAttribute('value')).toBe('中文模型');
});
