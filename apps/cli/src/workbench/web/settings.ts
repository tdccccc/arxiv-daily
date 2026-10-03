import type { WorkbenchSettings as SettingsSnapshot } from "../settings";
import type { Topic } from "@arxiv-daily/core";
export type { WorkbenchSettings as SettingsSnapshot } from "../settings";
const escape = (text: string) => text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
function field(name: string, label: string, value: string, type = 'text'): string {
  return `<label class="settings-field">${label}<input name="${name}" type="${type}" value="${escape(value)}" ${type === 'password' ? 'autocomplete="new-password"' : ''}></label>`;
}
function topicRow(topic: Topic): string {
  return `<fieldset class="settings-topic" data-topic-id="${escape(topic.id)}"><legend>关注主题</legend><div class="settings-grid">${field('topicName', '名称', topic.name)}${field('topicTag', '标签', topic.tag)}</div><label class="settings-field">筛选描述<textarea name="topicDescription" rows="3">${escape(topic.description)}</textarea></label><div class="settings-topic-actions"><label><input type="checkbox" name="topicDetail" ${topic.detail ? 'checked' : ''}> 自动生成详细总结</label><button type="button" class="quiet-button" data-settings="remove-topic">删除主题</button></div></fieldset>`;
}
export function settingsForm(snapshot: SettingsSnapshot): string {
  const v = snapshot.values;
  return `<form class="settings-form"><p class="dialog-description">${snapshot.setupRequired ? '选择研究记录保存位置，配置模型与关注主题，即可生成第一份日报。' : '保存后立即生效，已有研究记录仍保存在原目录。'}</p><div class="settings-layout"><nav class="settings-navigation" aria-label="设置分组">${[['records', '研究记录'], ['model', '模型 API'], ['discovery', '每日发现']].map(([key, label]) => `<button type="button" data-settings-group="${key}" aria-controls="settings-${key}" aria-current="${key === 'records' ? 'true' : 'false'}">${label}</button>`).join('')}</nav><div class="settings-panels"><section id="settings-records" data-settings-panel><h3>研究记录</h3>${field('vaultRoot', '保存根目录（本机绝对路径）', v.vaultRoot)}<div class="settings-grid">${field('dailyDir', '日报子目录', v.dailyDir)}${field('papersDir', '论文总结子目录', v.papersDir)}</div></section><section id="settings-model" data-settings-panel hidden><h3>模型 API</h3><p class="settings-hint">论文筛选和总结使用独立的模型 API，与 DSH 或 Claude Code 的对话模型独立配置。</p><div class="settings-grid">${field('provider', '服务商', v.provider)}${field('model', '模型名称', v.model)}</div>${field('baseUrl', 'API 地址', v.baseUrl)}${field('apiKey', 'API 密钥', '', 'password')}<p class="settings-hint">${v.apiKeyConfigured ? '已配置密钥；留空保留现有密钥。' : '尚未配置密钥。密钥保存后不会回显。'}</p></section><section id="settings-discovery" data-settings-panel hidden><h3>每日发现</h3>${field('categories', 'arXiv 分类（逗号分隔）', v.categories.join(', '))}<div class="settings-grid">${field('timezone', '时区', v.timezone)}<label class="settings-field">总结语言<select name="summaryLanguage"><option value="zh" ${v.summaryLanguage === 'zh' ? 'selected' : ''}>中文</option><option value="en" ${v.summaryLanguage === 'en' ? 'selected' : ''}>English</option></select></label></div><div class="settings-topic-list">${v.topics.map(topicRow).join('')}</div><button type="button" class="quiet-button" data-settings="add-topic">＋ 添加主题</button></section><p class="settings-hint">配置文件：<code class="config-path">${escape(snapshot.configPath)}</code></p></div></div><p class="form-error" role="alert" hidden></p><div class="dialog-footer"><button type="button" class="quiet-button" data-action="close-dialog">取消</button><button type="submit" class="primary-button">${snapshot.setupRequired ? '保存并开始使用' : '保存设置'}</button></div></form>`;
}
export function bindSettings(form: HTMLFormElement, snapshot: SettingsSnapshot, request: <T>(url: string, body?: unknown) => Promise<T>, saved: () => Promise<void>): void {
  let busy = false;
  form.addEventListener('click', event => {
    const group = event.target instanceof HTMLElement ? event.target.closest<HTMLElement>('[data-settings-group]') : null;
    if (group) {
      for (const panel of Array.from(form.querySelectorAll<HTMLElement>('[data-settings-panel]'))) panel.hidden = panel.id !== `settings-${group.dataset.settingsGroup}`;
      for (const button of Array.from(form.querySelectorAll<HTMLElement>('[data-settings-group]'))) button.setAttribute('aria-current', String(button === group));
      return;
    }
    const target = event.target instanceof HTMLElement ? event.target.closest<HTMLElement>('[data-settings]') : null;
    if (busy || !target) return;
    if (target.dataset.settings === 'add-topic') {
      form.querySelector('.settings-topic-list')!.insertAdjacentHTML('beforeend', topicRow({ id: crypto.randomUUID(), name: '', tag: '', description: '', detail: true }));
      form.querySelector<HTMLInputElement>('.settings-topic:last-child input')?.focus();
    } else target.closest('.settings-topic')?.remove();
  });
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (busy) return;
    const get = (name: string) => (form.elements.namedItem(name) as HTMLInputElement).value.trim();
    const topics = Array.from(form.querySelectorAll<HTMLElement>('.settings-topic')).map(row => ({
      id: row.dataset.topicId!,
      name: row.querySelector<HTMLInputElement>('[name="topicName"]')!.value.trim(),
      tag: row.querySelector<HTMLInputElement>('[name="topicTag"]')!.value.trim(),
      description: row.querySelector<HTMLTextAreaElement>('[name="topicDescription"]')!.value.trim(),
      detail: row.querySelector<HTMLInputElement>('[name="topicDetail"]')!.checked,
    }));
    const apiKey = get('apiKey');
    const values = { vaultRoot: get('vaultRoot'), provider: get('provider'), model: get('model'), baseUrl: get('baseUrl'), ...(apiKey ? { apiKey } : {}), categories: get('categories').split(/[,，\s]+/).filter(Boolean), timezone: get('timezone'), summaryLanguage: get('summaryLanguage'), dailyDir: get('dailyDir'), papersDir: get('papersDir'), topics };
    const button = form.querySelector<HTMLButtonElement>('[type="submit"]')!;
    const error = form.querySelector<HTMLElement>('[role="alert"]')!;
    const label = button.textContent;
    busy = true; button.disabled = true; button.textContent = '正在保存…'; error.hidden = true;
    void request<SettingsSnapshot>('api/settings', { revision: snapshot.revision, values }).then(async () => {
      if (form.isConnected) await saved();
    }).catch(reason => {
      if (!form.isConnected) return;
      error.hidden = false; error.textContent = reason instanceof Error ? reason.message : '保存失败，请重试。';
    }).finally(() => { busy = false; button.disabled = false; button.textContent = label; });
  });
}
