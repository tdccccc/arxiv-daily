import { describe, expect, it } from 'vitest';
import { getBusinessSetting, getBusinessSettingsSections, businessSettingsContext, SETTINGS_KEYS, TIMEZONE_OPTIONS, type BusinessSettingsContext } from '../src/settings/schema';
import { DEFAULT_SETTINGS } from '../src/settings/defaults';
const visibleIds = (context: Partial<BusinessSettingsContext> = {}) => getBusinessSettingsSections(context).flatMap(section => section.type === 'group' ? section.items.map(item => item.id) : [section.field.id]);
describe('shared business settings schema', () => {
 it('keeps the business order, standalone rows and settings keys shared across hosts', () => {
  expect(getBusinessSettingsSections().map(section=>section.type==='field'?section.field.id:section.heading)).toEqual(['scheduleEnabled','LLM','arXiv categories','Research topics','detailProfile','timezone','Output & schedule','Personal library','Email delivery','Advanced']);
  expect(getBusinessSetting('apiBaseUrl')).toMatchObject({key:SETTINGS_KEYS.llm.baseUrl,name:'API base URL',control:'text'});
  expect(getBusinessSetting('emailEnabled')).toMatchObject({key:SETTINGS_KEYS.email.enabled,control:'toggle'});
  expect(TIMEZONE_OPTIONS.map(zone=>zone.value)).toContain('Asia/Shanghai');
 });
 it('owns conditional visibility and descriptions for remote embedding, parser and delivery', () => {
  expect(visibleIds()).not.toContain('embeddingApiKey');expect(visibleIds()).not.toContain('sidecarParseUrl');expect(visibleIds()).not.toContain('hostedToken');expect(visibleIds()).toContain('emailApiKey');
  const context={embeddingMode:'remote',sidecarEnabled:true,emailMode:'hosted'} as const;
  expect(visibleIds(context)).toEqual(expect.arrayContaining(['embeddingApiKey','sidecarParseUrl','hostedToken']));expect(visibleIds(context)).not.toContain('emailApiKey');expect(visibleIds(context)).not.toContain('fromEmail');
  expect(getBusinessSetting('embeddingMode',context).description).toContain('Remote sends titles and abstracts');
  expect(getBusinessSetting('emailMode',context).description).toContain('shared free service');
 });
 it('shares exact options and preserves custom detail profile only when already selected', () => {
  expect(getBusinessSetting('reasoningEffort').options).toEqual({none:'None',low:'Low',medium:'Medium',high:'High'});
  expect(getBusinessSetting('detailProfile').options).toEqual({conservative:'Fewer',balanced:'Recommended',broad:'More'});
  expect(getBusinessSetting('detailProfile',{detailProfile:'custom'}).options?.custom).toBe('Custom (current values)');
  expect(getBusinessSetting('linkStyle').options).toEqual({wikilink:'Obsidian wikilink',relative:'Standard relative link'});
  const options=getBusinessSetting('linkStyle').options!;options.wikilink='changed';expect(getBusinessSetting('linkStyle').options?.wikilink).toBe('Obsidian wikilink');
 });
 it('projects host capability and settings without passing host objects into metadata', () => {
  const settings=structuredClone(DEFAULT_SETTINGS);settings.schedule.enabled=true;
  const context=businessSettingsContext(settings,false);
  expect(getBusinessSetting('scheduleEnabled',context).name).toBe('Enable · Running');
  expect(getBusinessSetting('emailEnabled',context).description).toContain('Automatic');
 });
});

it('can retain hidden controls for draft-preserving renderers while keeping core visibility authoritative',()=>{
 const section=getBusinessSettingsSections({}, {includeHidden:true}).find(s=>s.type==='group'&&s.id==='library');
 expect(section?.type==='group'&&section.items.find(f=>f.id==='embeddingApiKey')).toMatchObject({visible:false});
 expect(getBusinessSetting('detailProfile').defaultValue).toBe(DEFAULT_SETTINGS.detailSelection.profile);
 expect(getBusinessSetting('linkStyle').defaultValue).toBe(DEFAULT_SETTINGS.output.linkStyle);
});

it('retains existing custom reasoning options safely and shares topic editor fields and time choices',async()=>{
 const {getTopicSettingFields,runWindowTimeOptions}=await import('../src/settings/schema');
 expect(getBusinessSetting('reasoningEffort',{reasoningEffort:'vendor-effort'}).options?.['vendor-effort']).toBe('Custom (current values)');
 const options=getBusinessSetting('reasoningEffort',{reasoningEffort:'__proto__'}).options!;
 expect(Object.hasOwn(options,'__proto__')).toBe(true);expect(Object.getPrototypeOf(options)).toBe(Object.prototype);
 expect(getTopicSettingFields().map(field=>[field.key,field.control])).toEqual([['name','text'],['directions','textarea'],['detail','checkbox']]);
 expect(runWindowTimeOptions('08:07')).toHaveLength(97);
});
