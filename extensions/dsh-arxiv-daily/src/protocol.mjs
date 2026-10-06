export const PLUGIN = 'dsh-arxiv-daily';
export const ENDPOINT = 'arxiv-daily/open';
export function isWorkbenchUrl(value) {
  const match = typeof value === 'string' && /^http:\/\/127\.0\.0\.1:([1-9]\d{0,4})\/[a-f0-9]{48}\/$/.exec(value);
  return Boolean(match && Number(match[1]) <= 65535);
}
