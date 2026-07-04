/**
 * Fill {{name}} placeholders in a prompt template. Single braces (e.g. JSON
 * examples) are left alone. Throws if any {{...}} placeholder remains unfilled,
 * which catches template typos and missing variables at call time.
 */
export function renderPrompt(
  template: string,
  vars: Record<string, string>,
): string {
  const rendered = template.replace(/\{\{(\w+)\}\}/g, (match, key: string) =>
    key in vars ? vars[key] ?? match : match,
  );
  const leftover = /\{\{\w+\}\}/.exec(rendered);
  if (leftover) {
    throw new Error(`renderPrompt: unfilled placeholder ${leftover[0]}`);
  }
  return rendered;
}
