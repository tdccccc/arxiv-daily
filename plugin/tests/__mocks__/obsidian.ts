/* Test-only stub for the `obsidian` package. Mirrors the runtime surface we use. */
export class Notice {
  constructor(_message: string, _timeoutMs?: number) {}
}

export function normalizePath(p: string): string {
  return p;
}

export interface Vault {
  adapter: {
    write(path: string, content: string): Promise<void>;
    exists(path: string): Promise<boolean>;
    mkdir(path: string): Promise<void>;
    rename(from: string, to: string): Promise<void>;
    remove(path: string): Promise<void>;
  };
}

export interface App {}

export class Plugin {}

export class PluginSettingTab {
  constructor(_app: App, _plugin: Plugin) {}
}

export class Setting {
  constructor(_container: HTMLElement) {}
  setName(_v: string) { return this; }
  setDesc(_v: string) { return this; }
  addText(_cb: any) { return this; }
  addTextArea(_cb: any) { return this; }
  addToggle(_cb: any) { return this; }
  addDropdown(_cb: any) { return this; }
  addButton(_cb: any) { return this; }
}

export class Modal {
  contentEl: HTMLElement = (globalThis as any).document?.createElement?.("div") ?? ({} as any);
  constructor(_app: App) {}
  open(): void {}
  close(): void {}
  onOpen(): void {}
  onClose(): void {}
}

export async function requestUrl(_opts: any): Promise<{ status: number; text: string }> {
  return { status: 200, text: "" };
}
