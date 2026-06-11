/* Test-only stub for the `obsidian` package. Mirrors the runtime surface we use. */
export class Notice {
  constructor(_message: string, _timeoutMs?: number) {}
}

export function normalizePath(p: string): string {
  return p;
}

export interface Vault {
  adapter: {
    read(path: string): Promise<string>;
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

export class MenuItem {
  setTitle(_t: string) { return this; }
  setIcon(_icon: string | null) { return this; }
  setDisabled(_d: boolean) { return this; }
  onClick(_cb: (evt: MouseEvent | KeyboardEvent) => any) { return this; }
}

export class Menu {
  addItem(cb: (item: MenuItem) => any): this {
    cb(new MenuItem());
    return this;
  }
  addSeparator(): this { return this; }
  showAtMouseEvent(_evt: MouseEvent): this { return this; }
  showAtPosition(_p: { x: number; y: number }): this { return this; }
}

export async function requestUrl(_opts: any): Promise<{ status: number; text: string }> {
  return { status: 200, text: "" };
}
