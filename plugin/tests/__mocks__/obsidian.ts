/* Test-only stub for the `obsidian` package. Mirrors the runtime surface we use. */

export class Notice {
  static calls: Array<{ message: string; timeoutMs?: number }> = [];

  constructor(message: string, timeoutMs?: number) {
    Notice.calls.push({ message, timeoutMs });
  }
}

export function normalizePath(p: string): string {
  return p;
}

export function setIcon(_parent: HTMLElement, _iconId: string): void {}

export class TFile {
  constructor(readonly path: string = "") {}
}

export interface Vault {
  adapter: {
    read(path: string): Promise<string>;
    write(path: string, content: string): Promise<void>;
    exists(path: string): Promise<boolean>;
    mkdir(path: string): Promise<void>;
    rename(from: string, to: string): Promise<void>;
    remove(path: string): Promise<void>;
    readBinary?(path: string): Promise<ArrayBuffer>;
    writeBinary?(path: string, content: ArrayBuffer): Promise<void>;
  };
}

export interface WorkspaceLeaf {
  setViewState(state: any): Promise<void>;
}

export interface Workspace {
  getLeavesOfType(type: string): WorkspaceLeaf[];
  getLeaf(newLeaf?: boolean): WorkspaceLeaf | null;
  setActiveLeaf(leaf: WorkspaceLeaf, options?: { focus?: boolean }): void;
  revealLeaf(leaf: WorkspaceLeaf): Promise<void>;
  detachLeavesOfType(type: string): Promise<void>;
  openLinkText(path: string, sourcePath: string, newLeaf?: boolean): Promise<void>;
}

export interface App {
  vault?: Vault;
  workspace?: Workspace;
}

export class Plugin {}

export class ItemView {
  contentEl: HTMLElement = (globalThis as any).document?.createElement?.("div") ?? ({} as any);
  constructor(readonly leaf: WorkspaceLeaf) {}
  getViewType(): string { return ""; }
  getDisplayText(): string { return ""; }
  getIcon(): string { return ""; }
  onOpen(): Promise<void> | void {}
  onClose(): Promise<void> | void {}
}

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
  static opened: Modal[] = [];
  titleEl: HTMLElement = (globalThis as any).document?.createElement?.("div") ?? ({} as any);
  contentEl: HTMLElement = (globalThis as any).document?.createElement?.("div") ?? ({} as any);
  constructor(_app: App) {}
  open(): void {
    Modal.opened.push(this);
    this.onOpen();
  }
  close(): void {
    this.onClose();
  }
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
