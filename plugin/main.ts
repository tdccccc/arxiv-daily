import { Plugin } from "obsidian";

export default class ArxivDailyPlugin extends Plugin {
  async onload() {
    console.log("arxiv-daily plugin loaded");
  }
  onunload() {
    console.log("arxiv-daily plugin unloaded");
  }
}
