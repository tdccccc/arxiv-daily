import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const {
  DEFAULT_API_KEY_SECRET,
  createSecretProvider,
  promptAndStoreApiKey,
  secretStorageKey,
} = require("../src/secrets.js");

assert.equal(secretStorageKey("llm.apiKey"), "arxivDaily.llm.apiKey");
assert.throws(() => secretStorageKey(" "), /must not be empty/);

const context = createMockContext();
const provider = createSecretProvider(context);
assert.equal(await provider.getSecret(DEFAULT_API_KEY_SECRET), null);

await provider.setSecret(DEFAULT_API_KEY_SECRET, "sk-test");
assert.equal(await provider.getSecret(DEFAULT_API_KEY_SECRET), "sk-test");

await provider.deleteSecret(DEFAULT_API_KEY_SECRET);
assert.equal(await provider.getSecret(DEFAULT_API_KEY_SECRET), null);

const saved = await promptAndStoreApiKey(createMockVscodeApi("  sk-live  "), context);
assert.equal(saved, true);
assert.equal(await provider.getSecret(DEFAULT_API_KEY_SECRET), "sk-live");

const cancelled = await promptAndStoreApiKey(createMockVscodeApi(undefined), context);
assert.equal(cancelled, false);
assert.equal(await provider.getSecret(DEFAULT_API_KEY_SECRET), "sk-live");

const empty = await promptAndStoreApiKey(createMockVscodeApi("   "), context);
assert.equal(empty, false);
assert.equal(await provider.getSecret(DEFAULT_API_KEY_SECRET), "sk-live");

console.log("arXiv Daily VS Code SecretStorage adapter OK");

function createMockContext() {
  const values = new Map();
  return {
    secrets: {
      async get(key) {
        return values.get(key);
      },
      async store(key, value) {
        values.set(key, value);
      },
      async delete(key) {
        values.delete(key);
      },
    },
  };
}

function createMockVscodeApi(inputValue) {
  return {
    window: {
      async showInputBox() {
        return inputValue;
      },
      showInformationMessage() {},
      showWarningMessage() {},
    },
  };
}
