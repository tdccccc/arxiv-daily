const DEFAULT_API_KEY_SECRET = "llm.apiKey";
const SECRET_PREFIX = "arxivDaily";

function createSecretProvider(context) {
  return {
    async getSecret(key) {
      return (await context.secrets.get(secretStorageKey(key))) ?? null;
    },
    async setSecret(key, value) {
      await context.secrets.store(secretStorageKey(key), value);
    },
    async deleteSecret(key) {
      await context.secrets.delete(secretStorageKey(key));
    },
  };
}

async function promptAndStoreApiKey(vscodeApi, context) {
  const value = await vscodeApi.window.showInputBox({
    title: "arXiv Daily API Key",
    prompt: "Enter the LLM API key used by arXiv Daily.",
    password: true,
    ignoreFocusOut: true,
    placeHolder: "sk-...",
  });
  if (value === undefined) return false;

  const apiKey = value.trim();
  if (!apiKey) {
    void vscodeApi.window.showWarningMessage("arXiv Daily: API key was not saved.");
    return false;
  }

  await createSecretProvider(context).setSecret(DEFAULT_API_KEY_SECRET, apiKey);
  void vscodeApi.window.showInformationMessage("arXiv Daily: API key saved.");
  return true;
}

function secretStorageKey(key) {
  const normalized = String(key).trim();
  if (!normalized) throw new Error("secret key must not be empty");
  return `${SECRET_PREFIX}.${normalized}`;
}

module.exports = {
  DEFAULT_API_KEY_SECRET,
  createSecretProvider,
  promptAndStoreApiKey,
  secretStorageKey,
};
