const { buildDashboardState } = require("./dashboard-model");
const { PAPER_INDEX_PATH, findArxivDailyVault, normalizeStoragePath } = require("./workspace");

const DASHBOARD_VIEW_TYPE = "arxivDaily.dashboard";
const VALID_STATUSES = new Set(["inbox", "to_read", "reading", "read", "saved", "ignored"]);

async function openDashboard(vscodeApi, context) {
  const vault = await findArxivDailyVault(vscodeApi);
  if (!vault) {
    void vscodeApi.window.showWarningMessage(
      "arXiv Daily: no workspace folder contains arxiv-daily/.",
    );
    return null;
  }

  const panel = vscodeApi.window.createWebviewPanel(
    DASHBOARD_VIEW_TYPE,
    "arXiv Daily",
    vscodeApi.ViewColumn.One,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    },
  );

  const render = async () => {
    const index = await loadPaperIndex(vault.storage);
    panel.webview.html = renderDashboardHtml({
      nonce: createNonce(),
      state: buildDashboardState(index),
      workspaceName: vault.workspaceFolder.name,
    });
  };

  panel.webview.html = renderLoadingHtml("Loading arXiv Daily...");
  panel.webview.onDidReceiveMessage(async (message) => {
    try {
      if (message?.type === "refresh") {
        await render();
        return;
      }
      if (message?.type === "setStatus") {
        await updatePaperStatus(vault.storage, message.arxivId, message.status);
        await render();
        return;
      }
      if (message?.type === "openResource") {
        const index = await loadPaperIndex(vault.storage);
        await openResource(vscodeApi, vault, index, message.arxivId, message.resource);
      }
    } catch (error) {
      void vscodeApi.window.showErrorMessage(`arXiv Daily: ${error.message}`);
    }
  }, undefined, context.subscriptions);

  await render();
  return panel;
}

async function loadPaperIndex(storage) {
  if (!(await storage.exists(PAPER_INDEX_PATH))) {
    return { schemaVersion: 2, updatedAt: "", papers: {} };
  }
  return JSON.parse(await storage.readText(PAPER_INDEX_PATH));
}

async function updatePaperStatus(storage, arxivId, status, now = () => new Date()) {
  if (!VALID_STATUSES.has(status)) throw new Error(`invalid status: ${status}`);
  const index = await loadPaperIndex(storage);
  const entry = index.papers?.[arxivId];
  if (!entry) throw new Error(`${arxivId} is not in papers.json`);
  const timestamp = now().toISOString();
  entry.status = status;
  index.updatedAt = timestamp;
  await storage.writeText(PAPER_INDEX_PATH, JSON.stringify(index, null, 2));
  return entry;
}

async function openResource(vscodeApi, vault, index, arxivId, resource) {
  const entry = index.papers?.[arxivId];
  if (!entry) throw new Error(`${arxivId} is not in papers.json`);
  const target = resourceTargetForEntry(entry, resource);
  if (!target) {
    void vscodeApi.window.showWarningMessage(`arXiv Daily: no ${resource} for ${arxivId}.`);
    return;
  }

  if (target.kind === "url") {
    await vscodeApi.env.openExternal(vscodeApi.Uri.parse(target.value));
    return;
  }

  const uri = vscodeApi.Uri.joinPath(
    vault.vaultRootUri,
    ...normalizeStoragePath(target.value).split("/").filter(Boolean),
  );
  await vscodeApi.commands.executeCommand("vscode.open", uri);
}

function resourceTargetForEntry(entry, resource) {
  if (resource === "note" && entry.paperPath) {
    return { kind: "file", value: entry.paperPath };
  }
  if (resource === "daily" && entry.dailyReports?.[0]) {
    return { kind: "file", value: entry.dailyReports[0] };
  }
  if (resource === "arxiv" && entry.arxivUrl) {
    return { kind: "url", value: entry.arxivUrl };
  }
  if (resource === "pdf") {
    if (entry.pdfPath) return { kind: "file", value: entry.pdfPath };
    if (entry.pdfUrl) return { kind: "url", value: entry.pdfUrl };
  }
  return null;
}

function renderDashboardHtml({ nonce, state, workspaceName }) {
  const stateJson = safeJson({ ...state, workspaceName });
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>arXiv Daily</title>
  <style>
    :root { color-scheme: light dark; }
    body { margin: 0; padding: 16px; font-family: var(--vscode-font-family); color: var(--vscode-foreground); background: var(--vscode-editor-background); }
    header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
    h1 { margin: 0; font-size: 18px; font-weight: 600; }
    button, select, input { font: inherit; }
    button { border: 1px solid var(--vscode-button-border, transparent); background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); padding: 4px 8px; border-radius: 4px; cursor: pointer; }
    button:hover { background: var(--vscode-button-secondaryHoverBackground); }
    .tabs, .filters, .stats { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .tab[aria-pressed="true"] { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
    .filter { display: flex; align-items: center; gap: 6px; }
    .filter input { min-width: 240px; }
    .stat { border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 6px 8px; min-width: 88px; }
    .stat strong { display: block; font-size: 16px; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    th, td { border-bottom: 1px solid var(--vscode-panel-border); padding: 7px 6px; vertical-align: top; text-align: left; }
    th { color: var(--vscode-descriptionForeground); font-weight: 600; }
    .title { font-weight: 600; margin-bottom: 3px; overflow-wrap: anywhere; }
    .meta, .summary { color: var(--vscode-descriptionForeground); font-size: 12px; overflow-wrap: anywhere; }
    .actions { display: flex; flex-wrap: wrap; gap: 4px; }
    .empty { padding: 24px 0; color: var(--vscode-descriptionForeground); }
    @media (max-width: 760px) {
      body { padding: 10px; }
      .filter input { min-width: 140px; }
      th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>arXiv Daily</h1>
    <button id="refresh" type="button">Refresh</button>
  </header>
  <section class="tabs" id="tabs"></section>
  <section class="filters">
    <label class="filter">Search <input id="search" type="search" placeholder="ID, title, author, topic, summary"></label>
    <label class="filter">Status <select id="status"></select></label>
    <label class="filter">Priority <select id="priority"></select></label>
  </section>
  <section class="stats" id="stats"></section>
  <main id="results"></main>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const state = ${stateJson};
    const query = { tab: "watch", search: "", status: "", priority: "" };
    const labels = {
      inbox: "Inbox",
      to_read: "To read",
      reading: "Reading",
      read: "Read",
      saved: "Saved",
      ignored: "Ignored",
      high: "High",
      normal: "Normal",
      low: "Low",
    };

    const tabsEl = document.getElementById("tabs");
    const statsEl = document.getElementById("stats");
    const resultsEl = document.getElementById("results");
    const searchEl = document.getElementById("search");
    const statusEl = document.getElementById("status");
    const priorityEl = document.getElementById("priority");

    document.getElementById("refresh").addEventListener("click", () => {
      vscode.postMessage({ type: "refresh" });
    });
    searchEl.addEventListener("input", () => {
      query.search = searchEl.value;
      render();
    });
    statusEl.addEventListener("change", () => {
      query.status = statusEl.value;
      render();
    });
    priorityEl.addEventListener("change", () => {
      query.priority = priorityEl.value;
      render();
    });

    populateSelect(statusEl, state.statuses, "Any status");
    populateSelect(priorityEl, state.priorities, "Any priority");
    render();

    function populateSelect(select, values, anyLabel) {
      select.replaceChildren(option("", anyLabel), ...values.map((value) => option(value, labels[value] || value)));
    }

    function render() {
      tabsEl.replaceChildren(...state.tabs.map((tab) => {
        const button = document.createElement("button");
        button.className = "tab";
        button.type = "button";
        button.setAttribute("aria-pressed", String(query.tab === tab.id));
        button.textContent = \`\${tab.label} \${state.tabCounts[tab.id] || 0}\`;
        button.addEventListener("click", () => {
          query.tab = tab.id;
          render();
        });
        return button;
      }));

      const rows = (state.allRows || state.rows).filter((row) => matches(row));
      statsEl.replaceChildren(
        stat("Shown", rows.length),
        stat("Saved", rows.filter((row) => row.entry.status === "saved").length),
        stat("Missing citation", rows.filter((row) => row.entry.status === "saved" && !row.entry.citationKey.trim()).length),
        stat("Missing Zotero", rows.filter((row) => row.entry.status === "saved" && !row.entry.zoteroKey.trim() && !row.entry.zoteroUri.trim()).length),
      );

      if (rows.length === 0) {
        resultsEl.innerHTML = '<div class="empty">No papers in this view.</div>';
        return;
      }

      const table = document.createElement("table");
      table.innerHTML = '<thead><tr><th>Title</th><th>Status</th><th>Priority</th><th>Topic</th><th>First seen</th><th>Actions</th></tr></thead>';
      const tbody = document.createElement("tbody");
      for (const row of rows) tbody.appendChild(rowElement(row));
      table.appendChild(tbody);
      resultsEl.replaceChildren(table);
    }

    function matches(row) {
      if (!matchesTab(row.entry, query.tab)) return false;
      if (query.status && row.entry.status !== query.status) return false;
      if (query.priority && row.entry.priority !== query.priority) return false;
      const tokens = query.search.trim().toLowerCase().split(/\\s+/).filter(Boolean);
      if (tokens.length === 0) return true;
      const haystack = [
        row.arxivId,
        row.title,
        row.authors,
        row.topic,
        row.entry.summary?.coreProblem,
        row.entry.summary?.keyMethod,
        row.entry.summary?.mainResult,
        row.entry.summary?.whyRelevant,
      ].filter(Boolean).join(" ").toLowerCase();
      return tokens.every((token) => haystack.includes(token));
    }

    function matchesTab(entry, tab) {
      if (tab === "watch") return entry.status === "to_read" && entry.priority !== "high";
      if (tab === "highlight") return entry.status !== "ignored" && entry.priority === "high";
      if (tab === "reading") return entry.status === "reading";
      if (tab === "saved") return entry.status === "saved";
      if (tab === "read") return entry.status === "read";
      if (tab === "all") return entry.status !== "ignored";
      if (tab === "ignored") return entry.status === "ignored";
      return false;
    }

    function rowElement(row) {
      const tr = document.createElement("tr");
      tr.appendChild(cell(titleNode(row)));
      tr.appendChild(cell(statusSelect(row)));
      tr.appendChild(cell(text(labels[row.entry.priority] || row.entry.priority)));
      tr.appendChild(cell(text(row.topic)));
      tr.appendChild(cell(text(row.firstSeen || "-")));
      tr.appendChild(cell(actions(row)));
      return tr;
    }

    function titleNode(row) {
      const wrap = document.createElement("div");
      const title = document.createElement("div");
      title.className = "title";
      title.textContent = row.title;
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = \`\${row.arxivId} · \${row.authors || "Unknown authors"}\`;
      const summary = document.createElement("div");
      summary.className = "summary";
      summary.textContent = row.entry.summary?.coreProblem || row.entry.summary?.whyRelevant || "";
      wrap.append(title, meta, summary);
      return wrap;
    }

    function statusSelect(row) {
      const select = document.createElement("select");
      populateSelect(select, state.statuses, "");
      select.value = row.entry.status;
      select.addEventListener("change", () => {
        vscode.postMessage({ type: "setStatus", arxivId: row.arxivId, status: select.value });
      });
      return select;
    }

    function actions(row) {
      const wrap = document.createElement("div");
      wrap.className = "actions";
      for (const [resource, label] of [["note", "Note"], ["daily", "Daily"], ["arxiv", "arXiv"], ["pdf", "PDF"]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", () => {
          vscode.postMessage({ type: "openResource", arxivId: row.arxivId, resource });
        });
        wrap.appendChild(button);
      }
      return wrap;
    }

    function stat(label, value) {
      const el = document.createElement("div");
      el.className = "stat";
      const strong = document.createElement("strong");
      strong.textContent = String(value);
      const span = document.createElement("span");
      span.textContent = label;
      el.append(strong, span);
      return el;
    }

    function cell(child) {
      const td = document.createElement("td");
      td.appendChild(child);
      return td;
    }

    function text(value) {
      return document.createTextNode(value);
    }

    function option(value, label) {
      const el = document.createElement("option");
      el.value = value;
      el.textContent = label;
      return el;
    }
  </script>
</body>
</html>`;
}

function renderLoadingHtml(message) {
  return `<!doctype html><html><body>${escapeHtml(message)}</body></html>`;
}

function safeJson(value) {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function createNonce() {
  return Array.from({ length: 16 }, () => Math.floor(Math.random() * 36).toString(36)).join("");
}

module.exports = {
  DASHBOARD_VIEW_TYPE,
  loadPaperIndex,
  openDashboard,
  renderDashboardHtml,
  resourceTargetForEntry,
  updatePaperStatus,
};
