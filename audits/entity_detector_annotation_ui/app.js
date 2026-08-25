const ENTITY_TYPES = [
  "PERSON",
  "EMAIL",
  "PHONE",
  "ADDRESS",
  "LOCATION",
  "POSTAL_CODE",
  "DATE_OF_BIRTH",
  "USER_ID",
  "CUSTOMER_ID",
  "ACCOUNT_ID",
  "ORDER_ID",
  "RESERVATION_ID",
  "TICKET_ID",
  "TRANSACTION_ID",
  "TRACKING_ID",
  "PAYMENT_ID",
  "USERNAME",
  "PASSWORD",
  "API_KEY",
  "URL",
  "IP_ADDRESS",
  "FILE_PATH",
  "OTHER_SENSITIVE",
];

const initialParams = new URLSearchParams(window.location.search);
const initialMode = initialParams.get("mode") || "a";
const state = {
  mode: initialMode,
  showSuggestions: initialParams.has("suggestions")
    ? initialParams.get("suggestions") !== "0"
    : initialMode === "a",
  bootstrap: null,
  documents: [],
  filteredDocuments: [],
  currentItemId: null,
  currentDocument: null,
  currentCandidateId: null,
  selection: null,
  saveTimer: null,
};

const elements = Object.fromEntries(
  [
    "run-name",
    "global-progress",
    "save-state",
    "search",
    "domain-filter",
    "status-filter",
    "document-list",
    "document-title",
    "document-meta",
    "previous-document",
    "next-document",
    "review-toggle",
    "selection-toolbar",
    "selection-summary",
    "addition-type",
    "addition-notes",
    "add-selection",
    "cancel-selection",
    "document-text",
    "candidate-progress",
    "unlabeled-only",
    "candidate-list",
    "addition-count",
    "addition-list",
    "toast",
  ].map((id) => [id, document.getElementById(id)])
);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function labelText(value) {
  if (value === "1") return "敏感";
  if (value === "0") return "非敏感";
  return "未判断";
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(
    () => elements.toast.classList.remove("visible"),
    2200
  );
}

function setSaveState(status, message) {
  elements["save-state"].className = `save-state ${status}`;
  elements["save-state"].textContent = message;
}

async function api(path, options = {}) {
  setSaveState("saving", "保存中...");
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    setSaveState("error", "保存失败");
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  setSaveState("", "已保存");
  return data;
}

function typeOptions(selected) {
  const values = ENTITY_TYPES.includes(selected)
    ? ENTITY_TYPES
    : [selected, ...ENTITY_TYPES].filter(Boolean);
  return values
    .map(
      (type) =>
        `<option value="${escapeHtml(type)}" ${
          type === selected ? "selected" : ""
        }>${escapeHtml(type)}</option>`
    )
    .join("");
}

function modeLabel() {
  return state.mode === "adjudicated" ? "仲裁" : `标注员 ${state.mode.toUpperCase()}`;
}

async function loadBootstrap({ preserveCurrent = true } = {}) {
  const data = await api(`/api/bootstrap?mode=${encodeURIComponent(state.mode)}`);
  state.bootstrap = data;
  state.documents = data.documents;
  elements["run-name"].textContent =
    `${data.manifest.run_name} · ${data.manifest.detector_profile}`;
  populateDomains();
  applyFilters();
  updateGlobalProgress();

  const stillExists =
    preserveCurrent &&
    state.currentItemId &&
    state.documents.some((row) => row.item_id === state.currentItemId);
  if (!stillExists) {
    state.currentItemId =
      state.filteredDocuments[0]?.item_id || state.documents[0]?.item_id || null;
  }
  if (state.currentItemId) {
    await loadDocument(state.currentItemId);
  }
}

function populateDomains() {
  const selected = elements["domain-filter"].value;
  const domains = [...new Set(state.documents.map((row) => row.domain))].sort();
  elements["domain-filter"].innerHTML =
    '<option value="">全部领域</option>' +
    domains
      .map(
        (domain) =>
          `<option value="${escapeHtml(domain)}">${escapeHtml(domain)}</option>`
      )
      .join("");
  elements["domain-filter"].value = domains.includes(selected) ? selected : "";
}

function documentIsPending(row) {
  return (
    row.candidate_labeled < row.candidate_total ||
    row.addition_pending > 0 ||
    !row.review_complete
  );
}

function applyFilters() {
  const query = elements.search.value.trim().toLowerCase();
  const domain = elements["domain-filter"].value;
  const status = elements["status-filter"].value;
  state.filteredDocuments = state.documents.filter((row) => {
    if (domain && row.domain !== domain) return false;
    if (
      query &&
      !`${row.item_id} ${row.source_id} ${row.domain} ${row.doc_type}`
        .toLowerCase()
        .includes(query)
    ) {
      return false;
    }
    if (status === "pending" && !documentIsPending(row)) return false;
    if (status === "complete" && documentIsPending(row)) return false;
    if (status === "disagreement" && row.addition_pending === 0 &&
      row.candidate_labeled === row.candidate_total) {
      return false;
    }
    return true;
  });
  renderDocumentList();
}

function renderDocumentList() {
  if (!state.filteredDocuments.length) {
    elements["document-list"].innerHTML =
      '<div class="empty-state">没有符合筛选条件的文档。</div>';
    return;
  }
  elements["document-list"].innerHTML = state.filteredDocuments
    .map((row) => {
      const complete = !documentIsPending(row);
      const warning =
        row.candidate_labeled < row.candidate_total || row.addition_pending > 0;
      const dotClass = complete ? "complete" : warning ? "warning" : "";
      const level = row.level === "" ? "Trace" : `L${row.level}`;
      const pendingAddition = row.addition_pending
        ? ` · ${row.addition_pending} 漏检待仲裁`
        : "";
      return `
        <button class="document-row ${
          row.item_id === state.currentItemId ? "active" : ""
        }" data-item-id="${escapeHtml(row.item_id)}">
          <span class="document-row-title">
            <span class="status-dot ${dotClass}"></span>${escapeHtml(row.item_id)}
          </span>
          <span class="document-row-meta">
            <span>${escapeHtml(row.domain)} · ${level}</span>
            <span>${row.candidate_labeled}/${row.candidate_total}${pendingAddition}</span>
          </span>
        </button>`;
    })
    .join("");
  elements["document-list"]
    .querySelectorAll(".document-row")
    .forEach((button) =>
      button.addEventListener("click", () => loadDocument(button.dataset.itemId))
    );
}

function updateGlobalProgress() {
  const labeled = state.documents.reduce(
    (sum, row) => sum + Number(row.candidate_labeled),
    0
  );
  const total = state.documents.reduce(
    (sum, row) => sum + Number(row.candidate_total),
    0
  );
  const reviewed = state.documents.filter((row) => row.review_complete).length;
  elements["global-progress"].textContent =
    `${labeled} / ${total} 候选 · ${reviewed} / ${state.documents.length} 文档`;
}

async function loadDocument(itemId) {
  state.currentItemId = itemId;
  state.selection = null;
  elements["selection-toolbar"].hidden = true;
  const data = await api(
    `/api/document?mode=${encodeURIComponent(state.mode)}&item_id=${encodeURIComponent(
      itemId
    )}`
  );
  state.currentDocument = data;
  const firstPending = data.candidates.find(
    (row) => row.human_sensitive === ""
  );
  if (
    !data.candidates.some(
      (row) => row.candidate_id === state.currentCandidateId
    )
  ) {
    state.currentCandidateId =
      firstPending?.candidate_id || data.candidates[0]?.candidate_id || null;
  }
  renderDocumentList();
  renderCurrentDocument();
}

function renderCurrentDocument() {
  const data = state.currentDocument;
  if (!data) return;
  const documentRow = data.document;
  elements["document-title"].textContent = documentRow.item_id;
  elements["document-meta"].textContent =
    `${documentRow.domain} · ${documentRow.doc_type} · ${
      documentRow.level === "" ? "Trace" : `L${documentRow.level}`
    } · ${data.text.length} chars`;
  elements["review-toggle"].disabled = false;
  elements["review-toggle"].classList.toggle("complete", data.review_complete);
  elements["review-toggle"].textContent = data.review_complete
    ? "文档已完成"
    : "标记文档完成";
  renderDocumentText();
  renderCandidates();
  renderAdditions();
  updateNavigationButtons();
}

function renderedSpans() {
  const candidates = state.currentDocument.candidates.map((row) => ({
    start: Number(row.start),
    end: Number(row.end),
    id: row.candidate_id,
    kind: "candidate",
  }));
  const additions = state.currentDocument.additions.map((row) => ({
    start: Number(row.start),
    end: Number(row.end),
    id: row.key,
    kind: "addition",
  }));
  const modelExtras = state.showSuggestions
    ? (state.currentDocument.model_extra_suggestions || []).map((row) => ({
        start: Number(row.start),
        end: Number(row.end),
        id: row.suggestion_id,
        kind: "model",
      }))
    : [];
  const grouped = new Map();
  [...candidates, ...additions, ...modelExtras].forEach((span) => {
    const key = `${span.start}:${span.end}`;
    const current = grouped.get(key) || {
      start: span.start,
      end: span.end,
      ids: [],
      kinds: [],
    };
    current.ids.push(span.id);
    current.kinds.push(span.kind);
    grouped.set(key, current);
  });
  return [...grouped.values()]
    .sort((a, b) => a.start - b.start || b.end - a.end)
    .filter((span, index, spans) => {
      const previous = spans[index - 1];
      return !previous || span.start >= previous.end;
    });
}

function renderDocumentText() {
  const text = state.currentDocument.text;
  const spans = renderedSpans();
  let cursor = 0;
  let html = "";
  spans.forEach((span) => {
    html += escapeHtml(text.slice(cursor, span.start));
    const active = span.ids.includes(state.currentCandidateId);
    const addition = span.kinds.includes("addition");
    const model = span.kinds.includes("model");
    html += `<mark class="entity-mark ${active ? "active" : ""} ${
      addition ? "addition-mark" : ""
    } ${model ? "model-mark" : ""
    }" data-start="${span.start}" data-end="${span.end}">${escapeHtml(
      text.slice(span.start, span.end)
    )}</mark>`;
    cursor = span.end;
  });
  html += escapeHtml(text.slice(cursor));
  elements["document-text"].innerHTML = html;
}

function visibleCandidates() {
  const onlyPending = elements["unlabeled-only"].checked;
  return state.currentDocument.candidates.filter(
    (row) => !onlyPending || row.human_sensitive === ""
  );
}

function comparisonHtml(row) {
  if (state.mode !== "adjudicated") return "";
  const view = (name, value) => `
    <div>
      <strong>${name}</strong>
      ${labelText(value.human_sensitive)}
      ${value.human_sensitive === "1" ? ` · ${escapeHtml(value.human_type)}` : ""}
      ${value.notes ? `<br>${escapeHtml(value.notes)}` : ""}
    </div>`;
  return `<div class="comparison">${view("标注员 A", row.annotator_a)}${view(
    "标注员 B",
    row.annotator_b
  )}</div>`;
}

function candidateSuggestionHtml(row) {
  if (!state.showSuggestions) return "";
  const suggestion = row.model_suggestion;
  if (!suggestion) return "";
  const confidence =
    suggestion.confidence == null
      ? ""
      : ` · ${Math.round(Number(suggestion.confidence) * 100)}%`;
  const type =
    suggestion.human_sensitive === "1"
      ? ` · ${escapeHtml(suggestion.entity_type || row.proposed_type)}`
      : "";
  return `
    <div class="model-suggestion">
      <div>
        <strong>DeepSeek 建议</strong>
        ${labelText(suggestion.human_sensitive)}${type}${confidence}
        ${suggestion.reason ? `<br>${escapeHtml(suggestion.reason)}` : ""}
      </div>
      <button class="apply-model-candidate" type="button">采纳</button>
    </div>`;
}

function renderCandidates() {
  const all = state.currentDocument.candidates;
  const labeled = all.filter((row) => row.human_sensitive !== "").length;
  elements["candidate-progress"].textContent = `${labeled} / ${all.length}`;
  const rows = visibleCandidates();
  if (!rows.length) {
    elements["candidate-list"].innerHTML =
      '<div class="empty-state">当前筛选下没有候选。请通读正文并补充漏检实体。</div>';
    return;
  }
  elements["candidate-list"].innerHTML = rows
    .map(
      (row) => `
      <article class="candidate-row ${
        row.candidate_id === state.currentCandidateId ? "active" : ""
      } ${row.human_sensitive !== "" ? "resolved" : ""}"
        data-candidate-id="${escapeHtml(row.candidate_id)}">
        <div class="candidate-top">
          <span class="candidate-text">${escapeHtml(row.entity_text)}</span>
          <span class="candidate-position">${row.start}:${row.end}</span>
        </div>
        <div class="candidate-context">${escapeHtml(
          row.context.replaceAll("\\n", " ")
        )}</div>
        ${candidateSuggestionHtml(row)}
        ${comparisonHtml(row)}
        <div class="decision-row">
          <button class="decision yes ${
            row.human_sensitive === "1" ? "selected" : ""
          }" data-label="1">敏感 <kbd>1</kbd></button>
          <button class="decision no ${
            row.human_sensitive === "0" ? "selected" : ""
          }" data-label="0">非敏感 <kbd>2</kbd></button>
        </div>
        <div class="candidate-fields">
          <select class="candidate-type" aria-label="实体类型">
            ${typeOptions(row.human_type || row.proposed_type)}
          </select>
          <textarea class="candidate-notes" rows="1" placeholder="备注（可选）">${escapeHtml(
            row.notes || ""
          )}</textarea>
        </div>
      </article>`
    )
    .join("");

  elements["candidate-list"]
    .querySelectorAll(".candidate-row")
    .forEach((node) => {
      node.addEventListener("click", () => {
        state.currentCandidateId = node.dataset.candidateId;
        renderDocumentText();
        elements["candidate-list"]
          .querySelectorAll(".candidate-row")
          .forEach((row) =>
            row.classList.toggle(
              "active",
              row.dataset.candidateId === state.currentCandidateId
            )
          );
      });
      node.querySelectorAll(".decision").forEach((button) => {
        button.addEventListener("click", async (event) => {
          event.stopPropagation();
          await saveCandidate(node, button.dataset.label);
        });
      });
      node.querySelector(".apply-model-candidate")?.addEventListener(
        "click",
        async (event) => {
          event.stopPropagation();
          await applyCandidateSuggestion(node);
        }
      );
      node.querySelector(".candidate-type").addEventListener("change", () => {
        queueCandidateSave(node);
      });
      node.querySelector(".candidate-notes").addEventListener("input", () => {
        queueCandidateSave(node);
      });
    });
}

function candidateById(candidateId) {
  return state.currentDocument.candidates.find(
    (row) => row.candidate_id === candidateId
  );
}

async function saveCandidate(node, label = null) {
  const row = candidateById(node.dataset.candidateId);
  if (!row) return;
  if (label !== null) row.human_sensitive = label;
  row.human_type = node.querySelector(".candidate-type").value;
  row.notes = node.querySelector(".candidate-notes").value;
  state.currentCandidateId = row.candidate_id;
  try {
    await api("/api/candidate", {
      method: "POST",
      body: JSON.stringify({
        mode: state.mode,
        candidate_id: row.candidate_id,
        human_sensitive: row.human_sensitive,
        human_type: row.human_type,
        notes: row.notes,
      }),
    });
    const next = state.currentDocument.candidates.find(
      (candidate) => candidate.human_sensitive === ""
    );
    state.currentCandidateId = next?.candidate_id || row.candidate_id;
    await refreshAfterEdit();
  } catch (error) {
    showToast(error.message);
  }
}

async function applyCandidateSuggestion(node) {
  const row = candidateById(node.dataset.candidateId);
  if (!row?.model_suggestion) return;
  const suggestion = row.model_suggestion;
  node.querySelector(".candidate-type").value =
    suggestion.entity_type || row.proposed_type;
  await saveCandidate(node, suggestion.human_sensitive);
  showToast("已采纳候选建议");
}

function queueCandidateSave(node) {
  window.clearTimeout(state.saveTimer);
  state.saveTimer = window.setTimeout(() => saveCandidate(node), 450);
}

function renderAdditions() {
  const additions = state.currentDocument.additions;
  const modelExtras = state.showSuggestions
    ? state.currentDocument.model_extra_suggestions || []
    : [];
  elements["addition-count"].textContent = modelExtras.length
    ? `${additions.length} + ${modelExtras.length}`
    : String(additions.length);
  if (!additions.length && !modelExtras.length) {
    elements["addition-list"].innerHTML =
      '<div class="empty-state">在正文中选择漏检文本，即可添加实体。</div>';
    return;
  }
  const modelHtml =
    state.mode === "adjudicated"
      ? ""
      : modelExtras.map(modelExtraHtml).join("");
  const additionHtml = additions
    .map((row) => {
      if (state.mode === "adjudicated") return adjudicationAdditionHtml(row);
      return `
        <article class="addition-row" data-key="${escapeHtml(row.key)}">
          <div class="addition-top">
            <span class="addition-text">${escapeHtml(row.entity_text)}</span>
            <span class="candidate-position">${row.start}:${row.end}</span>
          </div>
          <div class="candidate-context">${escapeHtml(row.entity_type)}${
            row.notes ? ` · ${escapeHtml(row.notes)}` : ""
          }</div>
          <div class="addition-actions">
            <button class="delete-button">删除</button>
          </div>
        </article>`;
    })
    .join("");
  elements["addition-list"].innerHTML = modelHtml + additionHtml;
  bindModelExtras();
  if (state.mode !== "adjudicated") {
    elements["addition-list"].querySelectorAll(".delete-button").forEach(
      (button) =>
        button.addEventListener("click", async () => {
          const row = button.closest(".addition-row");
          await api("/api/addition/delete", {
            method: "POST",
            body: JSON.stringify({ mode: state.mode, key: row.dataset.key }),
          });
          await reloadCurrentDocument();
          showToast("已删除漏检实体");
        })
    );
  } else {
    bindAdjudicationAdditions();
  }
}

function modelExtraHtml(row) {
  const confidence =
    row.confidence == null ? "" : ` · ${Math.round(Number(row.confidence) * 100)}%`;
  return `
    <article class="addition-row model-extra-row" data-suggestion-id="${escapeHtml(
      row.suggestion_id
    )}" data-start="${row.start}" data-end="${row.end}">
      <div class="addition-top">
        <span class="addition-text">${escapeHtml(row.text)}</span>
        <span class="candidate-position">${row.start}:${row.end}</span>
      </div>
      <div class="candidate-context">DeepSeek 建议 · ${escapeHtml(
        row.entity_type
      )}${confidence}${row.reason ? ` · ${escapeHtml(row.reason)}` : ""}</div>
      <div class="addition-actions">
        <button class="accept-model-extra">采纳漏检建议</button>
        <button class="ignore-model-extra quiet-button">忽略</button>
      </div>
    </article>`;
}

function bindModelExtras() {
  elements["addition-list"].querySelectorAll(".model-extra-row").forEach((node) => {
    node.querySelector(".accept-model-extra").addEventListener("click", () =>
      acceptModelExtra(node)
    );
    node.querySelector(".ignore-model-extra").addEventListener("click", () =>
      ignoreModelExtra(node)
    );
  });
}

function modelExtraById(suggestionId) {
  return (state.currentDocument.model_extra_suggestions || []).find(
    (row) => row.suggestion_id === suggestionId
  );
}

async function markModelSuggestion(node, status) {
  await api("/api/model-suggestion", {
    method: "POST",
    body: JSON.stringify({
      mode: state.mode,
      suggestion_id: node.dataset.suggestionId,
      status,
    }),
  });
}

async function acceptModelExtra(node) {
  const extra = modelExtraById(node.dataset.suggestionId);
  if (!extra) return;
  try {
    await api("/api/addition", {
      method: "POST",
      body: JSON.stringify({
        mode: state.mode,
        item_id: state.currentItemId,
        start: Number(extra.start),
        end: Number(extra.end),
        entity_type: extra.entity_type,
        notes: `model suggestion: ${extra.reason || ""}`.trim(),
      }),
    });
    await markModelSuggestion(node, "accepted");
    await refreshAfterEdit();
    showToast("已采纳漏检建议");
  } catch (error) {
    showToast(error.message);
  }
}

async function ignoreModelExtra(node) {
  try {
    await markModelSuggestion(node, "ignored");
    await reloadCurrentDocument();
    showToast("已忽略漏检建议");
  } catch (error) {
    showToast(error.message);
  }
}

function adjudicationAdditionHtml(row) {
  const decision = row.decision || {};
  const included = decision.included;
  const sources = row.sources.join(" + ");
  const sourceDetails = Object.entries(row.source_types)
    .map(([source, type]) => `${source}: ${type}`)
    .join(" · ");
  return `
    <article class="addition-row" data-key="${escapeHtml(row.key)}">
      <div class="addition-top">
        <span class="addition-text">${escapeHtml(row.entity_text)}</span>
        <span class="candidate-position">${row.start}:${row.end}</span>
      </div>
      <div class="candidate-context">来源 ${sources} · ${escapeHtml(
        sourceDetails
      )}</div>
      <div class="decision-row">
        <button class="decision yes ${
          included === true ? "selected" : ""
        }" data-included="true">纳入</button>
        <button class="decision no ${
          included === false ? "selected" : ""
        }" data-included="false">排除</button>
      </div>
      <div class="addition-fields">
        <select class="addition-type">${typeOptions(
          decision.entity_type || row.entity_type
        )}</select>
        <textarea class="addition-notes" rows="1" placeholder="仲裁备注">${escapeHtml(
          decision.notes || ""
        )}</textarea>
      </div>
    </article>`;
}

function bindAdjudicationAdditions() {
  elements["addition-list"].querySelectorAll(".addition-row").forEach((node) => {
    node.querySelectorAll(".decision").forEach((button) => {
      button.addEventListener("click", () =>
        saveAdditionDecision(node, button.dataset.included === "true")
      );
    });
    node.querySelector(".addition-type").addEventListener("change", () => {
      const selected = node.querySelector(".decision.selected");
      saveAdditionDecision(
        node,
        selected ? selected.dataset.included === "true" : null
      );
    });
    node.querySelector(".addition-notes").addEventListener("change", () => {
      const selected = node.querySelector(".decision.selected");
      saveAdditionDecision(
        node,
        selected ? selected.dataset.included === "true" : null
      );
    });
  });
}

async function saveAdditionDecision(node, included) {
  await api("/api/addition/decision", {
    method: "POST",
    body: JSON.stringify({
      key: node.dataset.key,
      included,
      entity_type: node.querySelector(".addition-type").value,
      notes: node.querySelector(".addition-notes").value,
    }),
  });
  await refreshAfterEdit();
}

async function refreshAfterEdit() {
  const itemId = state.currentItemId;
  const bootstrap = await api(
    `/api/bootstrap?mode=${encodeURIComponent(state.mode)}`
  );
  state.bootstrap = bootstrap;
  state.documents = bootstrap.documents;
  applyFilters();
  updateGlobalProgress();
  if (itemId) await reloadCurrentDocument();
}

async function reloadCurrentDocument() {
  const data = await api(
    `/api/document?mode=${encodeURIComponent(state.mode)}&item_id=${encodeURIComponent(
      state.currentItemId
    )}`
  );
  state.currentDocument = data;
  renderCurrentDocument();
}

function textOffset(root, node, offset) {
  const range = document.createRange();
  range.selectNodeContents(root);
  range.setEnd(node, offset);
  return range.toString().length;
}

function captureSelection() {
  if (state.mode === "adjudicated" || !state.currentDocument) return;
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;
  const range = selection.getRangeAt(0);
  const root = elements["document-text"];
  if (
    !root.contains(range.startContainer) ||
    !root.contains(range.endContainer)
  ) {
    return;
  }
  let start = textOffset(root, range.startContainer, range.startOffset);
  let end = textOffset(root, range.endContainer, range.endOffset);
  if (end < start) [start, end] = [end, start];
  const selectedText = state.currentDocument.text.slice(start, end);
  if (!selectedText.trim()) return;
  state.selection = { start, end, text: selectedText };
  elements["selection-summary"].textContent =
    `"${selectedText.replaceAll("\n", " ").slice(0, 40)}" · ${start}:${end}`;
  elements["selection-toolbar"].hidden = false;
}

async function addSelection() {
  if (!state.selection) return;
  try {
    await api("/api/addition", {
      method: "POST",
      body: JSON.stringify({
        mode: state.mode,
        item_id: state.currentItemId,
        start: state.selection.start,
        end: state.selection.end,
        entity_type: elements["addition-type"].value,
        notes: elements["addition-notes"].value,
      }),
    });
    state.selection = null;
    elements["selection-toolbar"].hidden = true;
    elements["addition-notes"].value = "";
    window.getSelection()?.removeAllRanges();
    await reloadCurrentDocument();
    showToast("已添加漏检实体");
  } catch (error) {
    showToast(error.message);
  }
}

function cancelSelection() {
  state.selection = null;
  elements["selection-toolbar"].hidden = true;
  window.getSelection()?.removeAllRanges();
}

async function toggleReview() {
  if (!state.currentDocument) return;
  const complete = !state.currentDocument.review_complete;
  const unlabeled = state.currentDocument.candidates.filter(
    (row) => row.human_sensitive === ""
  ).length;
  const pendingAdditions =
    state.mode === "adjudicated"
      ? state.currentDocument.additions.filter(
          (row) => row.decision?.included == null
        ).length
      : 0;
  if (complete && (unlabeled || pendingAdditions)) {
    showToast(
      `仍有 ${unlabeled} 个候选和 ${pendingAdditions} 个漏检实体待处理`
    );
    return;
  }
  await api("/api/review", {
    method: "POST",
    body: JSON.stringify({
      mode: state.mode,
      item_id: state.currentItemId,
      complete,
    }),
  });
  await refreshAfterEdit();
}

function updateNavigationButtons() {
  const index = state.filteredDocuments.findIndex(
    (row) => row.item_id === state.currentItemId
  );
  elements["previous-document"].disabled = index <= 0;
  elements["next-document"].disabled =
    index < 0 || index >= state.filteredDocuments.length - 1;
}

function moveDocument(delta) {
  const index = state.filteredDocuments.findIndex(
    (row) => row.item_id === state.currentItemId
  );
  const next = state.filteredDocuments[index + delta];
  if (next) loadDocument(next.item_id);
}

function currentCandidateNode() {
  return elements["candidate-list"].querySelector(
    `[data-candidate-id="${CSS.escape(state.currentCandidateId || "")}"]`
  );
}

function keyboardShortcut(event) {
  if (
    event.metaKey ||
    event.ctrlKey ||
    event.altKey ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)
  ) {
    return;
  }
  const node = currentCandidateNode();
  if (event.key === "1" && node) {
    event.preventDefault();
    saveCandidate(node, "1");
  } else if (event.key === "2" && node) {
    event.preventDefault();
    saveCandidate(node, "0");
  } else if (event.key.toLowerCase() === "j") {
    event.preventDefault();
    moveCandidate(1);
  } else if (event.key.toLowerCase() === "k") {
    event.preventDefault();
    moveCandidate(-1);
  }
}

function moveCandidate(delta) {
  const candidates = visibleCandidates();
  const index = candidates.findIndex(
    (row) => row.candidate_id === state.currentCandidateId
  );
  const next = candidates[Math.max(0, Math.min(candidates.length - 1, index + delta))];
  if (!next) return;
  state.currentCandidateId = next.candidate_id;
  renderDocumentText();
  renderCandidates();
  currentCandidateNode()?.scrollIntoView({ block: "nearest" });
}

function bindEvents() {
  document.querySelectorAll(".mode-tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.mode);
    button.addEventListener("click", async () => {
      state.mode = button.dataset.mode;
      const url = new URL(window.location);
      state.showSuggestions = url.searchParams.has("suggestions")
        ? url.searchParams.get("suggestions") !== "0"
        : state.mode === "a";
      state.currentCandidateId = null;
      document.querySelectorAll(".mode-tabs button").forEach((tab) =>
        tab.classList.toggle("active", tab.dataset.mode === state.mode)
      );
      url.searchParams.set("mode", state.mode);
      window.history.replaceState({}, "", url);
      elements["status-filter"].value =
        state.mode === "adjudicated" ? "disagreement" : "pending";
      await loadBootstrap({ preserveCurrent: false });
      showToast(`已切换到${modeLabel()}`);
    });
  });
  [elements.search, elements["domain-filter"], elements["status-filter"]]
    .forEach((input) => input.addEventListener("input", applyFilters));
  elements["unlabeled-only"].addEventListener("change", renderCandidates);
  elements["document-text"].addEventListener("mouseup", captureSelection);
  elements["document-text"].addEventListener("keyup", captureSelection);
  elements["add-selection"].addEventListener("click", addSelection);
  elements["cancel-selection"].addEventListener("click", cancelSelection);
  elements["review-toggle"].addEventListener("click", toggleReview);
  elements["previous-document"].addEventListener("click", () => moveDocument(-1));
  elements["next-document"].addEventListener("click", () => moveDocument(1));
  document.addEventListener("keydown", keyboardShortcut);
}

async function initialize() {
  elements["addition-type"].innerHTML = typeOptions("PERSON");
  if (!["a", "b", "adjudicated"].includes(state.mode)) state.mode = "a";
  if (state.mode === "adjudicated") {
    elements["status-filter"].value = "disagreement";
  }
  bindEvents();
  try {
    await loadBootstrap({ preserveCurrent: false });
  } catch (error) {
    elements["document-text"].textContent = `无法加载标注数据：${error.message}`;
    showToast(error.message);
  }
}

initialize();
