let source = null;

function citeHtml(id) {
  // ctx=chat makes the photo a leaf of the CHAT grid, so close returns to the
  // conversation, not the library (§13.1).
  return `<a href="/photo/${id}?ctx=chat"><img class="cite" src="/thumb/${id}" alt="photo ${id}"></a>`;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// Escape the model's answer first, then substitute citations — so [photo:ID]
// becomes a thumbnail while any HTML in the answer stays inert (no XSS).
function answerHtml(text) {
  return escapeHtml(text).replace(/\[photo:(\d+)\]/g, (_, id) => citeHtml(id));
}

function el(cls, html) {
  const node = document.createElement("div");
  node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function scrollToBottom() {
  const log = document.getElementById("chat-log");
  log.scrollTop = log.scrollHeight;
}

// Open server-rendered history scrolled to the latest turn, and wire the input
// so Enter and Cmd/Ctrl+Enter both submit (Enter alone is native on a single-line
// input, but this makes both explicit and works even if defaults are prevented).
// Closing a cited photo returns here via history.back(). The browser may restore
// a frozen (bfcache) snapshot of this page — which can be stale or empty. Chat
// history is authoritative in the DB and re-rendered server-side, so on a restored
// back-navigation, reload to show the real current conversation, never a snapshot.
window.addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });

window.addEventListener("DOMContentLoaded", () => {
  scrollToBottom();
  const q = document.getElementById("chat-q");
  // Grow the textarea to fit its content (up to the CSS max-height, then scroll).
  const autogrow = () => { q.style.height = "auto"; q.style.height = q.scrollHeight + "px"; };
  q.addEventListener("input", autogrow);
  q.addEventListener("keydown", (e) => {
    // Enter sends; Shift+Enter inserts a newline (the textarea's default).
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      document.getElementById("chat-form").requestSubmit();
    }
  });
});

// One assistant turn: a bubble that starts as a processing indicator and is
// replaced by the streamed answer. Cited photos render inline in the answer as
// thumbnails — there is no separate candidate strip. Returns handles to update.
function addTurn(question) {
  const log = document.getElementById("chat-log");

  const user = el("msg user");
  user.appendChild(el("bubble")).textContent = question;
  log.appendChild(user);

  const assistant = el("msg assistant");
  const bubble = el("bubble");
  bubble.appendChild(el("typing", "<span></span><span></span><span></span>"));
  assistant.appendChild(bubble);
  log.appendChild(assistant);
  scrollToBottom();
  return { bubble };
}

function setBusy(busy) {
  document.getElementById("chat-q").disabled = busy;
  document.getElementById("chat-send").disabled = busy;
}

function askLibrary(event) {
  event.preventDefault();
  const input = document.getElementById("chat-q");
  const question = input.value.trim();
  if (!question) return false;
  input.value = "";
  input.style.height = "auto";  // collapse the grown textarea back to one row
  if (source) source.close();
  setBusy(true);

  const turn = addTurn(question);
  let buffer = "";
  let started = false;

  source = new EventSource("/chat/stream?q=" + encodeURIComponent(question));

  source.onmessage = (e) => {
    if (!started) {
      turn.bubble.innerHTML = "";  // drop the processing indicator on first token
      started = true;
    }
    buffer += JSON.parse(e.data).delta;
    turn.bubble.innerHTML = answerHtml(buffer);
    scrollToBottom();
  };

  const finish = () => {
    if (!started) turn.bubble.textContent = "(no answer)";
    setBusy(false);
    input.focus();
    if (source) source.close();
  };
  // The done event carries the model's decode speed — show it in the header.
  source.addEventListener("done", (e) => {
    try {
      const stats = JSON.parse(e.data || "{}");
      const tps = document.getElementById("chat-tps");
      if (tps && stats.tok_per_sec != null) tps.textContent = " · " + stats.tok_per_sec + " tok/s";
    } catch (_) { /* stats are best-effort */ }
    finish();
  });
  source.onerror = finish;
  return false;
}
