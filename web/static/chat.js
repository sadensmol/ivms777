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

// Local HH:MM, matching what session_messages renders for reloaded history.
function clockNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Same shape as chat.history.format_elapsed, so a turn reads identically live and
// after a reload: "0.8 s", "12.4 s", "1m 05s".
function formatElapsed(ms) {
  if (ms == null || ms < 0) return "";
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + " s";
  return Math.floor(s / 60) + "m " + String(Math.floor(s % 60)).padStart(2, "0") + "s";
}

function scrollToBottom() {
  const log = document.getElementById("chat-log");
  log.scrollTop = log.scrollHeight;
}

// Open server-rendered history scrolled to the latest turn, and wire the input so
// Enter (and Cmd/Ctrl+Enter) submit — a textarea does not submit on Enter natively.
// This runs on a full page load AND every time the nav's hx-boost swaps a fresh
// <main> into the chat view: DOMContentLoaded does NOT fire on a boosted swap, so we
// must not depend on it. The <script> sits at the end of the content, so #chat-q
// already exists when this runs.
function initChat() {
  const q = document.getElementById("chat-q");
  if (!q || q.dataset.wired) return;  // boost re-runs this script; wire once
  q.dataset.wired = "1";
  scrollToBottom();
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
  // The two global toggles persist immediately: any checkbox change posts the
  // prefs form, which redirects back to /chat with the new state reflected (§10).
  const prefs = document.getElementById("chat-prefs");
  if (prefs) prefs.addEventListener("change", () => prefs.submit());
}
initChat();

// Closing a cited photo returns here via history.back(). The browser may restore
// a frozen (bfcache) snapshot of this page — which can be stale or empty. Chat
// history is authoritative in the DB and re-rendered server-side, so on a restored
// back-navigation, reload to show the real current conversation, never a snapshot.
// Register once on window so a boosted re-run of this script does not stack it.
if (!window.__chatPageshowWired) {
  window.__chatPageshowWired = true;
  window.addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });
}

// One assistant turn: a bubble that starts as a processing indicator and is
// replaced by the streamed answer. Cited photos render inline in the answer as
// thumbnails — there is no separate candidate strip. Returns handles to update.
function addTurn(question) {
  const log = document.getElementById("chat-log");

  const user = el("msg user");
  user.appendChild(el("bubble")).textContent = question;
  user.appendChild(el("msg-meta")).textContent = clockNow();
  log.appendChild(user);

  const assistant = el("msg assistant");
  // Two separate lines, each where it belongs in time: the WAIT above the answer
  // (it happened before it, filled in on `thinking`), the message TIMESTAMP below
  // it like every other message's (filled in when the turn completes).
  const think = el("msg-meta msg-think");
  assistant.appendChild(think);
  const bubble = el("bubble");
  bubble.appendChild(el("typing", "<span></span><span></span><span></span>"));
  assistant.appendChild(bubble);
  const meta = el("msg-meta");
  assistant.appendChild(meta);
  log.appendChild(assistant);
  scrollToBottom();
  return { bubble, assistant, think, meta };
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

  // A "show me a memory" answer streams the memory card (server-rendered, trusted
  // HTML) after the prose — the same Organize card, drillable and paged within the
  // memory (§10). Appended once, below the answer bubble.
  source.addEventListener("memory", (e) => {
    try {
      const html = JSON.parse(e.data || "{}").html;
      if (!html) return;
      const card = el("msg-memory", html);
      turn.assistant.insertBefore(card, turn.meta);  // above the timestamp, which is last
      scrollToBottom();
    } catch (_) { /* card is best-effort */ }
  });

  const finish = () => {
    if (!started) turn.bubble.textContent = "(no answer)";
    // A turn that errored never gets a `done` event, so stamp the time here rather
    // than leave it as the only message in the log without one.
    if (!turn.meta.textContent) turn.meta.textContent = clockNow();
    // The timestamp is added AFTER the last token, so it lands below the fold the
    // stream had scrolled to. Scroll once more now that the turn is complete.
    scrollToBottom();
    setBusy(false);
    input.focus();
    if (source) source.close();
  };
  // The wait, announced with the first token — printed above the answer as it
  // starts, the same place the reloaded history shows it.
  source.addEventListener("thinking", (e) => {
    try {
      const took = formatElapsed(JSON.parse(e.data || "{}").elapsed_ms);
      if (took) turn.think.textContent = "thought for " + took;
    } catch (_) { /* the timing line is best-effort */ }
  });

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
