let source = null;

function citeHtml(id) {
  return `<a href="/photo/${id}"><img class="cite" src="/thumb/${id}" alt="photo ${id}"></a>`;
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

// Open server-rendered history scrolled to the latest turn.
window.addEventListener("DOMContentLoaded", scrollToBottom);

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
  source.addEventListener("done", finish);
  source.onerror = finish;
  return false;
}
