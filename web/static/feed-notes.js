// The viewer's notes on a memory (design §11.3) — the panel, independent of playback.
//
// Split out of the player on purpose. Notes are the input to the NEXT generation, so
// the screen where you most want to read and write them is the one where the film does
// not exist yet: the Prepare card. That screen has no canvas, no timeline and no
// `feed-gl.js`, and while the panel lived inside the player it simply was not there.
//
// So this file owns everything about notes that does not need a running film — the
// list, the form, editing, deleting, the bracket legend, the Regenerate nudge — and the
// player borrows it through `window.feedNotes` to float bubbles and to stamp a note
// with the moment it was paused at.
//
// The same list also holds the film's NARRATION, one row per passage (§11.3). A passage
// is never a note: it is not in `all()`, never floats, and saves through its own routes,
// one save at a time, each carrying the `(source, draft, revision)` of the latest reply.
//
// Under the list, "Ask AI to rephrase" proposes new words for those rows, and only Use
// this writes them. Its transcript and proposal are model text, set only as text.
(() => {
  const body = document.body;
  const key = body.dataset.key;
  const lang = body.dataset.lang;
  const panel = document.getElementById('notes-panel');
  if (!panel) return;

  const noteList = document.getElementById('notes-list');
  const noteForm = document.getElementById('note-form');
  const noteText = document.getElementById('note-text');
  const noteAt = document.getElementById('note-at');
  const notesTag = document.getElementById('notes');
  const narrationTag = document.getElementById('narration-data');
  const errorLine = panel.querySelector('.feed-narration-error');
  const errorMessage = errorLine.querySelector('.feed-narration-message');
  const errorTail = errorLine.querySelector('[data-error-tail]');
  const reload = errorLine.querySelector('[data-error-reload]');
  const undo = panel.querySelector('.feed-narration-undo');
  const toggle = document.getElementById('notes-toggle');
  const sheetBody = document.getElementById('notes-body');
  const count = document.getElementById('notes-count');
  const regenerate = document.querySelector('.feed-regen button');
  const rephrase = document.getElementById('rephrase');
  const rephraseLog = document.getElementById('rephrase-log');
  const rephraseForm = document.getElementById('rephrase-form');
  const rephraseText = document.getElementById('rephrase-text');
  const rephraseSend = document.getElementById('rephrase-send');
  const rephraseProposal = document.getElementById('rephrase-proposal');
  const rephraseSummary = document.getElementById('rephrase-summary');
  const rephraseChanges = document.getElementById('rephrase-changes');
  const rephraseOutdated = document.getElementById('rephrase-outdated');
  const rephraseUse = document.getElementById('rephrase-use');
  const rephraseDismiss = document.getElementById('rephrase-dismiss');
  const rephraseProgress = document.getElementById('rephrase-progress');
  const rephraseStage = document.getElementById('rephrase-stage');
  const rephraseCancel = document.getElementById('rephrase-cancel');
  const rephraseRefused = document.getElementById('rephrase-refused');
  const COLLAPSED_KEY = 'ivms777.notes.collapsed';
  // Present only once a film exists. Without it there is no clock and no seeking, and
  // the panel is simply always open.
  const hasPlayer = document.getElementById('stage') !== null;

  let notes = [];
  try {
    notes = JSON.parse((notesTag && notesTag.textContent) || '[]');
  } catch (error) {
    // Loud, not silent. Swallowing this is what hid an autoescaped payload: the panel
    // rendered perfectly and simply never listed a single note.
    console.error('[feed] could not read the notes payload', error);
    notes = [];
  }

  const NO_NARRATION = {
    state: 'none', source: null, draft: null, revision: 0, seekable: false, changed: false, rows: [],
  };
  let narration = NO_NARRATION;
  try {
    narration = JSON.parse((narrationTag && narrationTag.textContent) || 'null') || NO_NARRATION;
  } catch (error) {
    console.error('[feed] could not read the narration payload', error);
  }

  // The server joins every run of whitespace to one space, so a passage is compared the
  // way it will be stored: a newline typed with shift+Enter is not a change.
  const spoken = (text) => text.split(/\s+/).filter(Boolean).join(' ');

  const stamp = (seconds) => {
    const total = Math.max(Math.floor(seconds), 0);
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
  };

  // A note changes nothing until the film is REGENERATED, and nothing else on the page
  // admits that. A note written and then forgotten is worse than one never written —
  // the viewer believes they have been heard.
  const markStale = () => {
    if (regenerate) { regenerate.classList.add('is-stale'); }
  };

  // Enter SUBMITS, shift+Enter is a newline. A textarea does the opposite by default,
  // and this is a scratchpad you type one thought into — reaching for a button after
  // every note is what stops people writing them.
  const submitOnEnter = (field, submit) => {
    field.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        submit();
      }
    });
  };

  // Grow a field to fit its text instead of scrolling inside it. A note is a couple of
  // sentences you want to READ at a glance; a two-line box with its own scrollbar hides
  // most of one and makes the panel a set of little windows. The panel scrolls instead.
  const fitToText = (field) => {
    field.style.height = 'auto';
    field.style.height = `${field.scrollHeight}px`;
  };
  const autoGrow = (field) => {
    field.addEventListener('input', () => fitToText(field));
    fitToText(field);
  };

  const post = async (url, form) => {
    const response = await fetch(url, { method: 'POST', body: form });
    return response.ok ? response.json() : null;
  };

  const saveNote = async (shownNote, text) => {
    // A row kept through a re-render holds the note it was built with; the note's id
    // can survive an edit, so its current text is read here, not from that closure.
    const note = notes.find((n) => n.id === shownNote.id) || shownNote;
    if (text.trim() === note.text) return;
    const form = new FormData();
    form.set('at', note.at);
    form.set('text', text);
    const result = await post(`/feed/${key}/note/${note.id}`, form);
    notes = notes.filter((n) => n.id !== note.id);
    if (result && result.saved) { notes.push(result); }
    markStale();
    render();
  };

  // Collapsed or open is remembered per browser, not per page: whichever the viewer
  // chose, reopening the film and finding the other one is the panel arguing with them.
  const setCollapsed = (collapsed, remember = false) => {
    panel.classList.toggle('is-collapsed', collapsed);
    if (toggle) { toggle.setAttribute('aria-expanded', String(!collapsed)); }
    // Only a DELIBERATE click is remembered. Unfolding the sheet because the film was
    // paused is the player being helpful, not the viewer choosing — persisting it
    // would silently turn the default inside out after one pause.
    if (!remember) return;
    try { localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0'); } catch { /* private mode */ }
  };
  if (toggle) {
    toggle.addEventListener('click', () =>
      setCollapsed(!panel.classList.contains('is-collapsed'), true));
    let stored = null;
    try { stored = localStorage.getItem(COLLAPSED_KEY); } catch { stored = null; }
    // FOLDED on a first load. The sheet sits over the memory's title and description,
    // and the first thing anyone wants on opening a film is to look at it — not to be
    // handed a form. Once the viewer opens it, that choice is what is remembered.
    setCollapsed(stored !== '0');
  }

  // What the list shows now, by row key: the element, its field and the text it was
  // saved with. A re-render keeps the element of a row being typed in (D119).
  let shown = new Map();
  const unsaved = (item) => (item.kind === 'passage'
    ? spoken(item.field.value) !== item.saved
    : item.field.value.trim() !== item.saved);
  const inUse = (item) => item.field === document.activeElement || unsaved(item);

  // A 409 stops every passage save until the viewer loads the current narration: the
  // row numbers may now belong to another draft (D123).
  let frozen = false;
  // The passages a × asked to delete that have no 200 of their own yet. A refused,
  // failed or dropped delete leaves its row and its text, so nothing else marks it
  // unsaved: only its own 200 or a loaded narration clears it (D123).
  const pendingDeletes = new Set();
  // "Load the current narration" is running: no passage save posts until it ends.
  let loading = false;
  // Use this is in flight. The save barrier only proves the rows were saved when it
  // resolved, so nothing may change them until the answer comes.
  let held = false;
  const rowsLocked = () => frozen || held;

  const noteRow = (note) => {
    const row = document.createElement('li');
    const when = document.createElement('button');
    when.type = 'button';
    when.className = 'feed-note-at';
    when.textContent = note.stamp;
    if (hasPlayer) {
      when.title = 'Jump here';
      when.addEventListener('click', () => window.feedNotes.seek(note.at));
    } else {
      // No film yet, so nothing to jump to; the stamp is still worth showing,
      // because it says where this note will land once there is one.
      when.disabled = true;
      when.title = 'Where this lands once the film is made';
    }
    const field = document.createElement('textarea');
    field.rows = 1;
    field.value = note.text;
    // Saved on blur AND on Enter. Blur alone loses a note when the tab is closed
    // straight after typing, which is exactly when a good one gets written.
    field.addEventListener('blur', () => saveNote(note, field.value));
    submitOnEnter(field, () => field.blur());
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'feed-note-remove';
    remove.textContent = '×';
    remove.title = 'Delete';
    remove.addEventListener('click', () => saveNote(note, ''));
    row.append(when, field, remove);
    return { kind: 'note', el: row, field, saved: note.text };
  };

  const passageRow = (passage) => {
    const row = document.createElement('li');
    row.className = 'is-narration';
    const when = document.createElement('button');
    when.type = 'button';
    when.className = 'feed-note-at';
    when.textContent = passage.stamp;
    if (passage.start === null || passage.start === undefined) {
      when.disabled = true;
    } else if (!hasPlayer) {
      when.disabled = true;
      when.title = 'Where this lands once the film is made';
    } else if (!narration.seekable) {
      // A draft of an earlier film: its times are that film's, not this one's.
      when.disabled = true;
      when.title = 'From the film these words were taken from';
    } else {
      when.title = 'Jump here';
      when.addEventListener('click', () => window.feedNotes.seek(passage.start));
    }
    const tag = document.createElement('span');
    tag.className = 'feed-note-tag';
    tag.textContent = 'narration';
    const field = document.createElement('textarea');
    field.rows = 1;
    field.value = passage.text;
    field.readOnly = rowsLocked();
    // A locked row has nothing new, and its old words must not follow a rewrite.
    field.addEventListener('blur', () => {
      if (!field.readOnly) { savePassage(passage.n, field.value); }
    });
    submitOnEnter(field, () => field.blur());
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'feed-note-remove';
    remove.textContent = '×';
    remove.title = 'Delete';
    remove.disabled = rowsLocked();
    remove.addEventListener('click', () => removePassage(passage.n));
    row.append(when, tag, field, remove);
    return { kind: 'passage', el: row, field, remove, saved: passage.text };
  };

  // By time: a passage's start, a note's `at`; a passage first on a tie. Passages with
  // no time come first, in telling order.
  const untimed = (entry) => entry.kind === 'passage'
    && (entry.at === null || entry.at === undefined);
  const byTime = (a, b) => {
    if (untimed(a) || untimed(b)) {
      if (untimed(a) && untimed(b)) return a.order - b.order;
      return untimed(a) ? -1 : 1;
    }
    if (a.at !== b.at) return a.at - b.at;
    if (a.kind !== b.kind) return a.kind === 'passage' ? -1 : 1;
    return a.order - b.order;
  };

  // `replacePassages`: only "Load the current narration" replaces a passage row that
  // is in use; every other render keeps its element, its text, its focus and its caret.
  const render = (replacePassages = false) => {
    const entries = [
      ...narration.rows.map((passage, order) => ({
        key: `p${passage.n}`, kind: 'passage', at: passage.start, order, text: passage.text,
        build: () => passageRow(passage),
      })),
      ...notes.map((note, order) => ({
        key: `n${note.id}`, kind: 'note', at: note.at, order, text: note.text,
        build: () => noteRow(note),
      })),
    ].sort(byTime);
    if (count) { count.textContent = entries.length ? String(entries.length) : ''; }
    const next = new Map();
    const built = [];
    for (const entry of entries) {
      const old = shown.get(entry.key);
      if (old && inUse(old) && !(replacePassages && old.kind === 'passage')) {
        next.set(entry.key, { ...old, saved: entry.text });
      } else {
        const item = entry.build();
        next.set(entry.key, item);
        built.push(item);
      }
    }
    const wanted = new Set([...next.values()].map((item) => item.el));
    for (const child of [...noteList.children]) {
      if (!wanted.has(child)) { child.remove(); }
    }
    // Kept rows never move relative to each other, so only new rows are inserted and
    // a field in focus is never taken out of the document.
    entries.forEach((entry, index) => {
      const { el } = next.get(entry.key);
      if (noteList.children[index] !== el) {
        noteList.insertBefore(el, noteList.children[index] || null);
      }
    });
    shown = next;
    // After it is in the document: `scrollHeight` is 0 for a detached element.
    for (const item of built) { autoGrow(item.field); }
  };

  // The state line, the draft twins of the picker line and the legend lead, and the
  // buttons' titles follow the current rows: the first change makes a draft without a
  // page load (D124).
  const showState = () => {
    const draft = narration.state === 'draft';
    const lines = {
      film: narration.state === 'film',
      'draft-changed': draft && narration.changed,
      'draft-unchanged': draft && !narration.changed,
      draft,
      legacy: narration.state === 'legacy',
    };
    for (const line of panel.querySelectorAll('.feed-narration-state')) {
      line.hidden = !lines[line.dataset.state];
    }
    for (const twin of document.querySelectorAll('[data-narration]')) {
      twin.hidden = (twin.dataset.narration === 'draft') !== draft;
    }
    for (const button of document.querySelectorAll('[data-title-draft]')) {
      const title = draft ? button.dataset.titleDraft : button.dataset.titleNoDraft;
      if (title) { button.title = title; } else { button.removeAttribute('title'); }
    }
    showRephrase();
  };

  const showError = (message, conflict) => {
    errorMessage.textContent = message;
    errorTail.hidden = !conflict;
    reload.hidden = !conflict && pendingDeletes.size === 0;
    errorLine.hidden = false;
  };

  // ONE passage save at a time, in order. A refusal or a network failure bumps the
  // generation, which drops every save queued behind it: those rows stay unsaved, and
  // their next blur posts them again (D123).
  let queue = Promise.resolve();
  let generation = 0;
  const enqueue = (change) => {
    if (frozen || loading) return queue;
    const at = generation;
    queue = queue.then(() => (frozen || at !== generation ? undefined : change()));
    return queue;
  };

  const rowsForm = () => {
    const form = new FormData();
    form.set('lang', lang);
    form.set('source', narration.source || '');
    form.set('draft', narration.draft || '');
    form.set('revision', String(narration.revision));
    return form;
  };

  const apply = (reply) => {
    const before = narration;
    const { saved: _saved, ...rows } = reply;
    narration = rows;
    // A × with no 200 of its own keeps its message and the reload button.
    if (pendingDeletes.size === 0) { errorLine.hidden = true; }
    if (rows.state !== before.state || rows.draft !== before.draft
        || rows.revision !== before.revision) {
      markStale();
    }
    showState();
    render();
  };

  const lockRows = () => {
    for (const item of shown.values()) {
      if (item.kind === 'passage') {
        item.field.readOnly = rowsLocked();
        item.remove.disabled = rowsLocked();
      }
    }
  };

  const freeze = (message) => {
    frozen = true;
    generation += 1;
    lockRows();
    showError(message, true);
    showRephrase();
  };

  // `onSaved` runs on the 200, before the reply is applied.
  const send = async (url, form, onSaved = () => {}) => {
    let response;
    try {
      response = await fetch(url, { method: 'POST', body: form });
    } catch {
      generation += 1;
      showError(errorLine.dataset.offline, false);
      return;
    }
    let reply = null;
    try { reply = await response.json(); } catch { reply = null; }
    if (response.ok && reply && reply.saved) {
      onSaved();
      apply(reply);
    } else if (response.status === 409) {
      freeze(reply && reply.message ? reply.message : '');
    } else {
      generation += 1;
      showError(reply && reply.message ? reply.message : errorLine.dataset.offline, false);
    }
  };

  const savePassage = (n, text) => enqueue(async () => {
    const current = narration.rows.find((passage) => passage.n === n);
    if (!current || spoken(text) === current.text) return;
    const form = rowsForm();
    form.set('text', text);
    await send(`/feed/${key}/narration/${n}`, form);
  });

  // Marked at the click, before it queues: a delete dropped behind a failure never runs.
  const removePassage = (n) => {
    pendingDeletes.add(n);
    return enqueue(() => send(`/feed/${key}/narration/${n}/delete`, rowsForm(),
      () => pendingDeletes.delete(n)));
  };

  const drain = async () => {
    let last;
    do {
      last = queue;
      await last;
    } while (last !== queue);
  };

  // The save barrier before a build: the field in focus is blurred (a passage blur
  // queues its save), the queue drains, and the answer is true only when every passage
  // row is saved, no × waits for its 200 and no 409 froze the panel. A note's own save is
  // not waited for.
  const settled = async () => {
    const active = document.activeElement;
    if (active && panel.contains(active)) { active.blur(); }
    await drain();
    return !frozen && pendingDeletes.size === 0
      && ![...shown.values()].some((item) => item.kind === 'passage' && unsaved(item));
  };

  if (undo) {
    undo.addEventListener('click', async () => {
      if (rephrasePhase === 'applying' || !(await settled())) return;
      if (!confirm(undo.dataset.confirm)) return;
      await enqueue(() => send(`/feed/${key}/narration/discard`, rowsForm()));
    });
  }

  // The one re-render that replaces rows with unsaved text, and only on request: the
  // viewer has had the chance to copy what they typed. It waits for the save in flight,
  // and no passage save posts while it loads; a failed load changes nothing.
  const load = async () => {
    await drain();
    let response;
    try {
      response = await fetch(`/feed/${key}/narration?lang=${lang}`);
    } catch {
      return null;
    }
    try { return response.ok ? await response.json() : null; } catch { return null; }
  };
  reload.addEventListener('click', async () => {
    if (loading) return;
    loading = true;
    let current;
    try { current = await load(); } finally { loading = false; }
    if (!current) {
      errorMessage.textContent = errorLine.dataset.offline;
      return;
    }
    narration = current;
    frozen = false;
    pendingDeletes.clear();
    generation += 1;
    errorLine.hidden = true;
    if (narration.state === 'draft' && narration.changed) { markStale(); }
    showState();
    render(true);
  });

  // "Ask AI to rephrase" (§11.3). The conversation lives in this page session, per memory
  // and language, so a return shows it and polls its job again. No table holds it.
  const REPHRASE_KEY = `ivms777.rephrase.${key}.${lang}`;
  const POLL_MS = 1000;
  const HISTORY_MESSAGES = 6;
  let turns = [];                // { request, reply, error }
  let proposal = null;           // the cumulative patch on screen, against the saved rows
  let job = null;
  let rephrasePhase = 'idle';    // 'idle' | 'asking' | 'running' | 'applying'
  // Bumped by every start and stop: after an await, a reply with an older ticket is dropped.
  let ticket = 0;

  try {
    const stored = JSON.parse(sessionStorage.getItem(REPHRASE_KEY) || 'null');
    if (stored) {
      turns = stored.turns || [];
      proposal = stored.proposal || null;
      job = stored.job || null;
    }
  } catch { /* private mode, or a damaged entry: the conversation starts empty */ }

  const rememberRephrase = () => {
    try {
      sessionStorage.setItem(REPHRASE_KEY, JSON.stringify({ turns, proposal, job }));
    } catch { /* private mode */ }
  };

  const outdated = () => proposal !== null && (proposal.outdated
    || proposal.base.source !== narration.source
    || proposal.base.draft !== (narration.draft || '')
    || proposal.base.revision !== narration.revision);

  const showRephrase = () => {
    const busy = rephrasePhase !== 'idle';
    rephrase.hidden = narration.state !== 'film' && narration.state !== 'draft';
    rephraseText.disabled = frozen;
    rephraseSend.disabled = frozen || busy;
    rephraseProposal.hidden = proposal === null;
    rephraseOutdated.hidden = !outdated();
    rephraseUse.hidden = outdated();
    rephraseUse.disabled = frozen || busy;
    rephraseDismiss.disabled = busy;
    rephraseProgress.hidden = rephrasePhase !== 'running';
  };

  const transcriptLine = (who, text) => {
    const row = document.createElement('li');
    const tag = document.createElement('span');
    tag.className = 'feed-rephrase-who';
    tag.textContent = who;
    const words = document.createElement('span');
    words.textContent = text;
    row.append(tag, words);
    return row;
  };

  const changeRow = (change) => {
    const row = document.createElement('li');
    const when = document.createElement('span');
    when.className = 'feed-rephrase-stamp';
    when.textContent = change.stamp;
    const old = document.createElement('span');
    old.className = 'feed-rephrase-old';
    old.textContent = change.old;
    const next = document.createElement('span');
    next.className = 'feed-rephrase-new';
    next.textContent = `→ ${change.new}`;
    row.append(when, old, next);
    return row;
  };

  const showConversation = () => {
    rephraseLog.replaceChildren(...turns.flatMap((turn) => {
      const answer = turn.error || turn.reply;
      return [transcriptLine('you:', turn.request),
        ...(answer ? [transcriptLine('AI:', answer)] : [])];
    }));
    rephraseSummary.textContent = proposal ? proposal.summary : '';
    rephraseChanges.replaceChildren(...(proposal ? proposal.changes : []).map(changeRow));
  };

  // The transcript grows above the form, and only `.feed-notes-body` scrolls. A viewer in
  // the chat keeps the form in sight; an answer never moves one who is reading the rows.
  const showTurn = () => {
    showConversation();
    showRephrase();
    if (document.activeElement === rephraseText) {
      rephraseForm.scrollIntoView({ block: 'nearest' });
    }
  };

  const refuseRephrase = (message) => {
    rephraseRefused.textContent = message;
    rephraseRefused.hidden = false;
  };

  const endTurn = (outcome) => {
    Object.assign(turns[turns.length - 1], outcome);
    job = null;
    rephrasePhase = 'idle';
    rememberRephrase();
    showTurn();
  };

  const poll = async (mine) => {
    let response = null;
    let state = null;
    try {
      response = await fetch(`/feed/${key}/narration/rephrase/${job}`);
      state = response.status === 404 ? null : await response.json();
    } catch { /* the app reloading: the next poll tells */ }
    if (mine !== ticket) return;
    if (response && response.status === 404) {
      endTurn({ error: rephrase.dataset.lost });
    } else if (state && state.state === 'failed') {
      endTurn({ error: state.error });
    } else if (state && state.state === 'done') {
      const { result } = state;
      proposal = result.changes.length ? { ...result, outdated: state.outdated } : null;
      endTurn({ reply: result.reply || result.summary });
    } else {
      if (state) { rephraseStage.textContent = `${state.stage} · ${state.percent}% (estimate)`; }
      setTimeout(() => { if (mine === ticket) { poll(mine); } }, POLL_MS);
    }
  };

  const stopJob = (id) => {
    fetch(`/feed/${key}/narration/rephrase/${id}/cancel`, { method: 'POST' }).catch(() => {});
  };

  const pastMessages = () => turns.filter((turn) => turn.reply)
    .flatMap((turn) => [{ role: 'user', text: turn.request }, { role: 'ai', text: turn.reply }])
    .slice(-HISTORY_MESSAGES);

  const patch = () => proposal.changes.map((change) => ({ n: change.n, text: change.new }));

  const askAI = async () => {
    const request = rephraseText.value.trim();
    if (!request || rephrasePhase !== 'idle' || frozen) return;
    ticket += 1;
    const mine = ticket;
    rephrasePhase = 'asking';
    rephraseRefused.hidden = true;
    showRephrase();
    const chatting = rephrase.contains(document.activeElement);
    const saved = await settled();
    if (mine !== ticket) return;
    if (chatting) { rephraseText.focus(); }
    if (!saved) {
      rephrasePhase = 'idle';
      refuseRephrase(rephrase.dataset.unsaved);
      showRephrase();
      return;
    }
    const form = rowsForm();
    form.set('request', request);
    form.set('history', JSON.stringify(pastMessages()));
    form.set('proposal', JSON.stringify(proposal && !outdated() ? patch() : []));
    let response = null;
    let reply = null;
    try {
      response = await fetch(`/feed/${key}/narration/rephrase`, { method: 'POST', body: form });
      reply = await response.json();
    } catch { /* said below */ }
    if (mine !== ticket) {
      if (reply && reply.job) { stopJob(reply.job); }
      return;
    }
    if (response && response.status === 202 && reply && reply.job) {
      job = reply.job;
      rephrasePhase = 'running';
      turns.push({ request, reply: null, error: null });
      rephraseText.value = '';
      fitToText(rephraseText);
      rephraseStage.textContent = 'starting · 0% (estimate)';
      rememberRephrase();
      showTurn();
      poll(mine);
      return;
    }
    rephrasePhase = 'idle';
    if (response && response.status === 409 && reply && reply.error === 'stale') {
      freeze(reply.message);
      return;
    }
    refuseRephrase(reply && reply.message ? reply.message : rephrase.dataset.offline);
    showRephrase();
  };

  const cancelRephrase = () => {
    ticket += 1;
    if (job) {
      stopJob(job);
      endTurn({ error: rephrase.dataset.cancelled });
    } else {
      rephrasePhase = 'idle';
      showRephrase();
    }
  };

  // A rewrite changes rows the viewer did not touch, and `render` keeps a focused row, so
  // a clean row is forgotten and rebuilt with the new words. A row with unsaved text stays.
  const forgetCleanRows = () => {
    for (const [rowKey, item] of shown) {
      if (item.kind === 'passage' && !unsaved(item)) { shown.delete(rowKey); }
    }
  };

  const useProposal = async () => {
    if (proposal === null || rephrasePhase !== 'idle' || frozen || outdated()) return;
    rephrasePhase = 'applying';
    rephraseRefused.hidden = true;
    showRephrase();
    const saved = await settled();
    // Locked in the tick the barrier answered: no edit can land between the two.
    held = saved && !outdated();
    lockRows();
    if (held) {
      await enqueue(() => {
        const form = rowsForm();
        form.set('changes', JSON.stringify(patch()));
        return send(`/feed/${key}/narration/rewrite`, form, () => {
          proposal = null;
          rememberRephrase();
          forgetCleanRows();
        });
      });
    } else if (!saved) {
      refuseRephrase(rephrase.dataset.unused);
    }
    held = false;
    rephrasePhase = 'idle';
    lockRows();
    showTurn();
  };

  autoGrow(rephraseText);
  submitOnEnter(rephraseText, () => rephraseForm.requestSubmit());
  rephraseForm.addEventListener('submit', (event) => {
    event.preventDefault();
    askAI();
  });
  rephraseUse.addEventListener('click', useProposal);
  rephraseCancel.addEventListener('click', cancelRephrase);
  rephraseDismiss.addEventListener('click', () => {
    if (rephrasePhase !== 'idle') return;
    proposal = null;
    rememberRephrase();
    showTurn();
  });

  // WHERE the film is standing. Captured when the panel opens and not re-read at
  // submit time: the number shown next to the box has to be the number that gets
  // saved, or a note lands somewhere you did not choose. Zero on the Prepare screen,
  // where a note is about the film as a whole.
  let at = 0;

  if (noteForm) {
    autoGrow(noteText);
    submitOnEnter(noteText, () => noteForm.requestSubmit());
    noteForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const form = new FormData();
      form.set('at', at);
      form.set('text', noteText.value);
      const result = await post(`/feed/${key}/note`, form);
      if (result && result.saved) {
        notes.push(result);
        noteText.value = '';
        fitToText(noteText);          // back to one line after it empties
        markStale();
        render();
      }
    });
  }

  // The player borrows the panel: it opens on a pause and on the title card, and it
  // supplies the clock. Everything else here works without it.
  window.feedNotes = {
    all: () => notes,
    seek: () => {},                      // replaced by the player when there is one
    settled,
    // Esc: a running rephrase is a level of its own. False when none runs.
    escape() {
      if (rephrasePhase !== 'asking' && rephrasePhase !== 'running') return false;
      cancelRephrase();
      return true;
    },
    open(show, seconds = 0) {
      panel.hidden = !show;
      if (!show) return;
      at = seconds;
      noteAt.textContent = stamp(at);
      render();
      // Opening the panel to write something and finding it collapsed is worse than
      // the sheet being in the way, so a deliberate open always unfolds it.
      if (seconds > 0) { setCollapsed(false); }
    },
  };

  showConversation();
  showState();
  if (job && !rephrase.hidden) {
    rephrasePhase = 'running';
    rephraseStage.textContent = 'starting · 0% (estimate)';
    showRephrase();
    ticket += 1;
    poll(ticket);
  }
  // Lit at load only when the draft says something the film does not (D120).
  if (narration.state === 'draft' && narration.changed) { markStale(); }

  // With no film there is nothing to pause, so the panel is simply always open: the
  // Prepare screen is exactly where these are worth writing.
  //
  // EXCEPT while one is being made. That run has already read the notes and cannot act
  // on an edit, so offering the panel would be a lie; the only controls that mean
  // anything then are close and cancel.
  window.feedNotes.open(!hasPlayer && !body.classList.contains('is-building'), 0);
})();
