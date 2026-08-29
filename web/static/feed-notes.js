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
(() => {
  const body = document.body;
  const key = body.dataset.key;
  const panel = document.getElementById('notes-panel');
  if (!panel) return;

  const noteList = document.getElementById('notes-list');
  const noteForm = document.getElementById('note-form');
  const noteText = document.getElementById('note-text');
  const noteAt = document.getElementById('note-at');
  const notesTag = document.getElementById('notes');
  const toggle = document.getElementById('notes-toggle');
  const sheetBody = document.getElementById('notes-body');
  const count = document.getElementById('notes-count');
  const regenerate = document.querySelector('.feed-regen button');
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

  const saveNote = async (note, text) => {
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

  const render = () => {
    if (count) { count.textContent = notes.length ? String(notes.length) : ''; }
    noteList.replaceChildren();
    for (const note of [...notes].sort((a, b) => a.at - b.at)) {
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
      noteList.append(row);
      // After it is in the document: `scrollHeight` is 0 for a detached element.
      autoGrow(field);
    }
  };

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

  // With no film there is nothing to pause, so the panel is simply always open: the
  // Prepare screen is exactly where these are worth writing.
  //
  // EXCEPT while one is being made. That run has already read the notes and cannot act
  // on an edit, so offering the panel would be a lie; the only controls that mean
  // anything then are close and cancel.
  window.feedNotes.open(!hasPlayer && !body.classList.contains('is-building'), 0);
})();
