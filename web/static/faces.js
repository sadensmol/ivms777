// Face groups (design §13, §13.1): the group leaf /faces/{id} and the Faces tab of a
// person's leaf /voices/{id}.
//
// Every action is a fetch POST of a `.js-faces-form`. On the group leaf a 2xx reply is the
// leaf page of the group; its #face-group is swapped in place, and when the group shown
// changed (the last face left it) the URL is replaced, never added to history. On a
// person's Faces tab a 2xx reply takes the group out of the list. A 4xx/5xx reply shows
// its `detail` in the page's error line.
//
// Close (×, the "Voices" link, Esc) goes up one level: history.back() when the leaf was
// opened from the grid — marked `data-from-grid` at mount when the referrer is /voices of
// this origin — else location.replace("/voices"). The leaf holds no level of its own.
//
// htmx runs this script again on a boosted nav and on a history restore, so the wiring
// happens ONCE (`window.__facesWired`) and every listener resolves its nodes at event time.
(() => {
  if (window.__facesWired) { return; }
  window.__facesWired = true;

  function leafPage() {
    return document.getElementById('faces-page');
  }

  function errorLine() {
    return document.getElementById('faces-error') || document.getElementById('voices-error');
  }

  function showError(text) {
    const line = errorLine();
    if (!line) { return; }
    line.textContent = text;
    line.hidden = false;
  }

  function clearError() {
    const line = errorLine();
    if (line) { line.hidden = true; line.textContent = ''; }
  }

  function cameFromGrid(referrer, origin) {
    let url;
    try {
      url = new URL(referrer);
    } catch (_) {
      return false;
    }
    return url.origin === origin && url.pathname === '/voices';
  }

  function mount() {
    const page = leafPage();
    if (page && cameFromGrid(document.referrer, location.origin)) {
      page.setAttribute('data-from-grid', '');
    }
  }

  function close() {
    const page = leafPage();
    if (page && page.hasAttribute('data-from-grid')) {
      history.back();
    } else {
      location.replace('/voices');
    }
  }

  async function detailOf(response) {
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') { return body.detail; }
    } catch (_) {
      // Not JSON: the status line says enough.
    }
    return `The server answered ${response.status}.`;
  }

  function swapLeaf(html) {
    const fresh = new DOMParser().parseFromString(html, 'text/html').getElementById('face-group');
    const current = document.getElementById('face-group');
    if (!fresh || !current) { return; }
    current.replaceWith(fresh);
    const path = `/faces/${fresh.dataset.groupId}`;
    if (location.pathname !== path) { history.replaceState(history.state, '', path); }
  }

  function dropOwnedGroup(form) {
    const card = form.closest('.face-group-owned');
    const list = card && card.parentElement;
    if (!card) { return; }
    card.remove();
    const none = list && list.querySelector('.voice-faces-none');
    if (none && !list.querySelector('.face-group-owned')) { none.hidden = false; }
  }

  document.addEventListener('submit', async (event) => {
    const form = event.target;
    if (!form.matches || !form.matches('.js-faces-form')) { return; }
    event.preventDefault();
    clearError();
    let response;
    try {
      response = await fetch(form.getAttribute('action'), {
        method: 'POST',
        body: new FormData(form),
        cache: 'no-store',
      });
    } catch (_) {
      showError('The server did not answer. Try again.');
      return;
    }
    if (!response.ok) {
      showError(await detailOf(response));
      return;
    }
    if (form.closest('.face-group-owned')) {
      dropOwnedGroup(form);
      return;
    }
    swapLeaf(await response.text());
  });

  document.addEventListener('click', (event) => {
    if (!leafPage()) { return; }
    if (event.target.closest('.faces-close, .faces-up')) {
      event.preventDefault();
      close();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !leafPage() || document.querySelector('dialog[open]')) { return; }
    close();
  });

  document.addEventListener('htmx:historyRestore', mount);
  document.addEventListener('htmx:afterSettle', mount);
  window.addEventListener('pageshow', mount);
  mount();
})();
