// People (design §13, §13.1): the grid /people and the person leaf /people/{id}.
//
// Every action is a fetch POST of a `.js-people-form`; none pushes history. A 2xx reply:
// `data-swap="people-cards"` swaps the grid's #people-cards with the reply and clears the
// form; `data-swap="person"` swaps the leaf's #person with the reply's #person, or with
// the page at `data-reload` fetched after the action; `data-close` closes the leaf. A form
// with `data-confirm` asks first. A 4xx/5xx reply shows its `detail` in the error line.
//
// Close (×, the "People" link, Esc) goes up one level: history.back() when the leaf was
// opened from the grid — marked `data-from-grid` at mount when the referrer is /people of
// this origin — else location.replace("/people"). The leaf holds no level of its own.
//
// htmx runs this script again on a boosted nav and on a history restore, so the wiring
// happens ONCE (`window.__peopleWired`) and every listener resolves its nodes at event time.
(() => {
  if (window.__peopleWired) { return; }
  window.__peopleWired = true;

  function leafPage() {
    return document.getElementById('person-page');
  }

  function showError(text) {
    const line = document.getElementById('people-error');
    if (!line) { return; }
    line.textContent = text;
    line.hidden = false;
  }

  function clearError() {
    const line = document.getElementById('people-error');
    if (line) { line.hidden = true; line.textContent = ''; }
  }

  function cameFromGrid(referrer, origin) {
    let url;
    try {
      url = new URL(referrer);
    } catch (_) {
      return false;
    }
    return url.origin === origin && url.pathname === '/people';
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
      location.replace('/people');
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

  function swapById(html, id) {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const fresh = doc.getElementById(id);
    const current = document.getElementById(id);
    if (fresh && current) { current.replaceWith(fresh); }
  }

  async function afterAction(form, response) {
    if (form.hasAttribute('data-close')) {
      close();
      return;
    }
    const target = form.dataset.swap;
    if (!target) { return; }
    let html = await response.text();
    if (form.dataset.reload) {
      const page = await fetch(form.dataset.reload, { cache: 'no-store' });
      if (!page.ok) {
        showError(await detailOf(page));
        return;
      }
      html = await page.text();
    }
    swapById(html, target);
    if (target === 'people-cards') { form.reset(); }
  }

  document.addEventListener('submit', async (event) => {
    const form = event.target;
    if (!form.matches || !form.matches('.js-people-form')) { return; }
    event.preventDefault();
    const question = form.dataset.confirm;
    if (question && !confirm(question)) { return; }
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
    try {
      await afterAction(form, response);
    } catch (_) {
      showError('The server did not answer. Try again.');
    }
  });

  document.addEventListener('click', (event) => {
    if (!leafPage()) { return; }
    if (event.target.closest('.people-close, .people-up')) {
      event.preventDefault();
      close();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !leafPage() || document.querySelector('dialog[open]')) { return; }
    close();
  });

  // A grid restored from bfcache still shows what it showed before the leaf: a person
  // deleted, renamed or given faces there. Refresh its cards and "Faces to name" in place,
  // leaving the New person field and the scroll as they were.
  async function refreshGrid() {
    if (!document.getElementById('people-cards')) { return; }
    try {
      const response = await fetch('/people', { cache: 'no-store' });
      if (!response.ok) { return; }
      const html = await response.text();
      swapById(html, 'people-cards');
      swapById(html, 'faces-to-name');
    } catch (_) {
      // Offline: the grid keeps what it shows.
    }
  }

  document.addEventListener('htmx:historyRestore', () => { mount(); refreshGrid(); });
  document.addEventListener('htmx:afterSettle', mount);
  window.addEventListener('pageshow', (event) => {
    mount();
    if (event.persisted) { refreshGrid(); }
  });
  mount();
})();
