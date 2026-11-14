// Voices (design §6, §11.1, §13, §13.1): the grid /voices and its leaf /voices/{id}.
// Every action is a fetch: a 2xx reply swaps #voice-cards (grid) or #voices (leaf), a
// 4xx/5xx reply shows its `detail` in #voices-error. Nothing pushes history, except the
// one grid→leaf push (a card, or the voice just created).
//
// Lifecycle (D68): htmx runs this script again on a boosted nav and on a history restore,
// so the wiring happens ONCE (`window.__voicesWired`), every listener sits on `document`
// (or `window` for pageshow/pagehide) and resolves its nodes at event time. `mount()`
// binds a session to the current #voices-page; `unmount()` stops the microphone and every
// timer, and a reply that lands after it changes nothing.
//
// Every example the server gets is made HERE (D21): the recording or the picked file is
// decoded in an AudioContext at 48 kHz, mixed to mono and encoded as a PCM-16 WAV. The
// server has no audio decoder beyond stdlib `wave`, so it accepts only that format.
(() => {
  const RATE = 48000;
  const MIN_SECONDS = 5;
  const MAX_SECONDS = 30;
  const POLL_MS = 2000;
  const TABS = ['recordings', 'test'];
  const STATUS_URL = '/voices/preview';
  const HEARING_URL = '/voices/hearing';
  const DISCARD = 'Discard the recording you have not saved?';

  let mounted = null;    // the lifecycle session of the current page
  let sync = null;       // the leaf's view of the test job (previewSync)
  let pollTimer = null;  // the leaf's deleting poll
  let cardsTimer = null; // the grid's deleting poll
  let previewTimer = null;
  let hearingTimer = null;  // the leaf's hearing poll
  let bodyTimer = null;     // the leaf's pending body refresh
  let prepared = null;   // { voiceId, language, promptId, kit } before Start recording
  let live = null;       // a recording in progress
  let take = null;       // an unsaved example: being prepared (wav null) or ready
  const lastSaved = {};  // language -> the prompt id of the last SAVED take on this page

  // Close goes back only when the previous entry is the grid: a card is a full
  // navigation, so the referrer names the grid exactly then (§13.1, D67).
  function cameFromGrid(referrer, origin) {
    let url;
    try {
      url = new URL(referrer);
    } catch (_) {
      return false;
    }
    return url.origin === origin && url.pathname === '/voices';
  }

  function tabFrom(search) {
    const tab = new URLSearchParams(search).get('tab');
    return TABS.includes(tab) ? tab : 'recordings';
  }

  // A recording or an unsaved take is lost by leaving; a prepared recorder is not.
  function leaveNeedsConfirm(state) {
    return state === 'live' || state === 'take';
  }

  // Whether a new hearing job line changes what the leaf body shows: the work of this voice
  // (heard now, or queued after another voice) started or ended, or another pass of it
  // replaced the one heard now. `before`/`after`: { state, run, voiceId }.
  function hearingChanged(before, after, voiceId) {
    const busy = (job) => !!job && (job.state === 'running' || job.state === 'queued')
      && job.voiceId === voiceId;
    if (busy(before) !== busy(after)) { return true; }
    return busy(before) && before.state === 'running' && after.state === 'running'
      && before.run !== after.run;
  }

  // One session per mounted page. `deps`: page() → the current #voices-page or null,
  // setTimeout, clearTimeout, fetch, start(session, page), stop(session). A session's
  // timers die with it, its microphone tracks are stopped, and a fetch — its headers, its
  // `text()` and its `json()` — that answers after it ended never settles, so a late reply
  // changes nothing.
  function lifecycle(deps) {
    let generation = 0;
    let current = null;

    function release() {
      const held = current;
      current = null;
      if (!held) { return; }
      held.tracks.forEach((track) => track.stop());
      held.tracks.clear();
      held.timers.forEach((id) => deps.clearTimeout(id));
      held.timers.clear();
      deps.stop(held.session);
    }

    function mount() {
      const page = deps.page();
      if (!page || page.hasAttribute('data-mounted')) { return false; }
      generation += 1;
      release();
      page.setAttribute('data-mounted', '');
      const at = generation;
      const timers = new Set();
      const tracks = new Set();
      const never = () => new Promise(() => {});
      const settle = (promise) => promise.then(
        (value) => (session.alive() ? value : never()),
        (error) => (session.alive() ? Promise.reject(error) : never()),
      );
      const guarded = (reply) => ({
        ok: reply.ok,
        status: reply.status,
        headers: reply.headers,
        text: () => (session.alive() ? settle(reply.text()) : never()),
        json: () => (session.alive() ? settle(reply.json()) : never()),
      });
      const session = {
        page,
        alive: () => at === generation,
        schedule(run, ms) {
          if (!session.alive()) { return null; }
          const id = deps.setTimeout(() => {
            timers.delete(id);
            if (session.alive()) { run(); }
          }, ms);
          timers.add(id);
          return id;
        },
        cancel(id) {
          if (timers.delete(id)) { deps.clearTimeout(id); }
        },
        fetch(url, options) {
          if (!session.alive()) { return never(); }
          return settle(deps.fetch(url, options)).then(guarded);
        },
        hold(stream) { stream.getTracks().forEach((track) => tracks.add(track)); },
        timers: () => timers.size,
      };
      current = { session, timers, tracks };
      deps.start(session, page);
      return true;
    }

    function unmount() {
      generation += 1;
      release();
      const page = deps.page();
      if (page) { page.removeAttribute('data-mounted'); }
    }

    return {
      mount,
      unmount,
      pageshow(event) { if (event.persisted) { mount(); } },
      session: () => (current ? current.session : null),
    };
  }

  // The test runs in `app`; the page shows its job fragment. Every reply carries the
  // process `epoch`, the test `run` and `completed` (the run of the last stored test).

  // What to do with a job fragment, given the last one applied (`seen`) and the epochs
  // this page has left. Runs are compared only within one epoch.
  function previewVerdict(seen, retired, fragment) {
    if (retired.has(fragment.epoch)) { return 'ignore'; }
    if (fragment.epoch !== seen.epoch) { return 'reconcile'; }
    if (fragment.run < seen.run) { return 'ignore'; }
    if (fragment.run === seen.run && fragment.state === 'running'
        && (seen.state === 'done' || seen.state === 'failed')) { return 'ignore'; }
    return 'apply';
  }

  // The identity of a stored test, compared by equality of both fields, never by order.
  function completionKey(fragment) {
    if (!fragment.completed) { return null; }
    return { epoch: fragment.epoch, completed: fragment.completed };
  }

  // One list refresh for `key`. It stops, with no replace and no marker, once `key` is
  // no longer the target or the generation moved on; it tries again in 1 s while the
  // user types in the list or the fetch fails. Only a body the gate accepts marks it.
  async function refreshLoop(key, deps) {
    const { generation } = deps;
    const current = () => deps.isCurrent(key, generation);
    const again = () => deps.schedule(() => refreshLoop(key, deps), 1000);
    if (!current()) { return; }
    if (deps.typing()) { again(); return; }
    let html;
    try {
      const response = await deps.fetchList();
      if (!current()) { return; }
      if (!response.ok) { again(); return; }
      html = await response.text();
    } catch (_) {
      if (current()) { again(); }
      return;
    }
    if (!current()) { return; }
    if (deps.typing()) { again(); return; }
    if (deps.replaceList(html) === 'accept') { deps.markRefreshed(key); }
  }

  // The page's view of the job: which fragment to apply, when to reconcile after a
  // restart, when to refresh the list, and which #voices body may replace the list.
  // `generation` bumps on every mutation and reconciliation; a reply of an older
  // generation is dropped. `list` is the certificate of the page's own #voices.
  function previewSync(deps, job, list) {
    let generation = 0;
    let previewSeen = { epoch: list.epoch, run: job.run, state: job.state, completed: list.completed };
    const retiredEpochs = new Set();
    let refreshTarget = null;
    let refreshGeneration = -1;
    let previewRefreshed = { epoch: list.epoch, completed: list.completed };  // the page itself
    let listWanted = false;  // a refused body left the list behind the server
    let repairOwner = 0;     // the ticket of the one repair loop; 0 = none
    let tickets = 0;
    const same = (a, b) => !!a && !!b && a.epoch === b.epoch && a.completed === b.completed;
    const isCurrent = (key, at) => same(key, refreshTarget) && at === generation;
    const markRefreshed = (key) => { if (same(key, previewRefreshed)) { deps.refreshed(key); } };

    function refresh(key) {
      if (same(key, previewRefreshed)) { return; }
      if (same(key, refreshTarget) && refreshGeneration === generation) { return; }
      refreshTarget = key;
      refreshGeneration = generation;
      refreshLoop(key, {
        fetchList: deps.fetchList, replaceList: deps.replaceList, typing: deps.typing,
        schedule: deps.schedule, generation, isCurrent, markRefreshed,
      });
    }

    function take(fragment) {
      previewSeen = {
        epoch: fragment.epoch, run: fragment.run, state: fragment.state, completed: fragment.completed,
      };
      deps.apply(fragment);
      if (fragment.state === 'running') { deps.poll(); }
    }

    // Moves the generation on and retires the repair owner; its loop stops at its next check.
    function retire() {
      generation += 1;
      deps.stopTimers();
      repairOwner = 0;
      return generation;
    }

    function bump() {
      retire();
      if (listWanted) { repair(); }
      return generation;
    }

    function show(fragment, at) {
      if (at !== generation) { return 'dropped'; }
      const verdict = previewVerdict(previewSeen, retiredEpochs, fragment);
      if (verdict === 'apply') {
        take(fragment);
        const key = completionKey(fragment);
        if (key) { refresh(key); }
      } else if (verdict === 'reconcile') {
        reconcile(retire(), previewSeen.state === 'running');
      }
      return verdict;
    }

    // An unknown epoch: the app restarted. Only a fresh status read adopts the new one.
    function reconcile(at, wasRunning) {
      const retry = () => deps.schedule(() => { if (at === generation) { reconcile(at, wasRunning); } }, 1000);
      deps.fetchStatus().then((fragment) => {
        if (at !== generation) { return; }
        if (retiredEpochs.has(fragment.epoch)) { retry(); return; }
        if (fragment.epoch === previewSeen.epoch) { show(fragment, at); return; }
        retiredEpochs.add(previewSeen.epoch);
        take(fragment);
        if (wasRunning && fragment.state === 'idle') { deps.restarted(); }
        // A test stored just before the restart lives in SQLite, while the new process
        // may report `completed` 0: refresh the list once for the new process.
        refresh({ epoch: fragment.epoch, completed: fragment.completed });
        if (listWanted) { repair(); }
      }, () => { if (at === generation) { retry(); } });
    }

    // The gate of every #voices body. It accepts only the adopted epoch with the
    // `completed` this page has seen; an unknown epoch reconciles, and the body is
    // never applied. A refused body leaves the list to one repair loop.
    function acceptList(cert) {
      if (cert && !retiredEpochs.has(cert.epoch) && cert.epoch !== previewSeen.epoch) {
        reconcile(retire(), previewSeen.state === 'running');
        return 'reconcile';
      }
      if (!same(cert, previewSeen)) {
        listWanted = true;
        repair();
        return 'reject';
      }
      previewRefreshed = { epoch: cert.epoch, completed: cert.completed };
      if (same(cert, refreshTarget)) { refreshTarget = null; }
      listWanted = false;
      return 'accept';
    }

    function repair() {
      if (repairOwner !== 0) { return; }
      tickets += 1;
      repairOwner = tickets;
      repairList(tickets);
    }

    // Reads the test state, then the list, until a body passes the gate. GETs only:
    // a refused mutation reply is never sent again.
    async function repairList(ticket) {
      const at = generation;
      const owned = () => repairOwner === ticket && at === generation && listWanted;
      const release = () => { if (repairOwner === ticket) { repairOwner = 0; } };
      const again = () => deps.schedule(() => { if (owned()) { repairList(ticket); } else { release(); } }, 1000);
      if (!owned()) { release(); return; }
      if (deps.typing()) { again(); return; }
      try {
        const fragment = await deps.fetchStatus();
        if (!owned()) { release(); return; }
        show(fragment, at);
        if (!owned()) { release(); return; }
        const response = await deps.fetchList();
        if (!owned()) { release(); return; }
        if (!response.ok) { again(); return; }
        const html = await response.text();
        if (!owned()) { release(); return; }
        if (deps.typing()) { again(); return; }
        deps.replaceList(html);
      } catch (_) {
        if (owned()) { again(); } else { release(); }
        return;
      }
      if (owned()) { again(); } else { release(); }
    }

    return { bump, show, acceptList, generation: () => generation };
  }

  const pageNode = () => document.getElementById('voices-page');
  const onGrid = () => { const page = pageNode(); return !!page && !page.dataset.voiceId; };
  const leafUrl = () => `/voices/${pageNode().dataset.voiceId}`;
  const recorderNode = () => document.getElementById('voice-recorder');
  const recorderPart = (selector) => recorderNode().querySelector(selector);
  const recorderState = () => (live ? 'live' : take ? 'take' : prepared ? 'prepared' : 'idle');

  function showError(text) {
    const box = document.getElementById('voices-error');
    if (box) { box.textContent = text; box.hidden = false; }
  }

  function clearError() {
    const box = document.getElementById('voices-error');
    if (box) { box.textContent = ''; box.hidden = true; }
  }

  const canRecord = () => window.isSecureContext && !!navigator.mediaDevices
    && typeof navigator.mediaDevices.getUserMedia === 'function'
    && typeof window.MediaRecorder === 'function';

  // Shows the tab `name`; `record` also replaces the URL (a tab is never a history entry).
  function showTab(name, record) {
    const page = pageNode();
    if (!page || !page.dataset.voiceId) { return; }
    page.dataset.tab = name;
    for (const button of page.querySelectorAll('[role="tab"][data-tab]')) {
      button.setAttribute('aria-selected', String(button.dataset.tab === name));
    }
    for (const part of page.querySelectorAll('[data-tab-part]')) {
      part.hidden = part.dataset.tabPart !== name;
    }
    if (!record) { return; }
    const url = new URL(location.href);
    url.searchParams.set('tab', name);
    history.replaceState(history.state, '', url);
  }

  // Applied after every swap of #voices: the tab, the test panel, and no Record without
  // a secure context.
  function decorate() {
    const page = pageNode();
    showTab(page.dataset.tab || tabFrom(location.search), false);
    syncTest();
    if (canRecord()) { return; }
    for (const button of document.querySelectorAll('#voices .voice-record, #voices .kit-record')) {
      button.hidden = true;
    }
    for (const note of document.querySelectorAll('#voices .voice-record-note')) { note.hidden = false; }
  }

  function cardsOf(html) {
    const holder = document.createElement('div');
    holder.innerHTML = html.trim();
    return holder.querySelector('#voice-cards');
  }

  // A reply to Create carries the new card with `data-created`: open it (the one push).
  function replaceCards(html) {
    const next = cardsOf(html);
    const current = document.getElementById('voice-cards');
    if (!next || !current) { return; }
    current.replaceWith(next);
    const created = next.querySelector('[data-created]');
    if (created) { location.assign(created.getAttribute('href')); return; }
    pollCards();
  }

  function pollCards() {
    if (cardsTimer !== null
        || !document.querySelector('#voice-cards [data-deleting]')) { return; }
    const own = mounted;
    cardsTimer = own.schedule(() => { cardsTimer = null; refreshCards(own); }, POLL_MS);
  }

  // One read of the cards: at mount (a return from the leaf, restored from bfcache, shows
  // a delete made there) and on every poll tick.
  async function refreshCards(own) {
    try {
      const response = await own.fetch('/voices/list', { cache: 'no-store' });
      if (response.ok) {
        const html = await response.text();
        if (own.alive()) { replaceCards(html); }
        return;
      }
    } catch (_) { /* the next tick tries again */ }
    if (own.alive()) { pollCards(); }
  }

  function listOf(html) {
    const holder = document.createElement('div');
    holder.innerHTML = html.trim();
    return holder.querySelector('#voices');
  }

  // The test state a #voices body was read under: `{epoch, completed}`, or null.
  function certOf(element) {
    const { epoch, completed } = element.dataset;
    if (!epoch || completed === undefined) { return null; }
    return { epoch, completed: Number(completed) };
  }

  // The choices a swap must not reset: the take language and the test's language and text.
  function carryInputs(from, to) {
    for (const selector of ['.voice-language', '.test-language', '.test-text']) {
      const before = from.querySelector(selector);
      const after = to.querySelector(selector);
      if (before && after && [...after.options].some((option) => option.value === before.value)) {
        after.value = before.value;
      }
    }
  }

  // The recorder opens in the kit row whose Record opened it, else right under "Add a
  // take", else at the end of the takes (D68).
  function placeRecorder(list, recorder) {
    const held = live ? live.target : (take || prepared);
    const row = held && held.kit ? [...list.querySelectorAll('.kit-row')].find(
      (node) => node.dataset.lang === held.language && node.dataset.promptId === held.promptId) : null;
    if (row) { row.append(recorder); return; }
    const add = list.querySelector('.voice-add');
    if (add) { add.after(recorder); return; }
    (list.querySelector('[data-tab-part="recordings"]') || list).append(recorder);
  }

  // Every #voices body passes `s.acceptList` (`s`: the sync of the page that read it): only
  // a body read under the test state this page has adopted replaces the list; a refused
  // one leaves a repair to `s`. The recorder is detached first, so a live take survives
  // the swap (D68).
  function replaceList(html, s) {
    const next = listOf(html);
    const verdict = s.acceptList(next ? certOf(next) : null);
    if (verdict !== 'accept') { return verdict; }
    const current = document.getElementById('voices');
    carryInputs(current, next);
    const recorder = document.getElementById('voice-recorder');
    const held = !!recorder && current.contains(recorder);
    if (held) { recorder.remove(); }
    current.replaceWith(next);
    if (held) { placeRecorder(next, recorder); }
    decorate();
    pollWhileDeleting();
    // A saved take starts the hearing job: its reply body already shows it running.
    markHearing();
    if (hearingBusyHere()) { pollHearing(); }
    return verdict;
  }

  // A swap while the user types a new name would lose it: wait for the next tick.
  const typingInList = () => {
    const active = document.activeElement;
    return !!active && active.closest('#voices') !== null && active.matches('input[type="text"]');
  };

  function pollWhileDeleting() {
    if (pollTimer !== null || !document.querySelector('#voices [data-deleting]')) { return; }
    const own = mounted;
    const s = sync;
    pollTimer = own.schedule(async () => {
      pollTimer = null;
      let response = null;
      try {
        response = await own.fetch(`${leafUrl()}/list`, { cache: 'no-store' });
        if (response.ok && !typingInList()) {
          const html = await response.text();
          if (own.alive()) { replaceList(html, s); }
          return;
        }
      } catch (_) { /* the next tick tries again */ }
      // The drain removed the whole voice: there is nothing left to poll.
      if (response && response.status === 404) { return; }
      if (own.alive()) { pollWhileDeleting(); }
    }, POLL_MS);
  }

  async function detailOf(response) {
    try {
      const body = await response.json();
      if (body && typeof body.detail === 'string') { return body.detail; }
      if (body && body.detail) { return JSON.stringify(body.detail); }
    } catch (_) { /* not JSON */ }
    return `The server refused this (${response.status}).`;
  }

  async function send(url, body) {
    const own = mounted;
    const s = sync;
    clearError();
    let response;
    let html;
    try {
      response = await own.fetch(url, { method: 'POST', body });
      if (response.ok) { html = await response.text(); }
    } catch (_) {
      if (own.alive()) { showError('The server did not answer. Try again.'); }
      return false;
    }
    if (!response.ok) {
      const detail = await detailOf(response);
      if (own.alive()) { showError(detail); }
      return false;
    }
    if (!own.alive()) { return false; }
    // The action is done even when its body is refused: a repair shows a fresh list.
    if (onGrid()) { replaceCards(html); } else { replaceList(html, s); }
    return true;
  }

  // A recording starts PREPARED: the reading text is chosen before the microphone opens.

  // Shows the instruction and the text of `promptId` in `language`, and the line that the
  // text is only a suggestion; no id (an upload) hides them all.
  function showPrompt(language, promptId) {
    for (const line of recorderNode().querySelectorAll('.recorder-prompt, .recorder-instruction')) {
      line.hidden = line.dataset.lang !== language || line.dataset.promptId !== promptId;
    }
    recorderPart('.recorder-hint').hidden = !promptId;
  }

  // The text after `after` in `language`, wrapping to the first; the first for none.
  function nextPromptId(language, after) {
    const ids = [...recorderNode().querySelectorAll('.recorder-prompt')]
      .filter((prompt) => prompt.dataset.lang === language)
      .map((prompt) => prompt.dataset.promptId);
    return ids[(ids.indexOf(after) + 1) % ids.length];
  }

  // Ends the PREPARED state: the live recording or the take takes the panel over.
  function openPanel(target, recording) {
    prepared = null;
    showPrompt(target.language, target.promptId);
    recorderPart('.recorder-status').textContent = '';
    recorderPart('.recorder-prepare').hidden = true;
    recorderPart('.recorder-live').hidden = !recording;
    recorderPart('.recorder-take').hidden = true;
    recorderNode().hidden = false;
  }

  function closePanel() {
    prepared = null;
    const panel = recorderNode();
    if (!panel) { return; }
    panel.hidden = true;
    recorderPart('.recorder-prepare').hidden = true;
    recorderPart('.recorder-live').hidden = true;
    recorderPart('.recorder-take').hidden = true;
    recorderPart('.recorder-preview').removeAttribute('src');
  }

  // Record binds the voice and the language of this click. A kit row's Record keeps that
  // row's text; Add a take's shows the text after the one last saved in that language,
  // with "Different text". The recorder opens under the row that opened it.
  function prepare(target) {
    if (live || take) { return; }
    clearError();
    prepared = target.kit ? { ...target }
      : { ...target, promptId: nextPromptId(target.language, lastSaved[target.language]) };
    placeRecorder(document.getElementById('voices'), recorderNode());
    showPrompt(prepared.language, prepared.promptId);
    recorderPart('.recorder-next').hidden = !!prepared.kit;
    recorderPart('.recorder-status').textContent = '';
    recorderPart('.recorder-live').hidden = true;
    recorderPart('.recorder-take').hidden = true;
    recorderPart('.recorder-prepare').hidden = false;
    recorderNode().hidden = false;
  }

  function differentText() {
    if (!prepared) { return; }
    prepared.promptId = nextPromptId(prepared.language, prepared.promptId);
    showPrompt(prepared.language, prepared.promptId);
  }

  // What the user picked: a kit row's language and text, or "Add a take"'s language.
  function chosen(control) {
    const list = control.closest('#voices');
    const box = list ? list.querySelector('.voice-add') : null;
    if (!box) { return null; }
    const row = control.closest('.kit-row');
    if (row) {
      return { voiceId: box.dataset.voiceId, language: row.dataset.lang, promptId: row.dataset.promptId, kit: true };
    }
    return { voiceId: box.dataset.voiceId, language: box.querySelector('.voice-language').value };
  }

  const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

  // Start consumes the PREPARED state: the voice, the language and the text are frozen.
  async function startRecording() {
    if (!prepared || live || take) { return; }
    const target = prepared;
    const timer = recorderPart('.recorder-timer');
    const status = recorderPart('.recorder-status');
    const pause = recorderPart('.recorder-pause');
    clearError();
    // The microphone wait is already LIVE, so every `live || take` guard blocks it.
    // `elapsed` is the recorded time of the spans before a pause; `spanStart` is when the
    // running span began, null while paused.
    const session = { recorder: null, stream: null, discarded: false, tick: null, target,
                      elapsed: 0, spanStart: null };
    live = session;
    openPanel(target, true);
    timer.textContent = clock(0);
    pause.textContent = 'Pause';
    pause.disabled = true;
    status.textContent = 'Waiting for the microphone…';
    // Echo cancellation is off, so a playing Original or Generated would be recorded.
    document.querySelectorAll('audio').forEach((audio) => audio.pause());
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (_) {
      // A wait cancelled by Stop, Esc or a leave ends quietly: the page has moved on.
      if (live !== session) { return; }
      live = null;
      closePanel();
      showError('The microphone is not available. Allow it, or upload a file instead.');
      return;
    }
    if (live !== session) {
      // Cancelled while waiting: release the microphone and make no take.
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    mounted.hold(stream);
    status.textContent = '';
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    session.recorder = recorder;
    session.stream = stream;
    recorder.ondataavailable = (event) => { if (event.data.size) { chunks.push(event.data); } };
    recorder.onstop = () => {
      clearInterval(session.tick);
      stream.getTracks().forEach((track) => track.stop());
      if (live !== session) { return; }
      live = null;
      recorderPart('.recorder-live').hidden = true;
      if (session.discarded) { closePanel(); return; }
      makeTake(new Blob(chunks, { type: recorder.mimeType }), 'recorded', target);
    };
    recorder.start();
    session.spanStart = performance.now();
    pause.disabled = false;
    // The timer and the 30 s limit count recorded time only: a pause adds nothing.
    session.tick = setInterval(() => {
      const running = session.spanStart === null ? 0 : performance.now() - session.spanStart;
      const seconds = (session.elapsed + running) / 1000;
      timer.textContent = clock(seconds);
      if (seconds >= MAX_SECONDS && recorder.state === 'recording') {
        pause.disabled = true;
        recorder.stop();
      }
    }, 200);
  }

  // Pause and Resume act on the same recorder, stream and chunks: the pauses join into one
  // take. A recording still waiting for the microphone, or already stopped, has nothing to
  // pause.
  function togglePause() {
    const session = live;
    const recorder = session ? session.recorder : null;
    if (!recorder) { return; }
    const button = recorderPart('.recorder-pause');
    const status = recorderPart('.recorder-status');
    if (recorder.state === 'recording') {
      recorder.pause();
      session.elapsed += performance.now() - session.spanStart;
      session.spanStart = null;
      button.textContent = 'Resume';
      status.textContent = 'Paused — the pause is not recorded.';
    } else if (recorder.state === 'paused') {
      // Echo cancellation is off, so audio started during the pause would be recorded.
      document.querySelectorAll('audio').forEach((audio) => audio.pause());
      recorder.resume();
      session.spanStart = performance.now();
      button.textContent = 'Pause';
      status.textContent = '';
    }
  }

  function stopRecording(discard) {
    if (!live) { return; }
    // Still waiting for the microphone: nothing was recorded, so close at once.
    if (!live.recorder) { live = null; closePanel(); return; }
    live.discarded = discard;
    recorderPart('.recorder-pause').disabled = true;
    if (live.recorder.state !== 'inactive') { live.recorder.stop(); }
  }

  function mono(buffer) {
    const out = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const data = buffer.getChannelData(channel);
      for (let i = 0; i < data.length; i += 1) { out[i] += data[i] / buffer.numberOfChannels; }
    }
    return out;
  }

  function encodeWav(samples) {
    const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
    const text = (offset, value) => { for (let i = 0; i < value.length; i += 1) { view.setUint8(offset + i, value.charCodeAt(i)); } };
    text(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    text(8, 'WAVE');
    text(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, RATE, true);
    view.setUint32(28, RATE * 2, true);    // byte rate
    view.setUint16(32, 2, true);           // block align
    view.setUint16(34, 16, true);          // bits per sample
    text(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i += 1) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    }
    return new Blob([view.buffer], { type: 'audio/wav' });
  }

  async function makeTake(blob, source, target) {
    // The decode wait is already the unsaved take, so every `live || take` guard blocks it.
    const pending = { ...target, source, wav: null, url: null };
    take = pending;
    if (source === 'uploaded') { placeRecorder(document.getElementById('voices'), recorderNode()); }
    openPanel(target, false);
    recorderPart('.recorder-status').textContent = 'Preparing the example…';
    let context;
    let samples;
    try {
      const bytes = await blob.arrayBuffer();
      if (take !== pending) { return; }
      context = new AudioContext({ sampleRate: RATE });
      samples = mono(await context.decodeAudioData(bytes));
    } catch (_) {
      // A take discarded while it decoded ends quietly: the page has moved on.
      if (take !== pending) { return; }
      take = null;
      closePanel();
      showError('This file could not be read as audio.');
      return;
    } finally {
      if (context) { context.close(); }
    }
    if (take !== pending) { return; }
    // A recording stops by itself at 30 s, a moment late; cut it to the limit.
    if (source === 'recorded') { samples = samples.subarray(0, MAX_SECONDS * RATE); }
    const seconds = samples.length / RATE;
    if (seconds < MIN_SECONDS || seconds > MAX_SECONDS) {
      take = null;
      closePanel();
      showError(`The example must be ${MIN_SECONDS}-${MAX_SECONDS} seconds; this one is ${seconds.toFixed(2)} s.`);
      return;
    }
    pending.wav = encodeWav(samples);
    pending.url = URL.createObjectURL(pending.wav);
    recorderPart('.recorder-status').textContent = `${seconds.toFixed(1)} s — listen, then save or discard it.`;
    recorderPart('.recorder-preview').src = pending.url;
    recorderPart('.recorder-again').hidden = source !== 'recorded';
    recorderPart('.recorder-take').hidden = false;
  }

  function discardTake() {
    if (!take) { return; }
    if (take.url) { URL.revokeObjectURL(take.url); }
    take = null;
    closePanel();
  }

  // A retake of the same text, in place. `prepared` is set after discardTake because
  // closing the panel clears it.
  function recordAgain() {
    if (!take) { return; }
    const { voiceId, language, promptId, kit } = take;
    discardTake();
    prepared = { voiceId, language, promptId, kit };
    startRecording();
  }

  async function saveTake() {
    if (!take || !take.wav) { return; }
    const body = new FormData();
    body.append('language', take.language);
    body.append('source', take.source);
    if (take.source === 'recorded') { body.append('prompt_id', take.promptId); }
    body.append('file', new File([take.wav], 'example.wav', { type: 'audio/wav' }));
    const saved = take;
    if (await send(`/voices/${saved.voiceId}/examples`, body)) {
      if (saved.source === 'recorded') { lastSaved[saved.language] = saved.promptId; }
      discardTake();
    }
  }

  // Stops the microphone and forgets every unsaved state; the recorder goes back home,
  // hidden, right after #voices.
  function resetRecorder() {
    if (live) {
      clearInterval(live.tick);
      live.discarded = true;
      if (live.recorder && live.recorder.state !== 'inactive') { live.recorder.stop(); }
      if (live.stream) { live.stream.getTracks().forEach((track) => track.stop()); }
      live = null;
    }
    if (take && take.url) { URL.revokeObjectURL(take.url); }
    take = null;
    closePanel();
    const recorder = recorderNode();
    const list = document.getElementById('voices');
    if (recorder && list && list.contains(recorder)) { list.after(recorder); }
  }

  // One test of the person, from the take films speak from (§11.1): a ready language,
  // then the text box: the user's own text, or the heard words of a chosen take.

  const jobBox = () => document.getElementById('voice-preview-job');
  const previewBusy = () => !!jobBox() && jobBox().dataset.state === 'running';

  // The Test tab's choice: { voiceId, language, kind, textId, words, ready }, or null when
  // the voice has no language to test. `ready` = #voices[data-test-ready].
  function testTarget() {
    const list = document.getElementById('voices');
    const language = list ? list.querySelector('.test-language') : null;
    const text = list ? list.querySelector('.test-text') : null;
    if (!language || !language.value || !text) { return null; }
    const option = text.options[text.selectedIndex];
    const value = option ? option.value : '';
    const split = value.indexOf(':');
    return {
      voiceId: pageNode().dataset.voiceId, language: language.value,
      kind: split < 0 ? value : value.slice(0, split), textId: split < 0 ? '' : value.slice(split + 1),
      words: option ? (option.dataset.words || '') : '', ready: list.dataset.testReady === '1',
    };
  }

  // Shows the takes of the chosen language only, the text box with the chosen take's words,
  // the stored comparison, and Generate only while there is a text the voice can speak.
  // A take of another language leaves with its words; the user's own text stays.
  function syncTest() {
    const panel = document.getElementById('voice-preview');
    const list = document.getElementById('voices');
    if (!panel || !list) { return; }
    const language = list.querySelector('.test-language');
    const text = list.querySelector('.test-text');
    const written = panel.querySelector('.preview-custom');
    if (language && text) {
      let recordings = 0;
      for (const option of text.options) {
        if (!option.dataset.lang) { continue; }
        const shown = option.dataset.lang === language.value;
        option.hidden = !shown;
        option.disabled = !shown;
        if (shown) { recordings += 1; }
      }
      const picked = text.options[text.selectedIndex];
      if (picked && picked.dataset.lang && picked.disabled) {
        text.value = 'custom';
        written.value = '';
      }
      text.closest('.test-label').hidden = !recordings;
    }
    const target = testTarget();
    const take = !!target && target.kind === 'take';
    if (take) { written.value = target.words; }
    const typed = written.value.trim().length;
    const limit = written.maxLength;
    written.hidden = !target;
    const count = panel.querySelector('.preview-count');
    count.hidden = !target;
    count.textContent = take
      ? `Words of take #${target.textId}. Edit them to speak your own text.` : `${typed} / ${limit}`;
    panel.querySelector('.preview-generate').disabled = !target || !target.ready
      || (!take && (!typed || typed > limit));
    const spoken = language && language.selectedIndex >= 0 ? language.options[language.selectedIndex].dataset.take : '';
    panel.querySelector('.preview-through').textContent = spoken ? `from take #${spoken} only` : '';
    const box = panel.querySelector('.preview-compare');
    const template = list.querySelector('template.test-compare');
    const html = template ? template.innerHTML : '';
    // Rebuilt only when it changed, so a list refresh does not stop a playing ▶.
    if (box.shownHtml === html) { return; }
    box.shownHtml = html;
    if (template) { box.replaceChildren(template.content.cloneNode(true)); } else { box.replaceChildren(); }
  }

  // An edit makes the words the user's own text, so no take is chosen any more.
  function editTest() {
    const text = document.querySelector('#voices .test-text');
    if (text) { text.value = 'custom'; }
    syncTest();
  }

  function selectTest(language) {
    const select = document.querySelector('#voices .test-language');
    if (select && [...select.options].some((option) => option.value === language)) {
      select.value = language;
    }
  }

  function fragmentOf(element) {
    const data = element.dataset;
    return {
      element, state: data.state, run: Number(data.run), epoch: data.epoch,
      completed: Number(data.completed), voiceId: data.voiceId, language: data.language,
    };
  }

  function parseJob(html) {
    const holder = document.createElement('div');
    holder.innerHTML = html.trim();
    const element = holder.querySelector('#voice-preview-job');
    return element ? fragmentOf(element) : null;
  }

  async function fetchStatus(own) {
    const response = await own.fetch(STATUS_URL, { cache: 'no-store' });
    if (!response.ok) { throw new Error(`status ${response.status}`); }
    const fragment = parseJob(await response.text());
    if (!fragment) { throw new Error('no job in the reply'); }
    return fragment;
  }

  function stopPreviewTimers(own) {
    if (previewTimer !== null) { own.cancel(previewTimer); previewTimer = null; }
  }

  // A running test of ANOTHER voice reads "Testing «<name>» · N% · Cancel".
  function markOtherVoice() {
    const box = jobBox();
    const page = pageNode();
    if (!box || !page || box.dataset.state !== 'running') { return; }
    const other = box.dataset.voiceId !== page.dataset.voiceId;
    for (const line of box.querySelectorAll('.preview-what, .preview-bar, .preview-step')) { line.hidden = other; }
    const note = box.querySelector('.preview-other');
    if (note) { note.hidden = !other; }
  }

  function applyJob(fragment) {
    const box = jobBox();
    if (!box) { return; }
    box.replaceWith(fragment.element);
    markOtherVoice();
  }

  function showRestarted() {
    const line = document.createElement('p');
    line.className = 'preview-what';
    line.textContent = 'The app restarted, so this test stopped. Generate it again.';
    const box = jobBox();
    if (box) { box.replaceChildren(line); }
  }

  function showPreviewState(s, html, generation) {
    const fragment = parseJob(html);
    if (fragment) { s.show(fragment, generation); }
  }

  function pollPreview() {
    if (previewTimer !== null) { return; }
    const own = mounted;
    const s = sync;
    const generation = s.generation();
    previewTimer = own.schedule(async () => {
      previewTimer = null;
      if (!previewBusy() || s.generation() !== generation) { return; }
      try {
        const response = await own.fetch(STATUS_URL, { cache: 'no-store' });
        if (response.ok) {
          const html = await response.text();
          if (own.alive()) { showPreviewState(s, html, generation); }
        }
      } catch (_) { /* the next tick tries again */ }
      if (own.alive() && previewBusy() && s.generation() === generation) { pollPreview(); }
    }, 1000);
  }

  // Generate, Cancel and Close (a failed test). A failed one keeps its error and reads
  // the state again; a mutation is never replayed.
  async function mutate(url, body) {
    const own = mounted;
    const s = sync;
    const generation = s.bump();
    clearError();
    let response;
    let html;
    try {
      response = await own.fetch(url, { method: 'POST', body });
      if (response.ok) { html = await response.text(); }
    } catch (_) {
      if (!own.alive() || s.generation() !== generation) { return; }
      showError('The server did not answer. Try again.');
      fetchStatus(own).then((fragment) => { if (own.alive()) { s.show(fragment, generation); } }, () => {});
      return;
    }
    if (!response.ok) {
      const detail = await detailOf(response);
      if (!own.alive() || s.generation() !== generation) { return; }
      showError(detail);
      fetchStatus(own).then((fragment) => { if (own.alive()) { s.show(fragment, generation); } }, () => {});
      return;
    }
    if (own.alive()) { showPreviewState(s, html, generation); }
  }

  function generatePreview() {
    const panel = document.getElementById('voice-preview');
    const target = testTarget();
    if (!target || !target.ready || panel.querySelector('.preview-generate').disabled) { return; }
    const body = new FormData();
    body.append('language', target.language);
    body.append('text_kind', target.kind);
    body.append('text_id', target.textId);
    body.append('text', panel.querySelector('.preview-custom').value);
    mutate(`/voices/${target.voiceId}/preview`, body);
  }

  // The job runs in `app`; #voice-hearing-job under the header is its line as this voice
  // sees it, polled from /voices/hearing?voice=<id> every 1 s while the takes of this voice
  // are heard or queued; its end refreshes the body. Every async site
  // captures `own` before its first await and checks `own.alive()` before any effect (t36).

  const hearingBox = () => document.getElementById('voice-hearing-job');
  const hearingOf = (element) => (element
    ? { state: element.dataset.state, run: element.dataset.run, voiceId: element.dataset.voiceId } : null);
  const hearingBusyHere = () => {
    const job = hearingOf(hearingBox());
    const page = pageNode();
    return !!job && !!page && (job.state === 'running' || job.state === 'queued')
      && job.voiceId === page.dataset.voiceId;
  };

  // The job line shows only on the page of its voice.
  function markHearing() {
    const box = hearingBox();
    const page = pageNode();
    if (box && page) { box.hidden = box.dataset.voiceId !== page.dataset.voiceId; }
  }

  // A fresh leaf body: the header, the takes and the Test tab follow the heard words.
  async function refreshBody(own) {
    const s = sync;
    const again = () => {
      if (bodyTimer !== null) { return; }
      bodyTimer = own.schedule(() => { bodyTimer = null; refreshBody(own); }, 1000);
    };
    if (typingInList()) { again(); return; }
    let html;
    try {
      const response = await own.fetch(`${leafUrl()}/list`, { cache: 'no-store' });
      if (!response.ok) { again(); return; }
      html = await response.text();
    } catch (_) {
      if (own.alive()) { again(); }
      return;
    }
    if (!own.alive()) { return; }
    if (typingInList() || replaceList(html, s) !== 'accept') { again(); }
  }

  // Puts a job fragment in place; a start, an end or a replaced run of this voice also
  // refreshes the body.
  function applyHearing(own, html) {
    const box = hearingBox();
    const page = pageNode();
    if (!box || !page) { return; }
    const holder = document.createElement('div');
    holder.innerHTML = html.trim();
    const next = holder.querySelector('#voice-hearing-job');
    if (!next) { return; }
    const before = hearingOf(box);
    box.replaceWith(next);
    markHearing();
    if (hearingChanged(before, hearingOf(next), page.dataset.voiceId)) { refreshBody(own); }
    if (hearingBusyHere()) { pollHearing(); }
  }

  async function readHearing(own) {
    try {
      const response = await own.fetch(`${HEARING_URL}?voice=${own.page.dataset.voiceId}`,
                                       { cache: 'no-store' });
      if (!response.ok) { return; }
      const html = await response.text();
      if (own.alive()) { applyHearing(own, html); }
    } catch (_) { /* the next tick tries again */ }
  }

  function pollHearing() {
    if (hearingTimer !== null) { return; }
    const own = mounted;
    hearingTimer = own.schedule(async () => {
      hearingTimer = null;
      if (!hearingBusyHere()) { return; }
      await readHearing(own);
      if (own.alive() && hearingBusyHere()) { pollHearing(); }
    }, 1000);
  }

  // At mount: the job as it is now — a return from Back or the htmx cache.
  function reconnectHearing(own) {
    markHearing();
    if (hearingBusyHere()) { pollHearing(); }
    readHearing(own);
  }

  // Hear the words (`start`), or Cancel / Close the job. A refusal shows its reason.
  async function postHearing(url) {
    const own = mounted;
    clearError();
    let response;
    let html;
    try {
      response = await own.fetch(url, { method: 'POST' });
      if (response.ok) { html = await response.text(); }
    } catch (_) {
      if (own.alive()) { showError('The server did not answer. Try again.'); }
      return;
    }
    if (!response.ok) {
      const detail = await detailOf(response);
      if (own.alive()) { showError(detail); }
      return;
    }
    if (own.alive()) { applyHearing(own, html); }
  }

  function close() {
    if (leaveNeedsConfirm(recorderState()) && !confirm(DISCARD)) { return; }
    const page = pageNode();
    const fromGrid = !!page && page.hasAttribute('data-from-grid');
    life.unmount();
    if (fromGrid) { history.back(); } else { location.replace('/voices'); }
  }

  function mountLeaf(page) {
    if (cameFromGrid(document.referrer, location.origin)) { page.setAttribute('data-from-grid', ''); }
    const own = mounted;
    const alive = own.alive;
    const s = previewSync({
      fetchList: () => own.fetch(`${leafUrl()}/list`, { cache: 'no-store' }),
      replaceList: (html) => (alive() ? replaceList(html, s) : 'reject'),
      typing: typingInList,
      schedule: (run, ms) => own.schedule(run, ms),
      apply: (fragment) => { if (alive()) { applyJob(fragment); } },
      poll: () => { if (alive()) { pollPreview(); } },
      stopTimers: () => { if (alive()) { stopPreviewTimers(own); } },
      fetchStatus: () => fetchStatus(own),
      restarted: () => { if (alive()) { showRestarted(); } },
      refreshed: () => {},
    }, fragmentOf(jobBox() || page), certOf(document.getElementById('voices')));
    sync = s;
    // A running or failed test of this voice selects its language.
    const job = jobBox();
    if (job && (job.dataset.state === 'running' || job.dataset.state === 'failed')
        && job.dataset.voiceId === page.dataset.voiceId) {
      selectTest(job.dataset.language);
    }
    showTab(tabFrom(location.search), false);
    decorate();
    markOtherVoice();
    pollWhileDeleting();
    reconnectHearing(own);
    // Reconnect the test from the server: a return from Back, Forward or the htmx cache
    // shows the test as it is now, and a test stored meanwhile refreshes the body.
    if (!job) { return; }
    const at = s.generation();
    fetchStatus(own).then((fragment) => { if (alive()) { s.show(fragment, at); } }, () => {});
  }

  function start(session, page) {
    mounted = session;
    pollTimer = null;
    cardsTimer = null;
    previewTimer = null;
    hearingTimer = null;
    bodyTimer = null;
    if (page.dataset.voiceId) { mountLeaf(page); } else { refreshCards(session); }
  }

  function stop() {
    resetRecorder();
    pollTimer = null;
    cardsTimer = null;
    previewTimer = null;
    hearingTimer = null;
    bodyTimer = null;
    sync = null;
  }

  if (window.__voicesWired) { return; }
  window.__voicesWired = true;

  const life = lifecycle({
    page: () => document.getElementById('voices-page'),
    setTimeout: (run, ms) => window.setTimeout(run, ms),
    clearTimeout: (id) => window.clearTimeout(id),
    fetch: (url, options) => window.fetch(url, options),
    start,
    stop,
  });

  document.addEventListener('submit', async (event) => {
    const form = event.target;
    if (!mounted || !pageNode() || !form.matches('.js-voices-form')) { return; }
    event.preventDefault();
    if (form.dataset.confirm && !confirm(form.dataset.confirm)) { return; }
    const ok = await send(form.getAttribute('action'), new FormData(form));
    if (ok && form.matches('.voices-new')) { form.reset(); }
    // A deleted voice has nothing left to show: the page closes to the grid.
    if (ok && form.matches('.voice-delete')) { close(); }
  });

  document.addEventListener('click', (event) => {
    if (!mounted || !pageNode()) { return; }
    const tab = event.target.closest('[role="tab"][data-tab]');
    if (tab) { showTab(tab.dataset.tab, true); syncTest(); return; }
    if (event.target.closest('.voices-close, .voices-up')) { event.preventDefault(); close(); return; }
    const hear = event.target.closest('.voice-hear');
    if (hear) {
      if (hear.dataset.voiceId) { postHearing(`/voices/${hear.dataset.voiceId}/hear`); }
      return;
    }
    if (event.target.closest('.voice-hearing-cancel, .voice-hearing-dismiss')) {
      postHearing(`/voices/hearing/cancel?voice=${pageNode().dataset.voiceId}`);
      return;
    }
    const record = event.target.closest('.voice-record, .kit-record');
    if (record) {
      const target = chosen(record);
      if (target) { prepare(target); }
      return;
    }
    const upload = event.target.closest('.voice-upload');
    if (upload && (live || take || !chosen(upload))) { event.preventDefault(); return; }
    if (event.target.closest('.recorder-next')) { differentText(); return; }
    if (event.target.closest('.recorder-start')) { startRecording(); return; }
    if (event.target.closest('.recorder-pause')) { togglePause(); return; }
    if (event.target.closest('.recorder-stop')) { stopRecording(false); return; }
    if (event.target.closest('.recorder-save')) { saveTake(); return; }
    if (event.target.closest('.recorder-again')) { recordAgain(); return; }
    if (event.target.closest('.recorder-discard')) { discardTake(); return; }
    if (event.target.closest('.preview-generate')) { generatePreview(); return; }
    if (event.target.closest('.preview-cancel, .preview-dismiss')) { mutate('/voices/preview/cancel'); }
  });

  document.addEventListener('input', (event) => {
    if (mounted && pageNode() && event.target.matches('.preview-custom')) { editTest(); }
  });

  document.addEventListener('change', (event) => {
    const input = event.target;
    if (!mounted || !pageNode()) { return; }
    if (input.matches('.test-language, .test-text')) { syncTest(); return; }
    if (!input.matches('.voice-upload') || !input.files.length) { return; }
    const file = input.files[0];
    input.value = '';
    if (live || take) { return; }
    const target = chosen(input);
    if (target) { clearError(); makeTake(file, 'uploaded', target); }
  });

  // A boosted nav click (not ⚙ Settings, an overlay) would throw a recording or an
  // unsaved take away: ask first (htmx never fires `beforeunload` for it).
  document.addEventListener('htmx:confirm', (event) => {
    const source = event.target;
    if (!source.closest || !source.closest('nav.nav') || source.closest('#settings-open')) { return; }
    if (!leaveNeedsConfirm(recorderState())) { return; }
    event.preventDefault();
    if (!confirm(DISCARD)) { return; }
    life.unmount();
    event.detail.issueRequest(true);
  });

  document.addEventListener('keydown', (event) => {
    // An open dialog (the settings overlay) takes its own Esc; the grid has no close.
    const page = pageNode();
    if (event.key !== 'Escape' || !mounted || !page || !page.dataset.voiceId
        || document.querySelector('dialog[open]')) { return; }
    // The page's own levels first: a live recording, then an unsaved take, then a
    // prepared recorder, then a focused text field — and only then up one level. Tabs
    // are not a level, and a running test runs on in `app`.
    if (live) { stopRecording(true); return; }
    if (take) { discardTake(); return; }
    if (prepared) { closePanel(); return; }
    const active = document.activeElement;
    if (active && active.matches('input[type="text"], textarea')) { active.blur(); return; }
    close();
  });

  document.addEventListener('htmx:historyRestore', () => life.mount());
  document.addEventListener('htmx:afterSettle', () => life.mount());
  document.addEventListener('htmx:beforeHistorySave', () => life.unmount());
  window.addEventListener('pageshow', (event) => life.pageshow(event));
  window.addEventListener('pagehide', () => life.unmount());
  life.mount();
})();
