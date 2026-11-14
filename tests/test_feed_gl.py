"""The film player's clock and its readiness gate (§11.1), the real feed-gl.js under node."""

import json
import pathlib
import shutil
import subprocess

import pytest

PLAYER = pathlib.Path(__file__).parent.parent / "web" / "static" / "feed-gl.js"

PLAY = "▶ Play"
LOADING = "Loading…"
BOTH_TRACKS = 2
FILM_SECONDS = 30
READY_TIMEOUT_MS = 20_000
BOTH_IMAGE_WAITS_MS = 2 * READY_TIMEOUT_MS

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

HARNESS = """
const fs = require('fs');
const vm = require('vm');

const FRAME_MS = 16;
const shot = (at, photo_id) => ({
  at, seconds: 10, photo_id, line: '', grade: 'neutral', transition: 'cut',
  transition_seconds: 0,
  move: { ease: 'linear', scale_from: 1, scale_to: 1.1, x_from: 0, x_to: 0, y_from: 0, y_to: 0 },
});
const TIMELINE = {
  seconds: FILM_SECONDS,
  shots: [shot(0, 1), shot(10, 2), shot(20, 3)],
  grades: { neutral: { lift: [0, 0, 0], gamma: [1, 1, 1], gain: [1, 1, 1] } },
};

class Track {
  constructor(seconds) {
    this.seconds = seconds; this.time = 0; this.paused = true; this.ended = false;
    this.plays = 0; this.readyState = 4;
  }
  get currentTime() { return this.time; }
  set currentTime(value) { this.time = Math.min(value, this.seconds); this.ended = false; }
  play() {
    if (this.ended) { this.time = 0; this.ended = false; }
    this.plays += 1; this.paused = false;
    return Promise.resolve();
  }
  pause() { this.paused = true; }
  addEventListener() {}
  removeEventListener() {}
  tick(seconds) {
    if (this.paused) return;
    this.time += seconds;
    if (this.time >= this.seconds) { this.time = this.seconds; this.paused = true; this.ended = true; }
  }
}

const classList = () => {
  const names = new Set();
  return {
    add: (...added) => added.forEach((name) => names.add(name)),
    remove: (...removed) => removed.forEach((name) => names.delete(name)),
    contains: (name) => names.has(name),
  };
};

const element = (children = {}) => ({
  textContent: '', offsetWidth: 1, listeners: {}, classList: classList(),
  addEventListener(name, listener) { this.listeners[name] = listener; },
  querySelector: (selector) => children[selector] ?? null,
});

async function boot({ held = null, images = 'load', film = 'ready' } = {}) {
  let now = 0, uTime = null, draws = 0, recordings = 0, nextTimer = 1;
  let imageMode = held === 'image' ? 'held' : images;
  let frames = [], timers = [], heldImages = [], keydown = null;
  const errors = [];

  const gl = new Proxy({}, { get(_, name) {
    if (name === 'getShaderParameter' || name === 'getProgramParameter') return () => true;
    if (name === 'getUniformLocation') return (_program, uniform) => uniform;
    if (name === 'getAttribLocation') return () => 0;
    if (name === 'drawArrays') return () => { draws += 1; };
    if (name === 'uniform1f') {
      return (location, value) => { if (location === 'uTime') uTime = value; };
    }
    return () => ({});
  } });

  const playLabel = element();
  playLabel.textContent = PLAY;
  const start = element({ '.feed-start-play': playLabel });
  const download = element({ '.feed-download-text': element() });
  const music = new Track(FILM_SECONDS);
  const narration = new Track(FILM_SECONDS - 4);
  const elements = {
    start, download, music, narration, 'feed-text': element(),
    stage: { getContext: () => gl, captureStream: () => ({ addTrack() {} }), width: 1, height: 1 },
  };

  class FakeImage {
    set src(_url) {
      const load = () => {
        this.naturalWidth = 1600; this.naturalHeight = 1200;
        if (this.onload) this.onload();
      };
      const fail = () => { if (this.onerror) this.onerror(); };
      if (imageMode === 'held') heldImages.push(load);
      else queueMicrotask(imageMode === 'error' ? fail : load);
    }
  }

  let releaseTimeline = () => {};
  const timeline = held === 'timeline'
    ? new Promise((resolve) => { releaseTimeline = resolve; }) : Promise.resolve();

  const sandbox = {
    document: {
      body: { dataset: { key: 'memory-0', lang: 'en' }, classList: classList() },
      getElementById: (id) => elements[id] ?? null,
      addEventListener(name, listener) { if (name === 'keydown') keydown = listener; },
    },
    Image: FakeImage,
    HTMLElement: class {},
    performance: { now: () => now },
    requestAnimationFrame: (frame) => { frames.push(frame); },
    setTimeout: (run, ms) => { timers.push({ id: nextTimer, at: now + ms, run }); return nextTimer++; },
    clearTimeout: (id) => { timers = timers.filter((timer) => timer.id !== id); },
    addEventListener() {},
    devicePixelRatio: 1, innerWidth: 1600, innerHeight: 900,
    fetch: async () => {
      await timeline;
      if (film === 'unreachable') throw new Error('no answer');
      const body = film === 'empty' ? { ...TIMELINE, shots: [] } : TIMELINE;
      return { json: async () => JSON.parse(JSON.stringify(body)) };
    },
    console: { error: (...parts) => errors.push(parts.join(' ')), log() {} },
    AudioContext: class {
      constructor() { this.currentTime = 0; this.destination = {}; }
      createGain() {
        return { gain: { value: 0, linearRampToValueAtTime() {} }, connect: (next) => next };
      }
      createMediaElementSource() { return { connect: (next) => next }; }
      createMediaStreamDestination() { return { stream: { getAudioTracks: () => [] } }; }
    },
    MediaRecorder: class {
      static isTypeSupported() { return true; }
      constructor() { this.state = 'inactive'; }
      start() { this.state = 'recording'; recordings += 1; }
      stop() { this.state = 'inactive'; if (this.onstop) this.onstop(); }
    },
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(PLAYER_PATH, 'utf8'), sandbox);
  sandbox.feedNotes = { all: () => [], open() {}, seek() {} };

  const settle = () => new Promise((resolve) => setImmediate(resolve));

  const wait = async (ms) => {
    const until = now + ms;
    for (;;) {
      const due = timers.filter((timer) => timer.at <= until).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      timers = timers.filter((timer) => timer !== due);
      now = due.at;
      due.run();
      await settle();
    }
    now = until;
    await settle();
  };

  const run = async (seconds) => {
    for (let i = 0; i < Math.round(seconds * 1000 / FRAME_MS); i++) {
      await wait(FRAME_MS);
      music.tick(FRAME_MS / 1000);
      narration.tick(FRAME_MS / 1000);
      const due = frames;
      frames = [];
      for (const frame of due) frame(now);
    }
  };

  // A browser reports a throw in an event listener and carries on, and so does this.
  const fire = async (listener, event) => {
    try { await listener(event); } catch (error) { errors.push(String(error)); }
  };
  const press = (listener, event) => { fire(listener, event); return settle(); };

  const h = {
    run, wait, errors,
    uTime: () => uTime,
    draws: () => draws,
    plays: () => music.plays + narration.plays,
    recordings: () => recordings,
    label: () => playLabel.textContent,
    click: () => press(start.listeners.click),
    download: () => press(download.listeners.click),
    space: () => press(keydown, { key: ' ', target: {}, preventDefault() {} }),
    arrow: () => press(keydown, { key: 'ArrowRight', target: {}, preventDefault() {} }),
    escape: () => sandbox.feedFilm.escape(),
    release: async () => {
      releaseTimeline();
      imageMode = 'load';
      for (const load of heldImages.splice(0)) load();
      await settle();
    },
    // The picture clock just after a Play, then two seconds later.
    sample: async () => {
      await run(0.1);
      const first = uTime;
      await run(2);
      return { first, later: uTime, plays: music.plays + narration.plays };
    },
  };
  await settle();
  return h;
}
"""


def _player(scenario: str) -> dict:
    """Run `scenario`, the body of an async function with `boot` in scope; it returns JSON."""
    constants = (f"const PLAYER_PATH = {json.dumps(str(PLAYER))};\n"
                 f"const FILM_SECONDS = {FILM_SECONDS};\nconst PLAY = {json.dumps(PLAY)};\n")
    script = (f"{constants}{HARNESS}\n(async () => {{\n{scenario}\n}})()"
              ".then((out) => console.log(JSON.stringify(out)))"
              ".catch((error) => { console.error(error); process.exit(1); });")
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60,
                          check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _assert_picture_runs_from(out: dict, seconds: float) -> None:
    assert out["first"] == pytest.approx(seconds + 0.1, abs=0.05)
    assert out["later"] == pytest.approx(seconds + 2.1, abs=0.05)


@pytest.mark.parametrize("idle", [5, 60])
def test_play_after_idling_on_the_title_card_starts_the_picture_at_zero(idle):
    out = _player(f"""
      const h = await boot();
      await h.run({idle});
      await h.click();
      return h.sample();
    """)
    _assert_picture_runs_from(out, 0)


def test_play_after_the_end_restarts_the_picture_at_zero():
    out = _player(f"""
      const h = await boot();
      await h.click();
      await h.run({FILM_SECONDS + 1});
      const ended_at = h.uTime();
      await h.click();
      return {{ ended_at, ...await h.sample() }};
    """)
    assert out["ended_at"] == pytest.approx(FILM_SECONDS)
    _assert_picture_runs_from(out, 0)


@pytest.mark.parametrize("pause", [8, 40])
def test_resume_continues_the_picture_from_the_paused_frame(pause):
    out = _player(f"""
      const h = await boot();
      await h.click();
      await h.run(10);
      await h.space();
      await h.run({pause});
      await h.space();
      return h.sample();
    """)
    _assert_picture_runs_from(out, 10)


def test_play_after_esc_and_an_idle_starts_the_picture_at_zero():
    out = _player("""
      const h = await boot();
      await h.click();
      await h.run(10);
      h.escape();
      await h.run(50);
      await h.click();
      return h.sample();
    """)
    _assert_picture_runs_from(out, 0)


@pytest.mark.parametrize("held", ["timeline", "image"])
@pytest.mark.parametrize("press", ["click", "space", "download"])
def test_a_press_before_ready_starts_nothing_then_or_when_ready_arrives(press, held):
    out = _player(f"""
      const h = await boot({{ held: '{held}' }});
      await h.{press}();
      const plays_while_held = h.plays();
      await h.release();
      await h.run(1);
      return {{
        plays_while_held, plays_after_ready: h.plays(), recordings: h.recordings(),
        errors: h.errors,
      }};
    """)
    assert out == {"plays_while_held": 0, "plays_after_ready": 0, "recordings": 0, "errors": []}


@pytest.mark.parametrize("held", ["timeline", "image"])
@pytest.mark.parametrize("press", ["click", "download"])
def test_a_click_before_ready_says_loading_until_ready(press, held):
    out = _player(f"""
      const h = await boot({{ held: '{held}' }});
      await h.{press}();
      const waiting = h.label();
      await h.release();
      return {{ waiting, ready: h.label() }};
    """)
    assert out == {"waiting": LOADING, "ready": PLAY}


def test_an_arrow_key_before_ready_does_not_move_the_film():
    out = _player("""
      const h = await boot({ held: 'image' });
      await h.arrow();
      await h.release();
      await h.click();
      return h.sample();
    """)
    _assert_picture_runs_from(out, 0)


def test_play_is_refused_just_before_the_deadline_of_an_image_that_never_finishes():
    out = _player(f"""
      const h = await boot({{ held: 'image' }});
      await h.wait({BOTH_IMAGE_WAITS_MS - 1});
      await h.click();
      return {{ plays: h.plays(), label: h.label() }};
    """)
    assert out == {"plays": 0, "label": LOADING}


def test_play_runs_at_the_deadline_of_an_image_that_never_finishes():
    out = _player(f"""
      const h = await boot({{ held: 'image' }});
      await h.wait({BOTH_IMAGE_WAITS_MS});
      await h.click();
      return h.sample();
    """)
    assert out["plays"] == BOTH_TRACKS
    _assert_picture_runs_from(out, 0)


def test_a_deadline_passing_starts_no_audio():
    out = _player(f"""
      const h = await boot({{ held: 'image' }});
      await h.wait({BOTH_IMAGE_WAITS_MS});
      await h.run(1);
      return {{ plays: h.plays() }};
    """)
    assert out == {"plays": 0}


def test_play_runs_when_the_first_image_fails_to_load():
    out = _player("""
      const h = await boot({ images: 'error' });
      await h.click();
      return h.sample();
    """)
    assert out["plays"] == BOTH_TRACKS
    _assert_picture_runs_from(out, 0)


def test_the_first_image_arriving_starts_no_audio():
    out = _player("""
      const h = await boot({ held: 'image' });
      await h.release();
      await h.run(1);
      return { plays: h.plays(), draws: h.draws() };
    """)
    assert out["plays"] == 0
    assert out["draws"] > 0


@pytest.mark.parametrize("film", ["unreachable", "empty"])
def test_a_film_that_cannot_load_keeps_play_on_its_own_label(film):
    out = _player(f"""
      const h = await boot({{ film: '{film}' }});
      await h.click();
      return {{ label: h.label(), plays: h.plays() }};
    """)
    assert out == {"label": PLAY, "plays": 0}
