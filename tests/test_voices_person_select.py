"""The New voice person select on a returned Voices grid (design §13, §13.1; D7, D8).

`refreshPeople` in `web/static/voices.js` runs under node against the real `/voices` page
of the app, with a fake DOM that reads the select's options from that HTML.
"""

import json
import pathlib
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from tests.factories import add_person
from tests.test_web_voices import _js_function
from web.app import create_app

NEW_PERSON = ""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

HARNESS = r"""
const flush = () => new Promise((resolve) => setImmediate(resolve));
const { SHOWN, NOW, CHOSEN, REPLY, ALIVE } = /* DATA */;
function optionsIn(html) {
  const select = html.match(/<select id="voice-new-person"[^>]*>([\s\S]*?)<\/select>/);
  if (!select) { return null; }
  return [...select[1].matchAll(/<option value="([^"]*)"[^>]*>([^<]*)<\/option>/g)]
    .map(([, value, text]) => ({ value, text: text.trim() }));
}
// A <select>: setting a value no option has, or replacing the options under a value,
// falls back to the first option, as a browser does.
function select(options, value) {
  const s = { children: options, current: value };
  Object.defineProperty(s, 'options', { get: () => s.children });
  Object.defineProperty(s, 'value', {
    get: () => s.current,
    set: (v) => { s.current = s.children.some((o) => o.value === v) ? v : s.children[0].value; },
  });
  s.replaceChildren = (...next) => { s.children = next; s.value = s.current; };
  return s;
}
const page = select(optionsIn(SHOWN), CHOSEN);
const document = {
  getElementById: (id) => (id === 'voice-new-person' ? page : null),
  createElement: () => {
    const holder = { html: '' };
    Object.defineProperty(holder, 'innerHTML', { set: (v) => { holder.html = v; } });
    holder.querySelector = (q) => {
      const options = q === '#voice-new-person' ? optionsIn(holder.html) : null;
      return options && { children: options };
    };
    return holder;
  },
};
const asked = [];
const own = {
  alive: () => ALIVE,
  fetch: async (url, options) => {
    asked.push({ url, cache: options && options.cache });
    if (REPLY === 'down') { throw new Error('down'); }
    return { ok: REPLY === 'ok', status: REPLY === 'ok' ? 200 : 500, text: async () => NOW };
  },
};
/* REFRESH */
(async () => {
  await refreshPeople(own);
  await flush();
  console.log(JSON.stringify({ asked, value: page.value,
                               options: page.options.map((o) => o.text) }));
})();
"""


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _option_value(body: str, name: str) -> str:
    return re.search(rf'<option value="([^"]*)">{name}</option>', body).group(1)


def _refresh(shown: str, now: str, chosen: str, *, reply: str = "ok", alive: bool = True) -> dict:
    source = pathlib.Path("web/static/voices.js").read_text()
    refresh = f"{_js_function(source, 'async function refreshPeople')}\n}}"
    data = json.dumps({"SHOWN": shown, "NOW": now, "CHOSEN": chosen, "REPLY": reply,
                       "ALIVE": alive})
    script = HARNESS.replace("/* REFRESH */", refresh).replace("/* DATA */", data)
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30,
                          check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_a_returned_grid_offers_the_people_as_they_are_now_and_keeps_the_choice(client):
    add_person(_conn(client), name="Ann")
    add_person(_conn(client), name="Bob")
    shown = client.get("/voices").text
    ann, bob = _option_value(shown, "Ann"), _option_value(shown, "Bob")
    assert client.post("/voices", data={"name": "Nana"}).status_code == 200
    assert client.post(f"/people/{ann}/delete").status_code in (200, 204)

    out = _refresh(shown, client.get("/voices").text, bob)

    assert out["asked"] == [{"url": "/voices", "cache": "no-store"}]
    assert out["options"] == ["New person with this name", "Bob", "Nana"]
    assert out["value"] == bob


@needs_node
def test_a_chosen_person_deleted_meanwhile_is_not_kept(client):
    add_person(_conn(client), name="Ann")
    shown = client.get("/voices").text
    ann = _option_value(shown, "Ann")
    assert client.post(f"/people/{ann}/delete").status_code in (200, 204)

    out = _refresh(shown, client.get("/voices").text, ann)

    assert out["options"] == ["New person with this name"]
    assert out["value"] == NEW_PERSON


@needs_node
@pytest.mark.parametrize(("reply", "alive"), [("down", True), ("error", True), ("ok", False)],
                         ids=["offline", "server error", "grid left before the reply"])
def test_the_select_stays_as_it_is_when_no_fresh_list_arrives(client, reply, alive):
    add_person(_conn(client), name="Ann")
    shown = client.get("/voices").text
    ann = _option_value(shown, "Ann")
    add_person(_conn(client), name="Bob")

    out = _refresh(shown, client.get("/voices").text, ann, reply=reply, alive=alive)

    assert out["options"] == ["New person with this name", "Ann"]
    assert out["value"] == ann


def test_only_the_grid_reads_the_people_again_when_it_mounts():
    source = pathlib.Path("web/static/voices.js").read_text()
    wiring = source[source.index("const life = lifecycle({"):]
    mount = wiring[wiring.index("start(session, page) {"):wiring.index("stop,")]
    assert "start(session, page);" in mount
    assert "if (!page.dataset.voiceId) { refreshPeople(session); }" in mount
