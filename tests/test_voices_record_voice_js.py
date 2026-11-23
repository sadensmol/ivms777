"""A Voices grid opened by the person page's "Record a voice" (design §13 `/voices`, §13.1).

`refreshPeople` and the submit handler of `web/static/voices.js` run under node against the
real `/voices?person=<id>` page, with a fake form whose `reset()` returns each field to its
default, as a browser does. After the mount refresh and a Create, the form must still have
the linked person selected and an empty name: Back shows that form from the bfcache.
"""

import json
import pathlib
import subprocess

import pytest
from fastapi.testclient import TestClient

from tests.factories import add_person
from tests.test_web_voices import _js_function, needs_node
from web.app import create_app

NEW_PERSON = ""

HARNESS = r"""
const flush = () => new Promise((resolve) => setImmediate(resolve));
const { SHOWN, NOW, SENT, CREATE } = /* DATA */;
function optionsIn(html) {
  const select = html.match(/<select id="voice-new-person"[^>]*>([\s\S]*?)<\/select>/);
  if (!select) { return null; }
  return [...select[1].matchAll(/<option value="([^"]*)"([^>]*)>([^<]*)<\/option>/g)]
    .map(([, value, attrs, text]) => ({ value, text: text.trim(),
                                        defaultSelected: /\bselected\b/.test(attrs) }));
}
function nameIn(html) {
  const input = html.match(/<input[^>]*id="voice-new-name"[^>]*>/)[0];
  const value = input.match(/\bvalue="([^"]*)"/);
  return value ? value[1] : '';
}
// A <select>: it shows its default option, a value no option has falls back to the first
// option, and reset() returns to the default option.
function select(options) {
  const s = { children: options, current: '' };
  Object.defineProperty(s, 'options', { get: () => s.children });
  Object.defineProperty(s, 'value', {
    get: () => s.current,
    set: (v) => { s.current = s.children.some((o) => o.value === v) ? v : s.children[0].value; },
  });
  s.replaceChildren = (...next) => { s.children = next; s.value = s.current; };
  s.reset = () => {
    const chosen = s.children.find((o) => o.defaultSelected) || s.children[0];
    s.current = chosen.value;
  };
  s.reset();
  return s;
}
const page = select(optionsIn(SHOWN));
const name = { defaultValue: nameIn(SHOWN), value: nameIn(SHOWN) };
const form = {
  dataset: {},
  matches: (q) => q === '.js-voices-form' || q === '.voices-new',
  getAttribute: () => '/voices',
  elements: { namedItem: (n) => (n === 'name' ? name : null) },
  reset: () => { name.value = name.defaultValue; page.reset(); },
};
let onSubmit = null;
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
  addEventListener: (type, handler) => { if (type === 'submit') { onSubmit = handler; } },
};
const own = {
  alive: () => true,
  fetch: async () => ({ ok: true, status: 200, text: async () => NOW }),
};
const mounted = true;
const pageNode = () => true;
const confirm = () => true;
const close = () => {};
class FormData {}
const send = async () => SENT;
/* REFRESH */
/* SUBMIT */
(async () => {
  await refreshPeople(own);
  await flush();
  if (CREATE) {
    await onSubmit({ target: form, preventDefault: () => {} });
    await flush();
  }
  console.log(JSON.stringify({ person: page.value, name: name.value }));
})();
"""


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _conn(client):
    return client.app.state.context.conn


def _run(shown: str, now: str, *, sent: bool = True, create: bool = True) -> dict:
    source = pathlib.Path("web/static/voices.js").read_text()
    refresh = f"{_js_function(source, 'async function refreshPeople')}\n}}"
    submit = source[source.index("document.addEventListener('submit'") :]
    submit = submit[: submit.index("\n  });\n") + len("\n  });")]
    data = json.dumps({"SHOWN": shown, "NOW": now, "SENT": sent, "CREATE": create})
    script = (
        HARNESS.replace("/* REFRESH */", refresh)
        .replace("/* SUBMIT */", submit)
        .replace("/* DATA */", data)
    )
    done = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_a_create_from_the_linked_grid_keeps_the_person_and_empties_the_name(client):
    add_person(_conn(client), name="Bob")
    ann = add_person(_conn(client), name="Ann")
    shown = client.get("/voices", params={"person": ann}).text
    assert client.post("/voices", data={"name": "Ann", "person_id": ann}).status_code == 200

    out = _run(shown, client.get("/voices").text)

    assert out == {"person": str(ann), "name": ""}


@needs_node
def test_the_mount_refresh_keeps_the_linked_person_and_the_name(client):
    ann = add_person(_conn(client), name="Ann")
    shown = client.get("/voices", params={"person": ann}).text

    out = _run(shown, client.get("/voices").text, create=False)

    assert out == {"person": str(ann), "name": "Ann"}


@needs_node
def test_a_refused_create_leaves_the_form_as_typed(client):
    ann = add_person(_conn(client), name="Ann")
    shown = client.get("/voices", params={"person": ann}).text

    out = _run(shown, client.get("/voices").text, sent=False)

    assert out == {"person": str(ann), "name": "Ann"}


@needs_node
def test_a_create_on_the_plain_grid_resets_to_a_new_person(client):
    add_person(_conn(client), name="Ann")
    shown = client.get("/voices").text

    out = _run(shown, client.get("/voices").text)

    assert out == {"person": NEW_PERSON, "name": ""}


@needs_node
def test_a_linked_person_deleted_meanwhile_resets_to_a_new_person(client):
    ann = add_person(_conn(client), name="Ann")
    shown = client.get("/voices", params={"person": ann}).text
    assert client.post(f"/people/{ann}/delete").status_code in (200, 204)

    out = _run(shown, client.get("/voices").text)

    assert out == {"person": NEW_PERSON, "name": ""}
