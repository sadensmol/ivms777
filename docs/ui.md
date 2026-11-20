# UI — the app shell and per-route detail

The persistent shell, the resource bar, and what every route renders. Design §13
carries the shell concept, the nav order, and (authoritatively) the §13.1 navigation
model; this file is the per-route detail. Section references point into
`docs/design.md`; the navigation contract every drill-down obeys is design §13.1.

## App shell & resource bar

- **Persistent app shell** — the nav is a **full-width fixed header**; the content
  region **below it is the only scroll container** (the window never scrolls), so the
  page scrollbar starts under the menu and never crosses it. The six nav links
  (Upload/Library/Chat/Organize/People/Voices) are **HTMX-boosted** and swap **only `<main>`**, so
  the nav and its resource-bar polling stay resident across top-level navigation —
  the menu never re-renders and the resource bar never blanks. Boost is **scoped to
  these links only**: every grid↔leaf drill-down (photo, similar, prev/next) is a
  full navigation, so §13.1's bfcache-based `history.back()` is unaffected.
- **Resource bar** — live status at the **right of the nav row** on **every** page.
  It polls `GET /api/resources` (~2 s) and always renders the same shape:

  `RAM 7.1/7.4 GB · CPU 5% 51°C · GPU 99% 58°C · captioning · gemma4-E2B +vision`

- **The machine metrics are read by `app` itself, never proxied**
  (`models/resources.py::snapshot`). RAM and CPU come from `psutil`; GPU load from
  the kernel counter `/sys/devices/platform/gpu.0/load` on jetson (per mille — the
  same number `tegrastats` prints as `GR3D_FREQ`), `nvidia-smi` on cloud, `ioreg`
  (`AGXAccelerator` → `Device Utilization %`, no sudo) on mac; temperatures from the
  thermal zones `/sys/devices/virtual/thermal/thermal_zone*` (millidegrees — the
  same source as `tegrastats`' `cpu@…C`/`gpu@…C`). **None of this needs a CUDA
  context, a driver library or `runtime: nvidia`**, so reading it in `app` does not
  violate the one-model-process rule (§5.1) — verified on the board, the thin `app`
  container reads both paths with no extra mount. Sysfs rather than `tegrastats`
  because no container ships that binary. The consequence that matters: the bar
  **keeps showing RAM/CPU/GPU/temperature while the `models` service is down,
  restarting, or still loading** — which is exactly when you want to look at it.
- **CPU and GPU are ALWAYS shown**, load and temperature together so each number
  has an owner and the bar never changes shape as the box goes idle. An idle GPU
  reads `GPU 0%`, not a missing field. A genuinely absent sensor renders `—`
  rather than vanishing, because a field that disappears looks like a bug. Only
  `cpu-thermal` and `gpu-thermal` are shown (`tj-thermal`, the junction temperature
  throttling is keyed to, rides in the payload); the soc/cv zones are not — two
  numbers the user can act on, not nine. A zone that reads empty (the Orin's cv
  zones do) is **skipped, never reported as 0 °C**. Temperatures earn their place
  because sustained captioning holds the GPU at ~99% and is the hottest thing the
  board does (§3.1).
- **Only the model info comes over HTTP** — which models are resident and the op in
  flight, from the `models` service's own `GET /resources` (§5.1), the two things
  it alone knows. When it is unreachable those degrade to "no model info" and every
  other field is unaffected.

## Settings popup (⚙) — model slots

A **⚙ button sits in the nav, immediately after the resource bar**, on every page.
It opens a modal `<dialog>` whose body is fetched from `GET /settings/models` — an
overlay, not a route: no URL change, no history entry, so design §13.1 is untouched
and Esc/backdrop/× just closes it.

- **One section per slot** (design §4.1) — Image embeddings, Caption text
  embeddings, Captions, Planner & chat — each naming what the slot does in one line
  and listing its catalog entries as radios, with the **currently active one
  checked**. A slot the profile pins (all four on `cloud`) renders read-only with
  the reason.
- **Each entry shows** display name, download size, resident RAM cost — suffixed
  **"estimated"** when `cost_measured` is false — and its state: **on disk**, a
  **live progress bar** (`bytes / total`, polled ~1 s while any download is in
  flight, otherwise not polled at all), or **Download** for one not yet fetched. A
  failed download shows the error and offers a retry; nothing else in the dialog is
  blocked by it.
- **Selecting a different entry does not switch anything.** It reveals the
  consequence line — "Switching re-runs *embed, taxonomy* for **4,812 photos**"
  (the count comes from `jobs`), plus "search results will be incomplete until it
  finishes" for the two embedder slots — and enables a single **Switch** button.
  `planner` has no consequence line: it stores nothing.
- **Switch** is disabled while the model is not on disk. Pressing it POSTs, and the
  dialog re-renders with the new selection active; the requeued work then shows up
  in the ordinary per-stage progress rows on `/upload` (§8) and in the resource bar,
  not in the dialog — the dialog is not a job monitor.

## Routes

- `/upload` — leads with the **folder list**: every folder in the library (§3.2c)
  with its photo count and a confirm-guarded **Delete from library** button (a folder
  mid-deletion shows "deleting…"). Below it, a **directory-only** picker adds a new
  folder; watch client-side hashing, the transfer, then live processing progress per
  stage with counts, **per-stage throughput** (`done/sec`, measured from recent
  `jobs.updated_at` so it is derived — not stored — and the last speed **survives a
  restart**), and failed files (Web Worker for hashing/upload, HTMX polling for
  processing). Every stage row carries its own **Reprocess** button that re-runs just
  that stage over the already-uploaded library without re-uploading — `thumbnail`,
  `embed`, `taxonomy` (re-tag), `caption embed` (the §9 caption vector, shown with a
  space), and a confirm-guarded `caption` (re-caption, the slow
  one); the worker drains the reset jobs.
- `/organize/plan?by=…&grain=…` — reached by **Organize files…** on `/organize`. It
  previews the folder tree the current principle would create — every album folder
  with its photo count, plus `_unsorted/` and the theme-memory folders — asks
  "reorganize your folders this way?", and on confirm downloads `ivms777-plan.json`.
  It shows whether the collection is fully processed and the exact `ivms777-sync`
  command line to run next. Cancel returns to `/organize` with its state intact
  (design §13.1).
- `/library` — infinite-scroll thumbnail grid. Hover shows caption and top tags; an
  `×N` badge marks photos with exact duplicates. Left sidebar has two filter groups
  with counts: model-derived tags per dimension, and EXIF facets (camera, lens, ISO
  and aperture ranges, year, time of day, orientation). A sort control offers capture
  date or any numeric facet. Filters and sort **apply on change** — no Apply button —
  swapping only the grid via HTMX so the sidebar and its scroll position stay put; a
  single **Clear all filters** button at the top resets them. Top bar has the search
  box and parsed-filter chips.
- `/photo/{id}` — a full-screen view of one photo, always shown **inside the
  collection it was opened from** (the library with its filters/search/sort, an
  Organize album, or a memory). `‹`/`›` buttons and the ← / → arrow keys page to the
  previous/next photo **within that collection**, in its order (owner-scoped; the
  arrows are absent at the ends) — never leaking into another album or the wider
  library. The collection travels in a `ctx` URL parameter, which the route resolves
  to the ordered id list. The panel leads with the **collection's identity** — its
  title, description, and `N / M` position — shown on every photo of it, and only then
  the photo's own data. Closing returns to the collection's top-level grid (the
  album/memory/filtered library) with state and scroll intact via the browser's
  history; every in-photo nav (paging, similar) uses replace, so close always lands on
  the grid, never on a prior photo. The per-photo panel carries everything known about
  it: an **AI-written title and description**, the caption, tags grouped by dimension
  with scores and source badges (the "AI data"), the full EXIF panel — including GPS
  **coordinates**, which live here as a technical detail and nowhere else — every
  local path the file arrived from with the wasted-space total when there is more than
  one, a **People** line, and a "similar photos" strip. The People line names the
  people recognised in the photo (its faces whose group is attached to a person, design
  §6; the person's name in `people`, never a voice's), each name once, owner-scoped, as **plain text** — a link to a person would open a
  leaf from a leaf (design §13.1). With no named person the line is absent. The photo itself and everything above render on
  first paint; the similar strip is the one expensive part (a full-library scan), so
  it loads **asynchronously** — the page ships a placeholder that fires
  `GET /photo/{id}/similar` on load and swaps in the finished strip (§9.2) — so opening
  a photo is never delayed by it. When the photo was opened within an album or memory,
  a **collage of that whole collection** — every photo in it, the current one
  highlighted — sits between the photo and the similar strip; each thumbnail opens in
  place (`replace`, §13.1) so paging stays within the collection, and the collage uses
  the **same tile size as the similar strip** so the two read as one gallery. The
  similar strip then **excludes any photo already in that collection**, so a member
  never appears twice. Each similar thumbnail is **enlarged and labelled with why it
  matched** — its top-3 reasons (shared tags / caption meaning / "looks alike" / "same
  time & place") with confidence percentages, one per line, **sorted by what actually
  drove the match**, not by the biggest percentage, overlaid on the image — so
  similarity is never a black box (§9). When the strip comes back short, a
  **"Show more — looser matches"** button appends up to 6 more, found at lower gates
  and rendered **dimmed under a "Looser matches" heading** so they never read as
  equals of the strip above; they are appended, never merged, and never repeat a
  result already shown. Opening a similar photo opens in
  a **"Similar to <this photo>"** layer (`ctx=similar:<id>`) that shows the base
  photo's thumbnail (clickable, to jump back to it) and pages within this photo's
  similar set. **A photo is always exactly one level below a grid** (§13.1): every
  photo→photo move — prev/next, opening a similar, the origin thumbnail — **replaces**
  history, so it stays `[grid, photo]` and **close always goes up to the grid** (the
  library for `library`/`q`/`similar:*`, the album for `album:*`), never replaying the
  chain of photos visited. The layer panel leads with the two photos' **own words —
  `Base` then `This`**, each with that photo's AI title, description, and caption, in
  the same order as the comparison table's columns, so *what* is being compared is read
  before *how* it scored (in this layer the leaf's own title/description/caption block
  is not repeated below the table). Then the **"Why similar — base vs this"** table:
  every facet that ACTUALLY SCORED (tag, caption meaning, visual, moment), both photos'
  values side by side and their match %, **sorted by how much each drove the match** —
  so a weak match is visibly weak rather than a mystery. Facets below their gate get no
  row at all: they contributed nothing, so the panel must not claim them. Sorting by
  match % instead is what used to headline two unrelated photos with `light: low light
  69%` — a big percentage is not a big reason. The panel also
  offers **per-photo reprocess** — *Re-tag* and a confirm-guarded *Re-caption* — that
  re-run just this photo's model stages (§8); thumbnails and embeddings are static and
  are not offered. The AI title/description and tags fill in with the caption and
  planner phases; until then the panel shows EXIF, sources, and embedding status. This
  is where duplicate paths are seen, since there is no separate duplicates screen.
- `/organize` — a dropdown of organization principles (date, memories, camera, place)
  over a list of album cards, each with a cover, title, description, and a strip of
  its photos. `date` shows a grain sub-selector (day / month / year, default month).
  Live principles recompute on selection; `memories` reads stored rows and offers a
  "Rebuild memories" control that queues the background build, showing a live
  `done/total (%)` indicator (HTMX polling) while it runs and reloading to the finished
  albums when it completes. The **last-opened organizer and grain are remembered** (a
  per-owner cookie), so loading `/organize` from the nav returns to the view you last
  used — memories, place, or a specific date grain — rather than snapping back to the
  default date view ("never lose the user's place").
  - A **memory** card lists **one film row per language** (English, then Russian),
    each a plain link to `/feed/{key}?lang=<code>` (a leaf, never boosted) followed by
    its state. The first matching state wins:

    | State | Link | State text |
    |---|---|---|
    | being made | Open <Language> film | being made… |
    | waiting in the queue | Open <Language> film | waiting to be made… |
    | ready | ▶ Play <Language> film | ready · <narrator> (your choice) or ready · <narrator> (auto); ready · voice not recorded |
    | no film | Prepare <Language> film | not made yet |
    | the last build failed | Prepare <Language> film | the last try failed |
    | a build stopped mid-way | Prepare <Language> film | interrupted |
    | ready, but no narration | Prepare <Language> film | no voice |

    The narrator name comes from the film itself (design §11.1), never from the current
    choice.
  - **↻ Retell memory** (↻ Retelling… while it runs) sits under the rows, with its
    text below it: "Rewrites the title, description and photo order. Makes no film.
    Deletes the films in: English, Russian." (only the languages that have a film), or
    "… No film to delete." when there is none. When the memory has an edited narration
    (design §11.3), the text adds "Deletes your edited narration in: <Language>, …."
    (only the languages that have one). The narrator choices stay.
  - The film rows and the retell control **refresh in place**, never with a page
    reload: while a film of that memory is being made or waits, the card's block polls
    `/organize` every 1.5 s and swaps only itself, and stops when the work ends; on a
    return from `/feed` restored from bfcache, the page fetches `/organize` once and
    swaps each card's film block. The grid keeps its scroll, its controls and its
    expanded notes.
- `/people` — **People** (design §6, §13): the owner's people. It is a **top-level
  grid** (design §13.1), the fifth nav link, with no × and no `ctx`, on every profile
  (a person needs no voice copy). The head holds **New person**: a name (1-80
  characters, unique per owner, case-insensitive) and **Create**; an empty, too long or
  taken name is refused with its reason and nothing is written.
  - Each person is a **card**, a plain `<a href="/people/{id}">` (a full navigation,
    never boosted — the one grid→leaf push): the name, the names of their voices, and
    "N faces". With no person, the grid reads "No people yet."
  - **Faces to name** follows the cards: every face group of the owner with no person
    and at least the minimum number of faces (2 by default, env-overridable in
    `config.py`; one stranger's face never floods the list), plus every such group that
    holds a face the owner released (Not this person) whatever its size, the largest
    first. Each group
    shows up to four face crops (`<img src="/faces/crop/{face_id}">`) and "N faces", and
    is a plain `<a href="/faces/{id}">` (a full navigation, never boosted — the one
    grid→leaf push). With no such group the section is absent.
  - The routes (`web/people_api.py`): `GET /people`, `POST /people` (form `name`;
    answers the cards), `GET /people/{id}` (the person leaf),
    `POST /people/{id}/rename` (form `name`) and `POST /people/{id}/delete` (409 with its
    reason while the person owns a voice). A name error answers 400 (empty or too long)
    or 409 (taken) with its reason. A foreign or unknown id is a 404 on every
    `/people/{id}…` route.
- `/people/{id}` — one person, the **leaf** of the People grid (design §13.1).
  - **Header:** "People › <name>" — "People" is a link that closes the page — a **⋯**
    menu with **Rename…** (the New person rules) and **Delete person…**, and a **×**.
  - Then, all in place, with no link to another leaf (design §13.1): the person's
    **Voices** by name, as plain text ("No voices yet." with none); their **Faces**:
    each face group attached to them with up to four crops, "N faces" and **Remove from
    this person** (the group returns to "Faces to name"), "No faces yet." with none;
    and the thumbnails of the **Photos** with their faces, newest first, as plain images.
  - **Delete person** is refused with its reason while the person owns a voice, one
    being deleted included. Otherwise the person goes, their face
    groups return to "Faces to name", no photo, face or voice is deleted, and the page
    closes to the grid as below.
  - Every action is a `fetch` POST answered in place (`people.js`); a refused action
    shows its reason in the page's error line. None pushes history.
  - **Close** (×, the "People" link, Esc) goes up to the grid: `history.back()` when the
    page carries `data-from-grid` (set at mount when `document.referrer` has the same
    origin and the path `/people`), else `location.replace("/people")`. The page holds
    no level of its own, so Esc closes at once.
- `/voices` — **Voices** (design §6, §11.1, §13): the voices that can narrate a film,
  each owned by a person. It is a **top-level grid** (design §13.1), the sixth nav link,
  with no × and no `ctx`. The head reads "A voice is a person's own voice. It can narrate a
  film in each language it has a take of." and holds **New voice**: a name (1-80
  characters, unique per owner, case-insensitive), its **person** — a select of the
  owner's people, by default "New person with this name" — and **Create**. With the
  default, a person whose name is the voice's name already exists: the create is
  refused with its reason and asks the owner to pick that person; nothing is written.
  - Each voice is a **card**, a plain `<a class="voice-card" href="/voices/{id}">` (a
    full navigation, never boosted): the name, its person's name as plain text, one line per language with takes,
    "<Language> · m:ss · N takes", and one line of the languages it is ready for
    (design §11.1): "For films: English, Russian" or "Not ready for films yet". A voice
    with no take reads "No takes yet". A card plays nothing.
    No card names a take for films. With no voice, the grid reads "No voices yet."
  - A voice being deleted is a card with "deleting…" and no link. While a card shows
    "deleting…", the grid polls `GET /voices/list` every 2 s (design §6).
  - **Create** answers the cards. The new card carries `data-created`, and the page
    opens it at once with `location.assign` (the one grid→leaf push), on Recordings.
  - When the catalog lists no voice-copy model for this device, the head reads "Voice
    copy is not available on this device." and has no New voice. The cards still
    open, for rename and delete.
  - Faces are not on this grid: "Faces to name" lives on `/people`.
  - The old page moved: `GET /organize/voices` answers **308** with `Location:
    /voices`, and every other path under `/organize/voices/` is gone. The routes, with
    the status codes, the disk guard and the drain wake-up they had:
    `GET /voices`, `GET /voices/list`, `POST /voices` (form `name` and `person_id`,
    empty = a new person with the voice's name),
    `GET /voices/preview` (the test state, `Cache-Control: no-store`),
    `POST /voices/preview/cancel`, `GET /voices/hearing?voice=<id>` (the hearing job as
    that voice sees it, `Cache-Control: no-store`; polled every 1 s while the takes of the
    open voice are heard or queued), `POST /voices/hearing/cancel?voice=<id>` (Cancel of
    that voice's work, or **Close** of its failure),
    `GET /voices/{id}` (`?tab=`), `GET /voices/{id}/list` (the leaf body),
    `POST /voices/{id}/rename`, `POST /voices/{id}/delete`,
    `POST /voices/{id}/person` (form `person_id`: move the voice to another person of the
    owner; another owner's person is refused),
    `POST /voices/{id}/examples` (add a take; form `language`, `source` and
    `prompt_id`, the kit text shown; no words),
    `POST /voices/{id}/examples/{example_id}/delete`,
    `GET /voices/{id}/examples/{example_id}/audio`,
    `POST /voices/{id}/hear` (Hear the words; 409 "This voice is being deleted." or "Hearing the words of a take is not offered on this device yet."), `POST /voices/{id}/preview` (the test; form `language`, `text_kind`,
    `text_id`, `text`) and `GET /voices/{id}/preview/audio` (the stored test WAV). The
    literal paths come before `/voices/{id}`. A foreign or unknown id is a 404 on every
    `/voices/{id}/…` route. The per-take routes of the earlier page —
    `POST /voices/{id}/film-example`, `POST /voices/{id}/examples/{example_id}/preview`
    and `GET /voices/{id}/examples/{example_id}/preview/audio` — are gone, and no route
    takes the words of a take (`…/examples/{example_id}/transcript` does not exist):
    they answer 404 or 405, as every unknown path does. The bake routes were removed:
    `POST /voices/{id}/bake` and `POST /voices/bake/cancel` answer 404, and
    `GET /voices/bake` answers 422 (it reads "bake" as a voice id).
  - The face routes (`web/faces_api.py`): `GET /faces/{id}` (the group leaf),
    `POST /faces/{id}/assign` (form `person_id`), `POST /faces/{id}/person` (form
    `name`: create a person for the group), `POST /faces/{id}/unassign`,
    `POST /faces/face/{face_id}/release` (Not this person) and
    `GET /faces/crop/{face_id}` (a JPEG). The literal paths come before
    `/faces/{id}`. A foreign or unknown id is a 404 on every route.
- `/voices/{id}` — one voice, the **leaf** of the Voices grid (design §13.1).
  - **Header:** "Voices › <name>" — "Voices" is a link that closes the page (below) —
    a **⋯** menu with **Rename…** (the rules of New voice) and **Delete voice…**, and a
    **×** that closes the page. Under it, the voice's person as plain text, "Voice of
    <person>" (no link: a person page is another leaf, design §13.1), with **Move to
    another person**: a select of the owner's other people and **Move**, a POST answered
    in place that pushes no history. Then, on every tab, the state of the voice (design
    §11.1), one line per language with takes or that voice copy speaks here:
    - ready: "<Language> · films and tests speak from take #<id>", plus "· take #<n>
      waits for its words" for a newer take not heard yet, or "· take #<n> has no usable
      words" for a newer take whose heard words are unusable;
    - "<Language> · no take yet";
    - "<Language> · the words of its takes are not heard yet" (no take of the language
      heard yet);
    - "<Language> · no take with usable words".
    Then **Hear the words** when a take waits for its words and no hearing job runs for
    this voice, on a device that offers speech recognition; with none, "Hearing the
    words of a take is not offered on this device yet." Then the hearing job line of
    this voice: "<stage> · N%" with a bar and **Cancel**, or, while another voice is
    heard first, "waiting to hear the words, after another voice" with **Cancel**; its first
    state is "starting · 0%", its stages "waiting for the voice model" (a film or a
    test holds it) and "hearing the words · take n of m". A failed job shows its
    reason and **Close**: "Could not hear the words of take #<id> (<reason>).", "The
    audio of take #<id> could not be read (<reason>).", "Hearing the words of a take is
    not offered on this device yet.", "The models service is baking a voice for the
    bake check. Hear the words again when it ends." or "Hearing the words failed
    (<reason>)." A failure belongs to its voice: the job goes on with the other voices.
    Cancel stops the work of this voice after the take in hand; the words heard so far
    stay, and the other voices are heard as before. The job runs in `app`, so a return to
    the page shows it again. The page polls while its voice is heard or queued, and when
    that work ends it refreshes the body (the state lines and the Test tab).
  - **Tabs:** **Recordings** (the default) and **Test**. The URL carries
    `?tab=recordings|test`; an unknown value, `model` and `faces` too, shows Recordings.
    There is no Faces tab: face groups belong to the person (`/people/{id}`). A switch
    replaces the URL (`history.replaceState`), so Back never walks through tabs and a
    reload keeps the tab. A switch keeps the recorder, the typed test text and every
    selection.
  - **Recordings:** for every language that voice copy speaks on this device, also
    with no takes, a line "<Language> · N of 26 kit texts · m:ss recorded" — with no
    takes "<Language> · 0 of 26 kit texts · 0:00 recorded" — then the
    **recording kit** of that language (design §11.1):
    one group per style, headed by its label and instruction — "Normal — Read in your
    normal voice.", "Slow — Read slowly and clearly, with short pauses.", "Fast — Read
    quickly, as if you are in a hurry, but keep every word clear.", "Calm — Read calmly
    and softly.", "Happy — Read with joy, as good news.", "Sad — Read sadly and
    quietly.", "Serious — Read seriously, as an important announcement.", "Excited —
    Read with excitement.", "Questions — Read every question as a real question.",
    "Emphasis — Read with energy and stress the words that matter.", "Articulation —
    Read clearly and take your time with each sound.", "Numbers and names — Read the
    numbers and names as they are written." Each kit text shows its id and its first
    words, then its takes, or **Record** when it has none. A kit text is only a
    suggestion: a take on its row may say anything. Takes with no kit text (uploads,
    and recordings made with no kit text) follow under **Other takes**. Takes in other
    languages stay listed, with no Record controls. One row per take: "#<id> · ▶ m:ss ·
    recorded|uploaded · date · Record again (a kit text only) · Delete". No row shows,
    asks for or confirms words: the app hears them (design §11.1). A take not heard yet
    adds "· words not heard yet". A take whose heard words are unusable adds "· not
    used: <reason>", where the reason is "the speech recognition found no words in it"
    or "it holds more than 1000 characters of words"; it stays, and no film speaks from
    it. ▶ plays the take. **Delete** asks nothing (design §11.1). There is no
    Test button on a take. Delete and ▶ of a take that is unknown (▶ also: being
    deleted, or its file is gone) answer 404 "Unknown example.".
  - **Add a take** follows the takes: a language select (only the languages voice copy
    speaks on this device) and two sources, **Record** and **Upload a file**. Before them sits one tip: "Record clean, normal speech in a quiet room.
    Films and tests of «<name>» speak from its newest take of the language, once its
    words are heard." Record opens the recorder right under
    its row (a kit text's Record or Record again opens it under that text; Add a take's
    Record opens it under "Add a take" with one kit text of the chosen language, the one
    after the last text saved in that language on this page, or the first text when
    none was saved, and **Different text**, the next kit text, wrapping to the first),
    and it stays there across a list refresh: the style's instruction, the text and
    **Start recording**; no microphone is used yet. A kit row's Record or Record again
    uses that row's language and text, whatever language "Add a take" has selected.
    "Different text" shows only for "Add a take". Start recording first pauses every `<audio>` on
    the page (echo cancellation is off, so a playing take or test would be recorded),
    then shows the timer, **Pause** and **Stop**. Pause holds the recording: the button
    becomes **Resume**, the timer stops, and the status reads "Paused — the pause is not
    recorded." Resume pauses every `<audio>` on the page again and records on into the
    same take; any number of pauses still make one take. Pause is disabled while the
    page waits for the microphone (the status reads "Waiting for the microphone…") and
    after Stop. When the browser gives no microphone, the recorder closes with "The
    microphone is not available. Allow it, or upload a file instead." The recording stops by itself at 30 s
    of recorded time; paused time does not count. Under the text the
    recorder reads "Read it, or say anything else: the app hears what you say." The
    text stays fixed through the recording and the unsaved take, and Save sends its id,
    so the take stores which kit text it showed. Record needs a secure context: elsewhere it is hidden
    with "Recording needs HTTPS or localhost. Upload a file instead."
  - Both sources end the same way: the browser decodes the audio, mixes it to mono,
    encodes a PCM-16 WAV at 48 kHz and shows a preview with **Save** and **Discard**.
    A recording also shows **Record again**: it drops the unsaved take and starts a new
    recording of the same text at once, in the same place, so a retake needs no Discard
    and no Record. While it decodes, the status reads "Preparing the example…"; the preview
    reads "<s> s — listen, then save or discard it." Audio the browser cannot decode
    ends with "This file could not be read as audio."; a length outside 5-30 s with
    "The example must be 5-30 seconds; this one is <s> s." (a recording is first cut
    to 30 s). Save posts the take as it is (`language`, `source` and, for a
    recording, `prompt_id`); it asks for no words. The server accepts only that format, 5-30 s, ≤ 4 MiB, not silent, and names the format
    it found when it refuses a file. A file over 4 MiB answers 413 "The example is
    larger than 4 MiB."; the same audio twice in one voice answers 400 "This voice
    already has this example.", or 409 "This example is being deleted. Try again in a
    moment." while that take is being deleted. An error shows its reason above the
    takes and the page stays. A request with no answer shows "The server did not
    answer. Try again."; a refusal with no reason shows "The server refused this
    (<status>)." Every action is a `fetch` that replaces the leaf body; none pushes
    history. A saved take starts the hearing job of its voice (above), and the reply
    already shows it.
  - **Test:** one test of the person, from the take films speak from (design §11.1).
    **Language** lists the languages the voice is ready for that voice copy speaks
    here ("the newest take of the language with heard words speaks"). **Recording** ("Choose a recording
    to compare with") lists, for that language, "Take #<id> — <words>" for every take
    whose words were heard, newest first, with the words cut at 80 characters. It
    holds nothing else, and the row is hidden when the language has no such take. A
    take not heard yet, or one with no usable words, is not listed. The text box under
    it is always there ("Type what «<name>» should say"), at most 300 characters, kept
    for the page session, with "N / 300" under it. Writing in it needs no choice in
    the list. Picking a recording fills the box with all its heard words, and the line
    reads "Words of take #<id>. Edit them to speak your own text."; any edit makes them
    your own text and clears the Recording choice. Changing the language keeps your
    own text and clears a take's words. **Generate** ("from take #<id> only", the take
    of the chosen language) is disabled while the box is empty or your own text is over the limit. It
    starts the test and replaces a running one. While it runs, the job line reads
    "Testing «<name>» (<Language>): “<text>”" with a bar, "<stage> · N%" and
    **Cancel**; it first reads "starting · 0%", then the stages "waiting for the voice
    model" (a film, the hearing job or another
    test holds it), "reading the take", "generating the voice", "measuring the
    voice · take N of M" and "saving the test". A test of another voice reads "Testing «<name>» · N% · Cancel". Cancel stops
    the test for this page. A voice generation that has already started finishes in
    the background, and the next test waits for it. Under the job line, the
    **comparison** shows the stored test: "Generated — <Language> · “<text>” · from
    take #<id> · <model> @ <first 7 of revision> · made in <s> s · <date>" with ▶ ("·
    from a deleted take" once that take is gone; "· voice model v<N>" for a test made
    through a voice model before), plus "· films of «<name>» now speak from take #<m>
    — generate it again" when the take of that language changed since; and, when the test spoke the words of a take, "Original —
    take #<id>: “<words>”" with ▶, or "Original — the take was deleted". Under them, the
    measure of the test (design §11.1): "Speaker similarity 0.31 (cosine) · your own
    takes 0.72-0.85", never a percent or a probability, with no threshold and
    no pass mark. With under 2 takes in the language it reads "Speaker similarity 0.31
    (cosine)". When the takes of that language changed since the test, it adds
    "· measured against older takes". A failed measure reads "Speaker similarity not
    measured: <reason>", and a test made before the meter reads "Speaker similarity not
    measured"; neither is a zero, and neither fails the test. With no ready language
    the tab reads "Record a take first. Tests speak from the newest take of the
    language, once its words are heard."; Generate is disabled. A finished test refreshes the
    comparison once; a failed one shows its reason and **Close**, which clears it
    (`POST /voices/preview/cancel`). The reasons: "Record a <Language> take first.";
    "The words of the <Language> takes are not heard yet."; "No <Language> take has
    usable words."; "This voice is being deleted."; "Take #<id> of «<name>» could not
    be read (<reason>)."; "The models service did not speak «<name>» from its take.";
    "The models service could not make the test (<reason>)."; "The take of the test
    was deleted while the test ran, or the voice is being deleted. Generate it
    again."; "The test failed (<reason>)." (any other error). When
    the app restarted during a test, the job line reads "The app restarted, so this
    test stopped. Generate it again."; the typed text and the stored test stay. A test
    runs in `app`, so a return to the page shows it again. A test never changes a take
    or the narrator films use.
  - **Refusals of a test** show their reason above the takes (the page stays; a running
    test keeps running). Generate answers 404 "Unknown saved voice."; 409 "This voice
    is being deleted.", "Voice copy is not available on this device." (also when voice
    copy does not speak the language), "Record a <Language> take first.", "The words of
    the <Language> takes are not heard yet." or "No <Language> take has usable words."
    — at once, so Generate never starts a test that must fail; 507
    "only <N> MB free on the server" (below 512 MiB free); and 400 for the text:
    "Unknown test sentence for <Language>: '<id>'.", "Unknown take.", "The words of
    take #<id> are not heard yet.", "Take #<id> has no words to speak:
    <reason>.", "Type a text to test.", "A test text can have
    at most 300 characters." or "Unknown kind of test text.". The audio of ▶ Generated
    answers 404 "Unknown saved voice." or "This voice has no test yet.".
  - **Delete voice** asks first: "Delete «<name>» and its examples? Films already
    narrated with this voice keep their sound until you regenerate them." After the
    reply the page closes to the grid, where the card reads "deleting…". A delete of a
    voice or of one take only records the intent (design §6): a take being deleted stays
    in the list as "deleting…", with no ▶, words or Delete, and the page polls the leaf
    body every 2 s until the drain has removed it. A voice being deleted takes no edit,
    no new take and no new film, and its words are no longer heard.
  - **Close** (×, the "Voices" link, Esc) goes up to the grid: `history.back()` when
    the page was opened from the grid (it marks itself `data-from-grid` at mount when
    `document.referrer` has the same origin and the path `/voices`), else
    `location.replace("/voices")`. A deep link therefore closes to `/voices`, and Back
    does not return to the voice.
  - **Esc** unwinds the page's own levels first: a live recording, paused or not, is
    stopped and discarded → an unsaved take is discarded → a prepared recorder is closed → a
    focused text field (the test textarea too) is blurred → and only then the page
    closes. Tabs are not a level, and the test is not one either: it is a tab, and a
    running test runs on in `app`.
  - **Leaving with a recording or an unsaved take asks first.** A click on a nav link
    (not ⚙) while a recording runs or a take is unsaved asks "Discard the recording you
    have not saved?"; yes discards it and goes, no keeps the page and the recording.
    Close asks the same. ⚙ Settings is an overlay and asks nothing. On leaving, the
    page stops the microphone and its timers; on a return (Back, Forward, or the htmx
    history restore) it reconnects the deleting poll and the test from the server.
- `/faces/{id}` — one face group, a **leaf** of the People grid (design §13.1).
  - **Header:** "People › Faces to name", or "People › <person>" when the group is
    attached, and **×**, then "N faces". Then every face of the group as a crop (cut
    from the photo's 1600 px detail thumbnail by the face box plus a margin of 20% of
    its side, on request; no file is stored), newest photo first, each with **Not this
    person** when the group holds more than one face.
  - **Attach to a person:** a select of the owner's people (from `people`, by name)
    and **Attach**. **New person:** a name (the New person rules: 1-80 characters,
    unique per owner, case-insensitive) and **Create**; a taken or empty name is
    refused with its reason and nothing is written. Creating a person here creates no
    saved voice. Both work on every profile: a person needs no voice copy. An attached group leaves "Faces to name", and every
    face that later joins it is that person with no action by the owner. An attached
    group shows **Remove from this person** instead of Attach and New person; the group
    returns to "Faces to name".
  - **Not this person** moves that one face out to a new group of its own and pins it,
    so grouping never moves it again. The new group is listed under "Faces to name",
    so the released face can be named. A one-face group has no such button: the move
    would only pin the same face.
  - Every action is a `fetch` POST that answers the group leaf; `faces.js` swaps it in
    place. A refused action shows its reason in the page's error line. None pushes
    history: the URL changes only through `history.replaceState`.
  - **Close** (×, the "People" link, Esc) goes up to the grid: `history.back()` when
    the page carries `data-from-grid` (set at mount when `document.referrer` has the
    same origin and the path `/people`), else `location.replace("/people")`. The page
    holds no level of its own, so Esc closes at once.
- `/feed/{key}` — the **memory film**: one memory played fullscreen as a directed
  short (design §11.1). Its controls sit in **one header** at the top: an **action
  row** — the language selector, ↓ Download (ready film only) and the × close — that
  wraps on a narrow window, and below it, on a ready film, a **settings region** with
  the ↻ Regenerate form, which scrolls inside its own bounded height. No control has a
  fixed offset from another, each fades on its own (an exporting ↓ Download stays
  fully visible), and the header's empty parts pass clicks through to the film. The
  language selector reads **Film language** (English / Русский), with a **✓** on each
  language whose film is ready; each language is prepared and cached separately, so
  switching offers its own Prepare rather than discarding the other.
  A ready film's settings region starts with **"This film: <Language> · narrated by
  <label> — <who chose it>"**: "your choice", "picked automatically by the story",
  "picked automatically using the memory's mood", or "picked automatically" when the
  film did not record why; a film with no recorded voice reads "This film: <Language>
  · Voice not recorded". A film narrated by a saved voice adds the take it spoke from,
  "· spoken from take #<id>" ("narrated by me (saved voice) — your choice · spoken
  from take #12"; a saved voice that Automatic picked: "narrated by Grandma (saved
  voice) — picked automatically by the story · spoken from take #7", and its card
  reads "Grandma (auto)"); a film made through a voice model keeps "· voice model
  v<N>", and a saved-voice film made from one take before voice models adds "· made
  from one take, before voice models"; a preset film adds nothing. When other passages use other voices, one **"Other voices:"**
  line follows, e.g. "passage 3 → am_adam (male, neutral); passage 5 → bm_george
  (male, deep)". Both lines come from the film itself, never from the current choice.
  It is the one route that does **not** use the app shell — a fullscreen player has
  no room for the nav, and boosting it would break §13.1's history invariant — so it
  renders its own black page with its own stylesheet.
  - **Not prepared yet:** the memory's title and description over a **Prepare this
    memory** button, with a one-line note that generating the soundtrack and
    narration runs once, on this machine, and is kept until the memory is rebuilt.
    While the background job runs, an HTMX poll shows "Composing music and
    narration…" and reloads the page the moment it finishes.
  - **Tell the story better:** the notes panel (design §11.3) also lists the film's
    narration, one row per passage, so the user edits the words instead of writing the
    story again. Its data is `<script id="narration-data" type="application/json">`
    beside `#notes` (never `id="narration"`: that is the audio element), served again by
    `GET /feed/{key}/narration?lang=`.
    - **Row look:** a passage row is the stamp button, the tag **narration**, the
      auto-grown field and ×; a note row has no tag. One list, keyed by passage `n` and
      note id, sorted by time (a passage's start, a note's `at`; on a tie the passage
      first); passages with no time come first, in telling order, then the notes. The
      folded bar's count is all rows: passages plus notes. Passages never float as
      bubbles. The panel keeps its hide, open and fold rules.
    - **Scrolling:** `.feed-notes-body` is the panel's one scrolling region, with a visible
      thin scrollbar, so every row and the Add form can be reached however many rows there
      are. The title bar does not scroll. The body stays inside the panel's height limit,
      also the lower limit of a narrow window beside the settings.
    - **Times:** with a player and the times of this language's ready film, a passage
      stamp `[m:ss]` seeks the film to where that passage is spoken. A draft row keeps
      the time of the film it came from, also after its text changed. A draft of an
      earlier film shows the stamp disabled with the title "From the film these words
      were taken from"; on the Prepare screen it is disabled with the title "Where this
      lands once the film is made"; a passage with no time shows "–:––", disabled.
    - **State line**, above the list, one of:
      - the film's own rows: "The film's narration is in this list with your notes.
        Change its words, and ↻ Regenerate speaks them as written. With no change,
        ↻ Regenerate writes a new story."
      - an edited narration that differs from the film: "Your edited narration.
        ↻ Regenerate speaks it as written." ↻ Regenerate is lit from the page load on.
      - an edited narration that the film speaks: "This film speaks your edited
        narration. ↻ Regenerate speaks it again as written." ↻ Regenerate is not lit
        at load.
      - under either edited line: "Notes cannot rewrite these words or provide a new
        voice or music interpretation. Existing voice choices still pass current note
        gating and validation. Effect instructions reach the conductor; matched speech
        fixes shot timing." and the button **Write a new story**.
      - an older film: "This older film has no timed narration text. ↻ Regenerate makes
        a new film whose narration you can edit here."
      - no film: no line.
    - **With an edited narration**, the Regenerate and Prepare buttons get the title
      "Speak your edited narration, and score this memory again"; the picker's
      Automatic bracket line reads "With your edited narration, a [voice] note cannot
      ask for a new voice: it keeps the voice the story already gave the passage it
      points at."; the legend's first lead reads "With your edited narration, a note's
      words are not narrated and cannot change its words, voices or music. Effect
      brackets still reach the conductor; matched speech fixes shot timing." The
      chosen-narrator line has no twin. The page switches to these texts on load and
      after every reply, so the first change shows them without a reload.
    - **Saving:** a passage saves on blur and on Enter; shift+Enter is a newline, which
      the server joins to a space. × removes the passage. An unchanged text posts
      nothing. Saves run one at a time, in order, each with the `source`, `draft` and
      `revision` of the latest reply; every save that changed a row lights
      ↻ Regenerate. A re-render never replaces a row in focus or one with unsaved text:
      it keeps its text, its focus and its caret.
    - **Refusals** (the line under the state line; the text stays in its row): "The
      narration is read aloud word for word. Put a [bracket] instruction in a note.",
      "The narration is longer than this app allows: <N> characters, at most 10000.",
      "A film needs narration. Keep at least one passage.", "This passage is no longer
      in the narration.", and on a network failure "Not saved: the app did not answer.
      Your text stays in its row." Every save queued behind it is dropped; its row stays
      unsaved, and its next blur posts it again. A × that has no 200 of its own (refused,
      failed, or dropped behind a failure) leaves its row and counts as unsaved. Click ×
      again, or **Load the current narration**, which the error line then shows. The
      fields stay editable.
    - **409, another window or a new film:** "Not saved: this narration changed in
      another window, or a new film replaced it. Copy any text you want to keep, then
      load the current narration." with the button **Load the current narration**. The
      queue is dropped, no passage saves any more, and every passage field becomes
      read-only with its text. The button reloads every row from the server; then the
      panel saves again.
    - **Save barrier:** Prepare and ↻ Regenerate first wait for the panel's saves. When a
      passage row is unsaved or a 409 froze the panel, nothing is posted and the form's
      refusal line reads "Not started: an edit to the narration is not saved. See Tell
      the story better."
    - **Write a new story** waits the same way, then asks "Stop using your edited
      narration? The next Prepare or Regenerate writes a new story." and drops the
      edited narration: the list shows the ready film's own rows, or none after a failed
      build.
    - **Ask AI to rephrase:** a block under the list, above the legend (design §11.3).
      Its heading is "Ask AI to rephrase", then the transcript, the proposal, the progress
      line and the form: the field "Ask for a change…" and **Send**. Enter sends, and
      shift+Enter is a newline. The block is hidden in the older-film and no-film states, and
      its field and buttons are disabled while a 409 freezes the panel.
      - **Send** first waits for the panel's saves. With a row unsaved, it reads "Not sent:
        an edit to the narration is not saved." and the text stays in the field.
      - **While it runs:** the stage, then "· N% (estimate)", and **Cancel**. Esc cancels
        first; the next Esc goes on as before. The rows stay editable, and a saved edit makes
        the answer outdated. A turn keeps the form in view only while the field has the
        focus, so an answer never moves a viewer who is reading the rows.
      - **Transcript:** "you:" and "AI:" lines. The conversation stays in the page session
        for this memory and language. A failure or a refusal is the AI's line of that turn,
        for example "Cancelled.", "This did not finish. Ask again." or "Not used: passage 3
        has a [bracket]." Model text is shown as plain text, never as HTML.
      - **Proposal:** "Changed N of M passages.", then one row per changed passage: its
        stamp, the old words, →, the new words. A follow-up works on this proposal, and the
        rows list the whole patch against the saved words. **Use this** and **Dismiss** close
        it. A reply with no change shows no proposal.
      - **Use this** waits for the panel's saves, and every passage field and × is read-only
        until the reply. With a row unsaved, it reads "Not used: an edit to the narration is
        not saved." On a 200 the rows show the new words, the proposal closes and
        ↻ Regenerate lights. A refusal shows on the error line, gives the fields back and
        keeps the proposal. A 409 is the freeze above, and the proposal stays.
      - **Outdated:** a proposal for rows that changed reads "These words were written for
        an older narration. Ask again." It offers Dismiss and no **Use this**.
  - **Narrator:** the Prepare form and the ↻ Regenerate form each carry the select
    **"Narrator for the next film"**, always shown, on every device. Its options:
    "Automatic — the app picks a voice for this memory"; a **Preset voices**
    group, one option per catalog speaker of the film's language ("<name> — <gender>,
    <register>: <note>"); and a **Saved voices** group with every saved voice not being
    deleted. A voice that is ready for the film's language (design §11.1) is a plain
    option; every other one is a **disabled** option with its reason: "«<name>» — no
    <Language> take", "«<name>» — hearing its words" (its takes of the language wait
    for their words and the hearing job runs or queues this voice), "«<name>» — words
    not heard yet (open it under Voices)" (they wait and no hearing job runs) or "«<name>» — no <Language> take with usable words". The Saved voices group shows only when it has an
    option, or with the disabled line "Voice copy not available on this device" when
    voice copy does not speak the language here. The stored choice for this memory and
    language is preselected; after an Automatic film "Automatic" stays selected while
    "This film" names the voice it picked. A stored choice that cannot narrate now
    stays **selected** as one more option in its group — never shown as Automatic —
    labelled "«<name>» — not offered for <Language> films" (a preset), "A deleted saved
    voice", "«<name>» — being deleted", "«<name>» — voice copy is not available on this
    device", or the saved-voice labels above. When the selected saved voice is not
    ready for the language, a line under the select links to it: "«<name>» needs a
    <Language> take with heard words for <Language> films: open «<name>» in Voices",
    where the link goes to `/voices/<id>` (a plain link; the voice page's close then
    goes to `/voices`). Submitting a choice that cannot narrate answers 409 (a preset
    of another language: "«<name>» cannot narrate in <Language>. Choose another
    narrator."; a saved voice: "«<name>» has no <Language> take yet. Record one under
    Voices, or choose another narrator.", "The words of «<name>»'s <Language> takes are
    not heard yet. Open «<name>» under Voices and let it hear them, or choose another
    narrator.", "No <Language> take of «<name>» has usable words. Record a take where
    you speak, or choose another narrator." or "The saved voice «<name>» is being
    deleted. Choose another narrator.") or 404 (an
    unknown preset), and the reason shows under the form; the page stays. A film whose
    saved narrator cannot narrate fails with the same text. A saved-voice film also
    fails with "The saved voice chosen for this film was deleted. Choose another
    narrator.", "Take #<id> of «<name>» could not be read (<reason>).", "The models
    service did not speak «<name>» from its take." or, when a
    saved voice it used was deleted before its save, "The saved voice «<name>» was
    deleted while this film was made." Under the select, in this
    order:
    - always, when voice copy speaks the language here: "The story picks the voice: a
      preset voice or one of your saved voices ready for <Language>.
      Otherwise the memory's mood picks a preset voice."; else: "The story picks a
      preset voice; otherwise the memory's mood guides the choice."
    - when voice copy speaks the language here: "Add a voice under Voices.", where
      "Voices" is a link to `/voices`;
    - the bracket line, by the stored choice: a preset or a saved narrator → "The
      chosen narrator reads the whole film."; Automatic → "A [voice] note can request
      another voice for one passage: another preset when a preset narrates, another
      saved voice when a saved voice narrates." Only an Automatic film lets a `[voice]`
      note recast a passage (design §11.1).
  - **Prepared:** the first shot behind a large title card with a **▶ Play** button.
    Browsers refuse to start audio without a user gesture, so nothing plays until
    that click — this is deliberate, not a loading state.
  - **Playing:** a single **WebGL2 canvas**. The camera pushes, pulls and drifts
    through each photo on an eased curve; shots dissolve, a burst cuts hard, and a
    change of day or city dips to black. The whole film carries one colour grade
    picked from its `palette`/`light` tags, plus a vignette and fine grain. Each
    shot's written line fades up from the bottom. The generated track loops underneath
    and **ducks automatically while the narration speaks** (WebAudio gain ramp, not a
    volume jump).
  - **Controls:** **space** play/pause, **←** / **→** step a shot back or forward,
    **Esc** or the **×** close. Stepping never touches history — §13.1 rule 3 — so
    close is always exactly one step back to `/organize?by=memories`, with the grid's
    scroll restored by bfcache. `prefers-reduced-motion` drops only the CAPTION's
    slide-in — never the camera. A film is the feature, not decoration around one, and
    it plays only after a deliberate Play press; gating the camera on a global OS
    setting turned every film into a static slideshow.
- `/chat` — a normal chat view: a running **conversation history** of questions and
  their grounded answers, a text **input** at the bottom, a **processing indicator**
  while the model works, streamed answer tokens, and inline thumbnail citations.
  A `thought for … · HH:MM` line sits **above** each answer, in the order it
  happened: it is the **wait before the answer** — request in to first token out,
  covering routing, retrieval and any model swap — and deliberately **excludes
  streaming time**, since once tokens are arriving the user is reading, not waiting,
  and a long answer is not a slow one. The server announces it on its own `thinking`
  SSE event as the first token goes out, so it prints as the answer starts rather
  than after it ends, and it is persisted with the turn so a reload shows the same
  number. The log **follows the stream to the bottom** and scrolls once more when
  the turn finishes.
  History is **persisted** (`chat_sessions`/`chat_messages`, §6) and re-rendered
  server-side on load, so it survives navigation and restarts; a **New session** button
  starts a fresh conversation. Each question is grounded independently — retrieval runs
  per question, and the persisted history is the transcript, not multi-turn model
  memory. A cited thumbnail opens the photo as a **leaf of the chat grid** (`ctx=chat`,
  §13.1), so closing it returns to the conversation, not the library. A **"show me a
  memory"** answer additionally renders that memory as the **same Organize memory
  card** below the reply; opening a photo from it pages **within the memory**
  (`ctx=chat-memory:<key>`) and closes back to the conversation (§10, §13.1).

## Nav order

The nav order is **Upload → Library → Chat → Organize → People → Voices**: bring photos in,
browse and search them, ask about them to understand the collection, then — once you
know what you have — group and reorganize them. Organize is the final stage of the
process (it feeds stage 2, the on-disk reorg); chat is a review-and-understand tool, so
it sits before it. People and their voices are resources of their own, so they sit
last: People first — the faces in the photos are named there, and every voice belongs
to a person — then Voices, which narrate the films.
