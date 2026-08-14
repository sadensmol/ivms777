// Drives the three steps in spec section 3.2b: hash locally, ask what is new,
// send only that.

const CONCURRENCY = 4;

// Every still-image format worth ingesting, including phone and camera RAW.
// The server opens what it can and rejects the rest, so this list only has to
// keep obvious non-photos out of the upload.
const IMAGE_PATTERN =
  /\.(jpe?g|jpe|jfif|png|gif|bmp|webp|tiff?|heic|heif|hif|avif|jxl|dng|cr2|cr3|nef|nrw|arw|sr2|srf|raf|orf|rw2|pef|raw|3fr|erf|kdc|mos|mrw|x3f|gpr)$/i;
// Anything moving. Belt-and-braces with the image whitelist above: a folder of
// photos may still hold clips, and those must never be uploaded.
const VIDEO_PATTERN =
  /\.(mp4|mov|m4v|avi|mkv|webm|wmv|flv|mpe?g|mts|m2ts|3gp|3g2|ogv|mxf|insv)$/i;

const picker = document.getElementById('picker');
const startButton = document.getElementById('start');
const clearButton = document.getElementById('clear');
const status = document.getElementById('status');
const bar = document.getElementById('bar');
const failures = document.getElementById('upload-failures');

// Accumulates across picks so several folders upload as one batch. Keyed by
// relative path so re-picking the same folder does not queue it twice.
const selectedByPath = new Map();

function selectedEntries() {
  return Array.from(selectedByPath.values());
}

function say(text) {
  status.textContent = text;
}

function setProgress(done, total) {
  bar.max = total || 1;
  bar.value = done;
}

function noteFailure(relPath, message) {
  const item = document.createElement('li');
  item.textContent = `${relPath} — ${message}`;
  failures.appendChild(item);
}

function isPhoto(name) {
  return IMAGE_PATTERN.test(name) && !VIDEO_PATTERN.test(name);
}

picker.addEventListener('change', () => {
  let added = 0;
  for (const file of picker.files) {
    if (!isPhoto(file.name)) continue;
    const relPath = file.webkitRelativePath || file.name;
    if (selectedByPath.has(relPath)) continue;
    selectedByPath.set(relPath, { file, relPath });
    added += 1;
  }
  // Reset the input so picking the very same folder again still fires 'change'.
  picker.value = '';
  const total = selectedByPath.size;
  say(`${total} photos selected${added ? '' : ' (folder already added)'} — pick more folders or start`);
  startButton.disabled = total === 0;
  clearButton.disabled = total === 0;
});

clearButton.addEventListener('click', () => {
  selectedByPath.clear();
  failures.replaceChildren();
  setProgress(0, 1);
  say('selection cleared');
  startButton.disabled = true;
  clearButton.disabled = true;
});

function hashAll(entries) {
  return new Promise((resolve) => {
    const worker = new Worker('/static/hash-worker.js');
    const hashes = new Array(entries.length).fill(null);
    let done = 0;
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === 'hashed') {
        hashes[message.index] = message.hash;
      } else if (message.type === 'error') {
        noteFailure(entries[message.index].relPath, message.message);
      } else if (message.type === 'done') {
        worker.terminate();
        resolve(hashes);
        return;
      }
      done += 1;
      setProgress(done, entries.length);
      say(`hashing ${done} / ${entries.length}`);
    };
    worker.postMessage({ files: entries });
  });
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

async function sendOne(uploadId, entry, hash) {
  const form = new FormData();
  form.append('upload_id', uploadId);
  form.append('rel_path', entry.relPath);
  form.append('content_hash', hash);
  form.append('file', entry.file, entry.file.name);
  const response = await fetch('/api/upload/file', { method: 'POST', body: form });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.status }));
    noteFailure(entry.relPath, detail.detail);
  }
}

async function run() {
  startButton.disabled = true;
  clearButton.disabled = true;
  failures.replaceChildren();

  const selected = selectedEntries();
  const rootLabel = selected[0]?.relPath.split('/')[0] || 'photos';
  const hashes = await hashAll(selected);

  const { upload_id: uploadId } = await postJSON('/api/upload/start', {
    root_label: rootLabel,
  });

  const pairs = selected
    .map((entry, index) => ({ entry, hash: hashes[index] }))
    .filter((pair) => pair.hash !== null);

  const { needed } = await postJSON('/api/upload/probe', {
    upload_id: uploadId,
    files: pairs.map((pair) => ({ hash: pair.hash, rel_path: pair.entry.relPath })),
  });

  const needSet = new Set(needed);
  // One file per hash: the rest are copies whose paths the probe already recorded.
  const seen = new Set();
  const queue = pairs.filter((pair) => {
    if (!needSet.has(pair.hash) || seen.has(pair.hash)) return false;
    seen.add(pair.hash);
    return true;
  });

  const total = queue.length;
  say(`${total} of ${pairs.length} need uploading`);
  let done = 0;
  setProgress(0, total);

  async function drain() {
    while (queue.length > 0) {
      const pair = queue.shift();
      await sendOne(uploadId, pair.entry, pair.hash);
      done += 1;
      setProgress(done, total);
      say(`uploading ${done} / ${total}`);
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, drain));

  const summary = await postJSON('/api/upload/finish', { upload_id: uploadId });
  say(`done — ${summary.sent} uploaded, ${summary.offered} seen, ${summary.failed} failed`);
  selectedByPath.clear();
  if (window.htmx) htmx.trigger('#progress', 'refresh');
}

startButton.addEventListener('click', () => {
  run().catch((error) => {
    say(`upload failed: ${error.message}`);
    startButton.disabled = false;
    clearButton.disabled = selectedByPath.size === 0;
  });
});
