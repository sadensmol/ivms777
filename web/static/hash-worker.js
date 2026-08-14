// Hashes one file at a time and posts the digest back. Runs off the main thread
// so a 5,000-file selection never freezes the tab.
//
// WebCrypto has no incremental digest, so each file is read whole. Files are
// handled strictly one at a time, which bounds memory to the largest single
// photo rather than the size of the selection.

function toHex(buffer) {
  return Array.from(new Uint8Array(buffer))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

self.onmessage = async (event) => {
  const { files } = event.data;
  for (let index = 0; index < files.length; index += 1) {
    const entry = files[index];
    try {
      const buffer = await entry.file.arrayBuffer();
      const digest = await crypto.subtle.digest('SHA-256', buffer);
      self.postMessage({ type: 'hashed', index, hash: toHex(digest) });
    } catch (error) {
      self.postMessage({ type: 'error', index, message: String(error) });
    }
  }
  self.postMessage({ type: 'done' });
};
