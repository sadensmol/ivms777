// The ⚙ model-settings overlay (design §13, §4.1).
//
// A <dialog>, deliberately: it is an overlay, NOT a navigation. No pushState, no
// URL change, so the [grid, leaf] history invariant (§13.1) is untouched and Esc
// closes it without touching the back stack. The body is fetched by HTMX on the
// button itself; this file only opens, closes, and returns focus.
(() => {
  const dialog = document.getElementById("settings");
  const open = document.getElementById("settings-open");
  const close = document.getElementById("settings-close");
  if (!dialog || !open) return;

  open.addEventListener("click", () => {
    if (!dialog.open) dialog.showModal();
  });
  close?.addEventListener("click", () => dialog.close());

  // Clicking the backdrop closes it: the click lands on the dialog element itself
  // (its children swallow their own clicks), so compare against the content box.
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    const outside =
      event.clientX < box.left || event.clientX > box.right ||
      event.clientY < box.top || event.clientY > box.bottom;
    if (outside) dialog.close();
  });

  // Focus goes back to the ⚙ the user pressed, not to the top of the page.
  dialog.addEventListener("close", () => open.focus());
})();
