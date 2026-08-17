async function tick() {
  try {
    const r = await fetch("/api/resources");
    const d = await r.json();
    const gb = (mb) => (mb / 1024).toFixed(1);
    // CPU and GPU are ALWAYS shown, load and temperature together — "CPU 5% 51°C"
    // — so each number has an owner and the bar never changes shape as the box
    // goes idle. A missing sensor reads "—" rather than vanishing: a field that
    // disappears looks like a bug, and an idle GPU is 0%, not "no GPU".
    const pct = (v) => (v == null ? "—" : `${Math.round(v)}%`);
    const deg = (c) => (c == null ? "" : ` ${Math.round(c)}°C`);
    const models = (d.models && d.models.length) ? d.models.join("+") : null;
    // step: what the box is doing + what is loaded. e.g. "chat · siglip",
    // "captioning · caption", "idle · siglip", or just "idle" when nothing loaded.
    let step;
    if (d.active) step = models ? `${d.active} · ${models}` : d.active;
    else step = models ? `idle · ${models}` : "idle";
    document.getElementById("resbar").textContent =
      `RAM ${gb(d.ram_used_mb)}/${gb(d.ram_total_mb)} GB` +
      ` · CPU ${pct(d.cpu_pct)}${deg(d.cpu_c)}` +
      ` · GPU ${pct(d.gpu_pct)}${deg(d.gpu_c)}` +
      ` · ${step}`;
  } catch (_) { /* best-effort */ }
}
tick();
setInterval(tick, 2000);
