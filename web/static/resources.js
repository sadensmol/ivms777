async function tick() {
  try {
    const r = await fetch("/api/resources");
    const d = await r.json();
    const gb = (mb) => (mb / 1024).toFixed(1);
    const gpu = d.gpu_pct == null ? "" : ` · GPU ${Math.round(d.gpu_pct)}%`;
    const models = (d.models && d.models.length) ? d.models.join("+") : null;
    // step: what the box is doing + what is loaded. e.g. "chat · siglip",
    // "captioning · caption", "idle · siglip", or just "idle" when nothing loaded.
    let step;
    if (d.active) step = models ? `${d.active} · ${models}` : d.active;
    else step = models ? `idle · ${models}` : "idle";
    document.getElementById("resbar").textContent =
      `RAM ${gb(d.ram_used_mb)}/${gb(d.ram_total_mb)} GB · CPU ${Math.round(d.cpu_pct)}%${gpu} · ${step}`;
  } catch (_) { /* best-effort */ }
}
tick();
setInterval(tick, 2000);
