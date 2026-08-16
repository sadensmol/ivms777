async function tick() {
  try {
    const r = await fetch("/api/resources");
    const d = await r.json();
    const gb = (mb) => (mb / 1024).toFixed(1);
    const lease = d.workload
      ? `${d.workload.toLowerCase()} · ${d.models.join("+")} · ${gb(d.budget_used_mb)} GB`
      : "idle";
    document.getElementById("resbar").textContent =
      `RAM ${gb(d.ram_used_mb)}/${gb(d.ram_total_mb)} GB · CPU ${Math.round(d.cpu_pct)}% · ${lease}`;
  } catch (_) { /* best-effort */ }
}
tick();
setInterval(tick, 2000);
