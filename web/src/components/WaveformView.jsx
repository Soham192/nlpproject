import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";

const REGION = "rgba(194, 24, 91, 0.22)";
const REGION_FOCUS = "rgba(194, 24, 91, 0.5)";
const fmt = (s) => s.toFixed(2);

/**
 * Waveform of the MASKED file with masked spans overlaid. Driven by the redacted
 * <audio> element (`media`), so the waveform and the "Redacted" player are one player.
 */
export default function WaveformView({ url, media, spans, focus }) {
  const container = useRef(null);
  const ws = useRef(null);
  const regions = useRef(null);
  const [tip, setTip] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!container.current || !media) return;
    const plugin = RegionsPlugin.create();
    const w = WaveSurfer.create({
      container: container.current,
      media,
      url,
      height: 140,
      waveColor: "#6B2FA0",
      progressColor: "#2D1B4E",
      cursorColor: "#C2185B",
      normalize: false,
      autoScroll: true,
      plugins: [plugin],
    });
    ws.current = w;
    regions.current = plugin;
    setReady(false);
    const unsub = w.on("decode", () => setReady(true));
    return () => {
      unsub();
      w.destroy();
      ws.current = null;
      regions.current = null;
    };
  }, [url, media]);

  // Draw the masked spans as regions once the audio is decoded.
  useEffect(() => {
    const plugin = regions.current;
    if (!ready || !plugin) return;
    plugin.clearRegions();
    for (const s of spans) {
      // Regions live in wavesurfer's shadow DOM, so page CSS can't reach them: style inline,
      // and clip so labels of adjacent narrow spans don't run into each other.
      const label = document.createElement("div");
      label.textContent = s.entity_type;
      label.style.cssText = "font:600 10px ui-monospace,Menlo,Consolas,monospace;color:#C2185B;padding:2px 3px;" +
        "white-space:nowrap;overflow:hidden;text-overflow:clip;max-width:100%;box-sizing:border-box;";
      const r = plugin.addRegion({ id: s.id, start: s.start_sec, end: s.end_sec, color: REGION, drag: false, resize: false, content: label });
      const text = `${s.entity_type} · ${fmt(s.start_sec)}–${fmt(s.end_sec)} s`;
      r.element?.addEventListener("mouseenter", () => {
        const box = r.element.getBoundingClientRect();
        const host = container.current.getBoundingClientRect();
        setTip({ text, x: box.left - host.left + box.width / 2 + 10, y: 10 });
      });
      r.element?.addEventListener("mouseleave", () => setTip(null));
    }
  }, [ready, spans]);

  // Clicking a table row: scroll to and highlight the region covering it.
  useEffect(() => {
    const w = ws.current;
    const plugin = regions.current;
    if (!ready || !w || !plugin || !focus) return;
    let hit = null;
    for (const r of plugin.getRegions()) {
      const on = r.start <= focus.end && focus.start <= r.end;
      r.setOptions({ color: on ? REGION_FOCUS : REGION });
      if (on && !hit) hit = r;
    }
    if (hit) {
      const dur = w.getDuration();
      const span = hit.end - hit.start;
      // Zoom so the span plus some context fills ~40% of the view, then centre it.
      const width = container.current.clientWidth;
      w.zoom(Math.max(width / dur, (width * 0.4) / Math.max(span, 0.5)));
      w.setScrollTime(Math.max(0, hit.start - (width / w.options.minPxPerSec - span) / 2));
      w.setTime(hit.start);
      container.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [focus, ready]);

  const zoomOut = () => ws.current?.zoom(0);

  return (
    <div className="waveform">
      <div ref={container} />
      {tip && <div className="wave-tooltip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
      <div className="wave-legend">
        <span><span className="swatch" style={{ background: REGION_FOCUS }} />masked span (silence in the shared file)</span>
        <span><span className="swatch" style={{ background: "#6B2FA0" }} />untouched audio — sample-identical to the original</span>
        <span style={{ marginLeft: "auto" }}>
          {!ready ? "decoding…" : <button className="ghost small" style={{ padding: "2px 10px" }} onClick={zoomOut}>Fit whole file</button>}
        </span>
      </div>
    </div>
  );
}
