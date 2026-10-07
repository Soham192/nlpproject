const fmt = (s) => s.toFixed(2);

/** Mirrors `redact detect`: every detection, masked or not. Below-threshold rows are greyed with the reason. */
export default function DetectionTable({ rows, threshold, focus, onFocus }) {
  if (!rows.length) {
    return <p className="muted">No entities detected with the selected types. Nothing would be masked.</p>;
  }
  return (
    <table className="detections">
      <thead>
        <tr>
          <th>Entity</th>
          <th>Preview</th>
          <th>Confidence</th>
          <th>Time (s)</th>
          <th>Masked?</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => {
          const focused = focus && Math.abs(focus.start - r.start_sec) < 1e-6 && Math.abs(focus.end - r.end_sec) < 1e-6;
          return (
            <tr
              key={i}
              className={[r.will_mask ? "masked clickable" : "low", focused ? "focused" : ""].join(" ")}
              onClick={() => r.will_mask && onFocus({ start: r.start_sec, end: r.end_sec })}
              title={r.will_mask ? "Show this span on the waveform" : r.reason}
            >
              <td className="mono">{r.entity_type}</td>
              <td className="mono">{r.text_preview}</td>
              <td>{r.score.toFixed(2)}</td>
              <td className="mono">{fmt(r.start_sec)}–{fmt(r.end_sec)}</td>
              <td>
                {r.will_mask ? (
                  <span className="tag mask">mask</span>
                ) : (
                  <>
                    <span className="tag keep">keep</span>{" "}
                    <span className="small">{r.reason} ({threshold})</span>
                  </>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
