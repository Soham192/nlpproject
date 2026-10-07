import { useRef, useState } from "react";

/** WAV dropzone + optional ground-truth JSON. Validation happens server-side (same rules as the CLI). */
export default function Dropzone({ onUpload, busy, info, error, maxMb }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const [gt, setGt] = useState(null);
  const [gtError, setGtError] = useState(null);
  const [chosen, setChosen] = useState(null);
  const gtInput = useRef(null);

  const pick = (files) => {
    const f = files?.[0];
    if (!f) return;
    setChosen(f.name);
    onUpload(f, gt);
  };

  return (
    <>
      <div
        className={`dropzone ${over ? "over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files); }}
      >
        <button type="button" disabled={busy} onClick={(e) => { e.stopPropagation(); input.current?.click(); }}>
          {busy ? "Uploading…" : "Choose WAV file"}
        </button>
        <p style={{ marginTop: 10 }}><b>or drop it here</b></p>
        <p className="muted small">
          16 kHz · mono · 16-bit PCM WAV, up to {maxMb ?? "…"} MB. Compressed formats are rejected, not converted.
        </p>
        {chosen && <p className="small">Selected: <span className="mono">{chosen}</span>{busy ? " — uploading…" : ""}</p>}
      </div>
      {/* Kept outside the clickable dropzone: a file input nested inside it receives its own
          click back via bubbling, which some browsers treat as a second, cancelling open. */}
      <input
        ref={input}
        type="file"
        accept=".wav,audio/wav,audio/x-wav"
        style={{ display: "none" }}
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = ""; // so choosing the same file again still triggers an upload
        }}
      />

      <p className="small muted" style={{ marginTop: 10 }}>
        Optional, for precision/recall:{" "}
        <a href="#" onClick={(e) => { e.preventDefault(); gtInput.current?.click(); }}>
          {gt ? `ground truth: ${gt.name} (change)` : "attach a ground-truth JSON"}
        </a>
        {gt && <> · <a href="#" onClick={(e) => { e.preventDefault(); setGt(null); }}>remove</a></>}
        {" "}— choose it <i>before</i> the WAV. Files from <span className="mono">data/samples/</span> pick up
        their <span className="mono">*.ground_truth.json</span> automatically.
      </p>
      <input
        ref={gtInput}
        type="file"
        accept=".json,application/json"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0] ?? null;
          e.target.value = "";
          if (!f) return;
          if (/\.wav$/i.test(f.name)) {
            // Picked the recording in the labels picker — just upload it as the recording.
            setGtError(null);
            pick([f]);
            return;
          }
          if (!/\.json$/i.test(f.name)) {
            setGtError(`"${f.name}" is not a ground-truth JSON file.`);
            return;
          }
          setGtError(null);
          setGt(f);
        }}
      />

      {gtError && <div className="error">{gtError}</div>}

      {error && (
        <div className="error">
          <strong>File rejected.</strong> {error.hint ?? ""}
          <div className="mono small" style={{ marginTop: 4 }}>{error.message}</div>
        </div>
      )}

      {info && (
        <dl className="fileinfo">
          <dt>File</dt><dt>Duration</dt><dt>Sample rate</dt><dt>Channels</dt>
          <dd className="mono">{info.filename}</dd>
          <dd>{info.duration.toFixed(2)} s</dd>
          <dd>{(info.sample_rate / 1000).toFixed(1)} kHz</dd>
          <dd>{info.channels === 1 ? "mono" : info.channels}</dd>
        </dl>
      )}
      {info && (
        <p className="small muted" style={{ marginBottom: 0 }}>
          Ground truth: {info.has_ground_truth ? `${info.ground_truth_entities} labelled entities loaded` : "none — precision/recall will be unavailable"}.
        </p>
      )}
    </>
  );
}
