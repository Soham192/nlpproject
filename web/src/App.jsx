import { useCallback, useEffect, useRef, useState } from "react";
import { fileUrl, getEntityConfig, getManifest, runJob, upload } from "./api.js";
import DetectionTable from "./components/DetectionTable.jsx";
import Dropzone from "./components/Dropzone.jsx";
import EntityConfig from "./components/EntityConfig.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import RoundTripBadge from "./components/RoundTripBadge.jsx";
import StageProgress from "./components/StageProgress.jsx";
import WaveformView from "./components/WaveformView.jsx";

const idle = { status: "idle", job: null, result: null, error: null };
const sameSet = (a, b) => a && b && a.length === b.length && a.every((x) => b.includes(x));
const num = (v) => (v == null ? "n/a" : v.toFixed(3));

function ErrorBox({ error }) {
  if (!error) return null;
  return (
    <div className="error">
      <strong>Failed.</strong> {error.message}
      {error.hint && <div className="small">{error.hint}</div>}
    </div>
  );
}

export default function App() {
  const [cfg, setCfg] = useState(null);
  const [cfgError, setCfgError] = useState(null);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [entities, setEntities] = useState([]);
  const [detect, setDetect] = useState(idle);
  const [mask, setMask] = useState(idle);
  const [manifest, setManifest] = useState(null);
  const [unmask, setUnmask] = useState(idle);
  const [evaluation, setEvaluation] = useState(idle);
  const [focus, setFocus] = useState(null);
  const [redactedEl, setRedactedEl] = useState(null);
  const originalEl = useRef(null);
  const detectSeq = useRef(0);

  useEffect(() => {
    getEntityConfig()
      .then((c) => { setCfg(c); setEntities(c.defaults); })
      .catch((e) => setCfgError(e));
  }, []);

  const onUpload = async (f, gt) => {
    setUploading(true);
    setUploadError(null);
    try {
      const info = await upload(f, gt);
      setFile(info);
      setDetect(idle); setMask(idle); setManifest(null); setUnmask(idle); setEvaluation(idle); setFocus(null);
    } catch (e) {
      setUploadError(e);
    } finally {
      setUploading(false);
    }
  };

  const runDetect = useCallback(async (types) => {
    if (!file) return;
    const seq = ++detectSeq.current;
    setDetect((d) => ({ ...d, status: "running", error: null }));
    try {
      const result = await runJob("detect", { file_id: file.file_id, entity_types: types },
        (job) => seq === detectSeq.current && setDetect((d) => ({ ...d, job })));
      if (seq === detectSeq.current) setDetect((d) => ({ ...d, status: "done", result }));
    } catch (e) {
      if (seq === detectSeq.current) setDetect((d) => ({ ...d, status: "error", error: e }));
    }
  }, [file]);

  // After the first dry run, toggling an entity type re-runs detection live
  // (ASR + alignment are reused server-side, so this takes a couple of seconds).
  const onEntitiesChange = (types) => {
    setEntities(types);
    if (detect.result || detect.status === "running") runDetect(types);
  };

  const runMask = async () => {
    setMask({ ...idle, status: "running" });
    setUnmask(idle); setEvaluation(idle); setFocus(null);
    try {
      const result = await runJob("mask", { file_id: file.file_id, entity_types: entities },
        (job) => setMask((m) => ({ ...m, job })));
      setManifest(await getManifest(file.file_id)); // ciphertext/nonce/tag already stripped server-side
      setMask((m) => ({ ...m, status: "done", result, entities: [...entities], version: Date.now() }));
    } catch (e) {
      setMask((m) => ({ ...m, status: "error", error: e }));
    }
  };

  const runVerify = async () => {
    setUnmask({ ...idle, status: "running" });
    setEvaluation(idle);
    try {
      const result = await runJob("unmask", { file_id: file.file_id }, (job) => setUnmask((u) => ({ ...u, job })));
      setUnmask((u) => ({ ...u, status: "done", result }));
    } catch (e) {
      setUnmask((u) => ({ ...u, status: "error", error: e }));
      return;
    }
    setEvaluation({ ...idle, status: "running" });
    try {
      const result = await runJob("evaluate", { file_id: file.file_id }, (job) => setEvaluation((v) => ({ ...v, job })));
      setEvaluation((v) => ({ ...v, status: "done", result }));
    } catch (e) {
      setEvaluation((v) => ({ ...v, status: "error", error: e }));
    }
  };

  // Playing one player pauses the other.
  const onPlay = (which) => () => {
    const other = which === "original" ? redactedEl : originalEl.current;
    if (other && !other.paused) other.pause();
  };

  const d = detect.result;
  const maskStale = mask.status === "done" && !sameSet(mask.entities, entities);
  const willMask = d ? d.spans.length : 0;

  return (
    <div className="page">
      <header className="top">
        <h1>Selective audio redaction</h1>
        <p>Detect sensitive spans → encrypt them out of the audio → prove the rest is untouched and the original is recoverable.</p>
      </header>
      {cfgError && <div className="error"><strong>Cannot reach the API.</strong> Start it with <span className="mono">uvicorn api.main:app --port 8000</span>. ({cfgError.message})</div>}

      {/* 1 · Upload */}
      <section className="card">
        <h2><span className="num">1</span> Upload</h2>
        <p className="lede">A PCM WAV recording. It is stored server-side for this session only.</p>
        <Dropzone onUpload={onUpload} busy={uploading} info={file} error={uploadError} maxMb={cfg?.max_upload_mb} />
      </section>

      {/* 2 · Detect */}
      {file && cfg && (
        <section className="card">
          <h2><span className="num">2</span> Detect <span className="muted small">(dry run — nothing is modified)</span></h2>
          <p className="lede">
            Whisper transcribes, WhisperX aligns each word, Presidio finds sensitive spans. Spans scoring below{" "}
            <span className="mono">{cfg.score_threshold}</span> are reported but kept.
          </p>
          <div className="detect-grid">
            <div>
              <div className="row">
                <button onClick={() => runDetect(entities)} disabled={detect.status === "running"}>
                  {d ? "Re-run detection" : "Run detection"}
                </button>
                {detect.status === "running" && <span className="muted small">running… toggles re-run automatically</span>}
              </div>
              {detect.status === "running" && <StageProgress job={detect.job} reused={!!d} />}
              <ErrorBox error={detect.error} />
              {d && (
                <>
                  <div className="summary-strip" style={{ marginTop: 14 }}>
                    <div><b>{d.rows.filter((r) => r.will_mask).length}</b><span>detections to mask</span></div>
                    <div><b>{d.rows.filter((r) => !r.will_mask).length}</b><span>below threshold</span></div>
                    <div><b>{willMask}</b><span>merged ranges (+{cfg.padding_ms} ms padding)</span></div>
                    <div><b>{(100 * d.fraction_masked).toFixed(1)}%</b><span>of audio would be masked</span></div>
                    {d.entities && (
                      <>
                        <div><b>{num(d.entities.precision)}</b><span>precision vs ground truth</span></div>
                        <div><b>{num(d.entities.recall)}</b><span>recall vs ground truth</span></div>
                      </>
                    )}
                  </div>
                  <DetectionTable rows={d.rows} threshold={d.score_threshold} focus={focus} onFocus={setFocus} />
                </>
              )}
            </div>
            <EntityConfig available={cfg.available} defaults={cfg.defaults} selected={entities} onChange={onEntitiesChange} />
          </div>
        </section>
      )}

      {/* 3 · Mask */}
      {file && d && (
        <section className="card">
          <h2><span className="num">3</span> Mask</h2>
          <p className="lede">
            Each span's original samples are AES-GCM encrypted into the manifest and replaced by silence in the shared file.
            Every other sample is left exactly as it was.
          </p>
          <div className="row">
            <button onClick={runMask} disabled={mask.status === "running" || willMask === 0}>
              {mask.status === "done" ? "Re-mask" : `Mask ${willMask} span${willMask === 1 ? "" : "s"}`}
            </button>
            {willMask === 0 && <span className="muted small">Nothing to mask with the current entity selection.</span>}
          </div>
          {mask.status === "running" && <StageProgress job={mask.job} reused={!!d} />}
          <ErrorBox error={mask.error} />
          {maskStale && (
            <div className="notice warn">The entity selection changed since this file was masked. Re-mask to apply it.</div>
          )}
          {mask.status === "done" && manifest && (
            <>
              <div className="summary-strip" style={{ marginTop: 14 }}>
                <div><b>{manifest.stats.spans_masked}</b><span>spans encrypted</span></div>
                <div><b>{manifest.stats.duration_masked_sec.toFixed(2)} s</b><span>of {manifest.source.duration_sec.toFixed(1)} s masked</span></div>
                <div><b className="mono" style={{ fontSize: "1rem" }}>{manifest.masked.filename}</b><span>+ manifest (access-controlled)</span></div>
                <div><b>{manifest.timings_sec.total.toFixed(1)} s</b><span>pipeline time</span></div>
              </div>
              <WaveformView
                url={fileUrl(file.file_id, "masked.wav", mask.version)}
                media={redactedEl}
                spans={manifest.spans}
                focus={focus}
              />
            </>
          )}
          {mask.status === "done" && (
            <div className="players">
              <div className="player pii">
                <h3>Original (contains PII — demo only)</h3>
                {cfg.allow_original_playback ? (
                  <>
                    <div className="small muted">
                      Served only because <span className="mono">api.allow_original_playback</span> is on. A real deployment turns this off.
                    </div>
                    <audio ref={originalEl} controls preload="none" onPlay={onPlay("original")}
                      src={fileUrl(file.file_id, "original.wav")} />
                  </>
                ) : (
                  <div className="small muted">
                    Disabled: <span className="mono">api.allow_original_playback</span> is off, as it should be outside a demo.
                  </div>
                )}
              </div>
              <div className="player safe">
                <h3>Redacted (safe to share)</h3>
                <div className="small muted">Plays without any key. Linked to the waveform above.</div>
                <audio ref={setRedactedEl} controls onPlay={onPlay("redacted")} />
              </div>
            </div>
          )}
        </section>
      )}

      {/* 4 · Verify */}
      {file && mask.status === "done" && manifest && (
        <section className="card">
          <h2><span className="num">4</span> Verify</h2>
          <p className="lede">
            Restore the original on the server and compare digests, then re-transcribe the masked audio to measure what masking cost.
          </p>
          <div className="row">
            <button onClick={runVerify} disabled={unmask.status === "running" || evaluation.status === "running"}>
              {unmask.status === "done" ? "Re-run verification" : "Run verification"}
            </button>
            {unmask.status === "running" && <span className="muted small">unmasking on the server…</span>}
            {evaluation.status === "running" && (
              <span className="muted small">re-transcribing the masked audio for WER / BERTScore (about half the masking time)…</span>
            )}
          </div>
          <ErrorBox error={unmask.error} />
          {unmask.status === "done" && <div style={{ marginTop: 16 }}><RoundTripBadge result={unmask.result} /></div>}
          {evaluation.status === "running" && <StageProgress job={evaluation.job} />}
          <ErrorBox error={evaluation.error} />
          {evaluation.status === "done" && <MetricsPanel metrics={evaluation.result} />}
        </section>
      )}
    </div>
  );
}
