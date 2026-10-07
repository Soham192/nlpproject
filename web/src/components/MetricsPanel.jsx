const pct = (v, d = 1) => (v == null ? "n/a" : `${(100 * v).toFixed(d)}%`);
const num = (v, d = 3) => (v == null ? "n/a" : v.toFixed(d));

function Metric({ value, label, explain }) {
  return (
    <div className="metric">
      <b>{value}</b>
      <div>
        <div className="label">{label}</div>
        <div className="explain">{explain}</div>
      </div>
    </div>
  );
}

/** Preservation metrics (non-masked spans only) and detection metrics vs ground truth. */
export default function MetricsPanel({ metrics }) {
  const { wer, bertscore, stats, entities } = metrics;
  const frac = stats.fraction_masked;
  return (
    <>
      <div className="metrics">
        <div className="metric-group">
          <h3>Preservation</h3>
          <div className="scope">Computed on non-masked spans only</div>
          <Metric
            value={pct(wer.wer, 2)}
            label="Word error rate"
            explain={`Re-transcribed masked audio vs. original, outside masked spans. ${wer.substitutions} sub · ${wer.deletions} del · ${wer.insertions} ins over ${wer.ref_words} words. Lower is better.`}
          />
          <Metric
            value={num(bertscore.f1, 4)}
            label="BERTScore F1"
            explain={`Semantic similarity of the two transcripts (${bertscore.model}); 1.0 means the meaning is unchanged.`}
          />
          <Metric
            value={metrics.outside_span_identical ? "identical" : "CHANGED"}
            label="Samples outside spans"
            explain="Every sample outside the masked ranges compared with the original."
          />
        </div>
        <div className="metric-group">
          <h3>Detection</h3>
          <div className="scope">Masked spans vs. hand-labelled ground truth</div>
          {entities ? (
            <>
              <Metric value={num(entities.precision)} label="Precision"
                explain={`${entities.true_positive_spans} of ${entities.masked_spans} masked spans overlap a real entity.`} />
              <Metric value={num(entities.recall)} label="Recall"
                explain={`${entities.recalled_entities} of ${entities.ground_truth_entities} labelled entities are ≥${entities.min_coverage * 100}% covered by masking.`} />
              <Metric value={num(entities.f1)} label="F1" explain="Harmonic mean of precision and recall." />
            </>
          ) : (
            <p className="muted small">No ground truth for this file. Upload a ground-truth JSON with the WAV to get precision, recall and F1.</p>
          )}
        </div>
      </div>
      <p className="usable">
        <b className="masked">{pct(frac)}</b> of this call was masked;{" "}
        <b className="kept">{pct(1 - frac)}</b> remains usable.
      </p>
    </>
  );
}
