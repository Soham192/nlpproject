/** Real pipeline stages from the job status (asr → alignment → detection → mapping → masking). */
export default function StageProgress({ job, reused }) {
  if (!job) return null;
  const idx = job.stages.indexOf(job.stage);
  return (
    <div className="stages" aria-label="pipeline progress">
      {job.stages.map((s, i) => {
        let cls = "";
        if (job.status === "done") cls = "done";
        else if (reused && (s === "asr" || s === "alignment") && idx > 1) cls = "skipped";
        else if (i < idx) cls = "done";
        else if (i === idx) cls = "active";
        return (
          <div key={s} className={`stage ${cls}`}>
            {s}
            {cls === "skipped" ? " (reused)" : ""}
          </div>
        );
      })}
    </div>
  );
}
