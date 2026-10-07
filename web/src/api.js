// Thin fetch wrappers around the FastAPI backend. Nothing here ever handles key material:
// unmasking is triggered from the browser but performed entirely on the server.

export class ApiError extends Error {
  constructor(detail, status) {
    const message = typeof detail === "string" ? detail : detail?.message ?? `HTTP ${status}`;
    super(message);
    this.status = status;
    this.hint = typeof detail === "object" ? detail?.hint : undefined;
  }
}

async function req(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => null);
  if (!r.ok) throw new ApiError(body?.detail ?? r.statusText, r.status);
  return body;
}

const post = (path, body) =>
  req(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const getEntityConfig = () => req("/api/config/entities");

export function upload(file, groundTruth) {
  const fd = new FormData();
  fd.append("file", file);
  if (groundTruth) fd.append("ground_truth", groundTruth);
  return req("/api/upload", { method: "POST", body: fd });
}

export const getManifest = (fileId) => req(`/api/files/${fileId}/manifest`);

// Raw WAV bytes straight from disk. `v` busts the browser cache after a re-mask.
export const fileUrl = (fileId, name, v = "") => `/api/files/${fileId}/${name}${v ? `?v=${v}` : ""}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Start a job (detect | mask | unmask | evaluate) and poll until it finishes. */
export async function runJob(kind, body, onUpdate) {
  const { job_id } = await post(`/api/${kind}`, body);
  for (;;) {
    const job = await req(`/api/jobs/${job_id}`);
    onUpdate?.(job);
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new ApiError(job.error, 500);
    await sleep(400);
  }
}
