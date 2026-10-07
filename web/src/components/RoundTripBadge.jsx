const short = (h) => (h ? `${h.slice(0, 16)}…${h.slice(-8)}` : "—");

/** The project's strongest single result: unmask on the server, compare SHA-256 digests. */
export default function RoundTripBadge({ result }) {
  const pass = result.match && result.source_sha256 === result.restored_sha256;
  return (
    <div className={`badge ${pass ? "pass" : "fail"}`}>
      <div className="verdict">{pass ? "✓ PASS" : "✗ FAIL"}</div>
      <div className="title">
        {pass ? "Original restored byte-for-byte from masked audio + manifest + key" : "Restored audio does NOT match the original"}
      </div>
      <div className="muted small">
        Unmasking ran on the server with the server-held key; the key never left it. Digests are SHA-256 of the whole WAV file.
      </div>
      <dl className="digests">
        <dt>original (from manifest)</dt>
        <dd title={result.source_sha256}>{short(result.source_sha256)}</dd>
        <dt>restored (just computed)</dt>
        <dd title={result.restored_sha256}>{short(result.restored_sha256)}</dd>
        <dt>masked (for contrast)</dt>
        <dd title={result.masked_sha256} className="muted">{short(result.masked_sha256)}</dd>
      </dl>
    </div>
  );
}
