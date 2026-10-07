const GROUPS = [
  ["Pattern-based (regex)", ["CREDIT_CARD", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE", "DATE_TIME"]],
  ["NER-based (spaCy)", ["PERSON", "LOCATION", "NRP"]],
  ["Custom recognizers", ["ACCOUNT_NUMBER", "MEDICAL_RECORD_NUMBER", "POLICY_NUMBER"]],
];

/** Entity-type checkboxes, defaulting to config.yaml. Toggling re-runs detection. */
export default function EntityConfig({ available, defaults, selected, onChange, disabled }) {
  const known = new Set(GROUPS.flatMap(([, g]) => g));
  const groups = [
    ...GROUPS.map(([name, g]) => [name, g.filter((e) => available.includes(e))]),
    ["Other", available.filter((e) => !known.has(e))],
  ].filter(([, g]) => g.length);

  const toggle = (e) =>
    onChange(selected.includes(e) ? selected.filter((x) => x !== e) : available.filter((x) => x === e || selected.includes(x)));

  const isDefault = defaults.length === selected.length && defaults.every((e) => selected.includes(e));

  return (
    <div className="entity-config">
      <h3>Entity types</h3>
      <div className="small muted">Defaults from <span className="mono">config.yaml</span></div>
      {groups.map(([name, list]) => (
        <div key={name}>
          <h4>{name}</h4>
          {list.map((e) => (
            <label key={e}>
              <input type="checkbox" checked={selected.includes(e)} disabled={disabled} onChange={() => toggle(e)} />
              {e}
            </label>
          ))}
        </div>
      ))}
      <button className="ghost small" style={{ marginTop: 10, padding: "4px 10px" }} disabled={disabled || isDefault}
        onClick={() => onChange(defaults)}>
        Reset to defaults
      </button>
    </div>
  );
}
