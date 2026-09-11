# Conflict Calibration Data Contract

The FLEN runtime consumes `conflict_calibration_manifest.json`, which records the
probe location, record count, and routing metadata. Two evidence levels are supported:

- `automatic_proxy_pilot`: permits implementation and optimization screening only;
- `expert_adjudicated`: required before legal conflict, authority transfer, or
  cross-jurisdiction claims may be released.

The CaseHOLD development pilot uses held-out training examples as a public probe,
maps gold-label verdict error to client-specific exposure, and treats all clients
as transferable within one US CaseHOLD task scope. It does not infer independent
jurisdictions from heuristic partitions.

Formal calibration requires two independent legally trained annotators and an
adjudicator. Records must follow `expert_annotation_schema.json`; no expert labels
are generated automatically by the repository.
