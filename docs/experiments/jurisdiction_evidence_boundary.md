# Jurisdiction Evidence Boundary

## MultiEURLEX

MultiEURLEX contains aligned EU legal documents in multiple languages. Its
language partitions share the `european_regulatory_law` authority scope. A
language holdout therefore supports multilingual EU-law transfer only. It does
not identify independent national courts, national authority hierarchies, or
national rules, even when the language is associated with a member state.

The MultiEURLEX runner records `claim_scope: multilingual_eu_law_transfer` and
sets `national_jurisdiction_claim_permitted: false` in every acceptance manifest.
The held-out language is never used for threshold selection.

## FEDLEGAL

FEDLEGAL is a separate authentic-court benchmark. Its ACL 2023 paper reports
public Chinese court judgments with natural partitions by city/court or case
category. It can support a comparison of FedAvg, FedProx, and FLEN under
real-court non-IID data after the released data archive is obtained, its schema
is mapped without inference, and the metadata audit passes.

That evidence should be described as cross-court or cross-region transfer within
one national legal system. It still cannot support independent-country legal
rule transfer.

## Release gates for national transfer

An independent-national-jurisdiction claim requires a common task ontology,
explicit jurisdiction metadata in every record, authoritative source or court
metadata, disjoint source/target identifiers, an audited rule mapping, and
expert adjudication of transferability/conflict labels. Results from unrelated
country-specific tasks must not be merged into one accuracy score.

Until those gates pass, the paper must use the narrower terms “multilingual
EU-law transfer” and “cross-court non-IID evaluation,” and must not call either
one cross-national rule migration.
