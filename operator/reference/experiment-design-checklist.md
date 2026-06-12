# reference/experiment-design-checklist.md

Phase 1 runs this checklist before emitting the experiment plan. Every item resolves to a value or a documented inference -- never "unknown."

## Intake checklist

- [ ] **Target variable identified.** If undefined → the only Phase-1 escalation allowed.
- [ ] **Target sanity:** unique-value count, distribution/imbalance computed by the operator (don't trust supplied numbers -- verify if data is available)
- [ ] **Row and feature counts** confirmed
- [ ] **Feature type census:** numeric / categorical / text / image / datetime counts
- [ ] **Temporal flag:** any datetime column, or row order that implies time? → forces time-based split (Rule 1.4)
- [ ] **Entity/group flag:** any ID column with repeated values (patient, user, SKU, site)? → forces group k-fold (Rule 1.4)
- [ ] **Near-duplicate scan:** > 1% near-dup rows → dedupe before split, document count
- [ ] **Missingness map:** % missing per feature; any feature > 30% triggers rubric adjustment
- [ ] **Compute budget** stated by researcher; if absent, default to {{DEFAULT_BUDGET}} and state the default in the plan
- [ ] **Researcher requests logged:** model asks, metric asks -- each marked HONORED or OVERRIDDEN (with rule citation)

## Plan emission checklist

- [ ] Task framing + any overrides at the TOP of the plan
- [ ] Candidates listed with simplicity class and budget share
- [ ] Both mandatory baselines present (rubric)
- [ ] Primary metric + secondaries + any disqualified metrics named
- [ ] Validation strategy with the rule that selected it
- [ ] Budget allocation sums to 90% (10% reserve held for Phase 2 extensions)
- [ ] Leakage pre-screen note: which features look post-outcome and will be watched (Phase 2 S2 audit list seeded here)

## Common mis-framings to auto-correct

| Researcher says | Operator does |
|---|---|
| "Classify this score from 1–100" | Regression (or ordinal if genuinely discrete buckets); states override |
| "Optimize accuracy" on >10:1 imbalance | PR-AUC primary; accuracy disqualified (Rule 1.3) |
| "Use a random 80/20 split" on temporal data | Time-based split; states override |
| "Just sweep the neural net" on 800 rows | Class gate applies; linear + GBT primary, NN excluded, gate cited |
| "Skip baselines, I know the data" | Baselines run anyway -- non-negotiable (Rule 1.2) |
