# Stage 04 -- Promotion Decision

Apply Phase 3 rules to the sweep results and emit exactly one verdict. The tiebreak chain resolves internally -- the decision does not escalate to the researcher.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Stage 03 output | `../03-execution/output/sweep-log.md` | Full file | All run results and triage history |
| Operator rules | `../../operator/rules.md` | Phase 3 -- §3.1 through §3.3 | Blocks, promotion logic, tiebreak chain, verdict format |
| Reference | `../../operator/reference/metric-priority-matrix.md` | "Eval-set sizing rule" and "Bootstrap CI procedure" | Eval set adequacy and confidence interval guidance |
| Reference | `../../operator/reference/model-selection-rubric.md` | "Simplicity classes" section | Tiebreak step 3 -- simplicity class assignments |

## Process

1. Check §3.1 blocks in order: leaky split → HARD BLOCK; eval set < 200 → no promotion; best model < 1% above baseline → kill all
2. If no block: apply §3.2 promotion logic -- clear winner, marginal gain, or tiebreak chain
3. Resolve tiebreak chain internally -- latency → size → simplicity class → secondary metric; chain always terminates
4. Write verdict following §3.3 format: one keyword, one-line decision, rule citations, evidence, single prescribed next action

## Audit

| Check | Pass Condition |
|-------|---------------|
| Verdict keyword | Exactly one of: `PROMOTED`, `PROMOTED-WITH-FLAGS`, `KILLED`, `BLOCKED -- <reason>` |
| Rule citations | Every decision traces to a section number in `operator/rules.md` |
| Next action | Exactly one prescribed next action -- not a list of options |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Verdict | `output/verdict.md` | Verdict keyword, one-line decision, rule citations, evidence, single prescribed next action |
