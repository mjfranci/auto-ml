# Stage 02 -- Experiment Design

Apply Phase 1 decision rules to design the experiment plan. Override researcher preferences that conflict with the rules and state all overrides prominently in the plan header. Present the finished plan to the researcher for visibility, then execute it -- the Stage 01 -> 02 -> 03 flow never pauses for input or approval (Rule 0).

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Stage 01 output | `../01-intake/output/problem-brief.md` | Full file | Research goal, dataset profile, budget ceiling, skill level |
| Operator rules | `../../operator/rules.md` | Phase 1 -- §1.1 through §1.5 | Task framing, model selection, metrics, validation strategy |
| Reference | `../../operator/reference/experiment-design-checklist.md` | Full file | Completeness check before emitting the plan |

## Process

1. Apply §1.1 -- task framing; override mis-framed requests and state override in plan header
2. Apply §1.2 -- model family selection; always include both mandated baselines
3. Apply §1.3 -- metric selection; override researcher-stated metric if it conflicts; state override in plan header
4. Apply §1.4 -- validation strategy; apply time-based or group k-fold rules if triggered
5. Allocate compute budget across candidates; hold 10% reserve for Phase 2 extensions
6. Run the experiment-design-checklist; every item resolves to a value or a documented inference
7. Write the experiment plan to output and present it to the researcher for visibility, then proceed directly to Stage 03 -- presenting the plan is not a checkpoint; do not wait for input or approval (Rule 0)

## Audit

| Check | Pass Condition |
|-------|---------------|
| Checklist complete | Every item in experiment-design-checklist.md has a value or documented inference -- no "unknown" entries |
| Override flags visible | Any researcher-request override appears at the top of the plan, not buried in the body |
| Baselines present | Both mandatory baselines are listed as candidates with their budget share |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Experiment plan | `output/experiment-plan.md` | Structured doc: task framing, override flags, model candidates with simplicity classes, budget shares, and hyperparameter ranges (including `patience` and `min_delta` for early stopping), primary + secondary metrics, validation strategy |
