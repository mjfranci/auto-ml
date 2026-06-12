# ml-experiment-pilot

Automated ML experiment pipeline for biomedical technology researchers.

## Task Routing

| Task Type | Go To | Description |
|-----------|-------|-------------|
| New experiment run | `stages/01-intake/CONTEXT.md` | Collect problem brief and dataset profile |
| Experiment design | `stages/02-experiment-design/CONTEXT.md` | Design the experiment plan from the brief |
| Execute sweep | `stages/03-execution/CONTEXT.md` | Autonomous training loop -- write, run, triage, adjust |
| Promotion verdict | `stages/04-promotion/CONTEXT.md` | Apply Phase 3 rules; emit one verdict |
| Library extension | `library/README.md` | Add a new model template |

## Shared Resources

| Resource | Location | Contains |
|----------|----------|---------|
| Decision rules | `operator/rules.md` | All three phases of operator decision logic |
| Operator reference | `operator/reference/` | Rubrics, matrices, checklists, stopping criteria |
| Model library | `library/` | PyTorch model templates and helper utilities |
| Dashboard | `dashboard/` | Live sweep monitoring interface for researchers |
