# Dashboard

Node.js live monitoring interface for the ML Experiment Pilot. The dashboard is the researcher's window into the autonomous execution loop -- it displays run status, checkpoint progress, and triage decisions in real time.

## Setup

```bash
npm install
node server.js
```

Dashboard runs at `http://localhost:3000` by default. Keep it running for the duration of any sweep.

See `stages/03-execution/references/setup-guide.md` for full environment requirements.

## What the dashboard shows

- Active and completed runs with their current status (running / converged / killed / suspended)
- Per-checkpoint primary metric and loss curves
- Triage decision log -- each decision cites the rule that triggered it
- Hard-stop alerts -- displayed prominently when a suspension condition fires

## Hard-stop notifications

When Stage 03 triggers a hard-stop condition, the dashboard displays the condition, the diagnosis, and the instruction to restart from Stage 01. The researcher does not need to take any action inside the dashboard -- they restart the pipeline conversation.

## Folder structure

```
dashboard/
├── README.md
├── package.json
├── server.js
└── public/
    ├── index.html
    ├── style.css
    └── app.js
```
