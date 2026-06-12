const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// In-memory data store for runs
// A run object structure:
// {
//   id: string,
//   run_name: string,
//   model_type: string,
//   simplicity_class: number,
//   hyperparameters: object,
//   start_time: number,
//   end_time: number|null,
//   status: 'RUNNING' | 'CONVERGED' | 'KILLED' | 'SUSPENDED' | 'PROMOTED' | 'PROMOTED-WITH-FLAGS' | 'BLOCKED',
//   verdict_message: string|null,
//   checkpoints: Array<{
//     epoch: number,
//     train_loss: number,
//     val_loss: number,
//     val_metric: number,
//     lr: number|null,
//     additional_metrics: object,
//     timestamp: number
//   }>,
//   triage_history: Array<object>,
//   baseline_metric: number|null, // Default baseline for comparison
//   primary_metric_name: string
// }
const runs = {};
const globalConfig = {
  baseline_metric: 0.35, // Mock baseline (e.g. random guess / simple model score)
  primary_metric_direction: 'maximize' // 'maximize' (accuracy/F1) or 'minimize' (loss/MAE)
};

// Generates a simple ID
function generateId() {
  return Math.random().toString(36).substring(2, 9);
}

function getMetricDirection(metricName) {
  const lowerMetric = (metricName || 'loss').toLowerCase();
  if (lowerMetric.includes('loss') || lowerMetric.includes('mae') || lowerMetric.includes('mse') || lowerMetric.includes('rmse') || lowerMetric.includes('mape')) {
    return 'minimize';
  }
  return 'maximize';
}

// -------------------------------------------------------------
// Auto-ML Rules Engine (Phase 2 & Phase 3)
// -------------------------------------------------------------

function runPhase2Rules(run, newCheckpoint) {
  const checkpoints = [...run.checkpoints, newCheckpoint];
  const idx = checkpoints.length - 1;
  const current = newCheckpoint;
  
  // Rule 2.1 — Hard Stops (Override everything)
  
  // NaN/Inf loss Check (K4)
  if (isNaN(current.train_loss) || !isFinite(current.train_loss) || 
      isNaN(current.val_loss) || !isFinite(current.val_loss)) {
    return {
      status: 'KILL',
      rule: 'Rule 2.1 / K4',
      reason: 'Loss is NaN or Inf.',
      recommendation: 'Kill immediately. Check learning rate or numerical stability in scaling/data.'
    };
  }

  // S1: Val metric > train metric by margin (generalization anomaly/leakage)
  if (run.primary_metric_name === 'accuracy' || run.primary_metric_name === 'f1' || run.primary_metric_name === 'pr_auc') {
    // For maximization metrics
    if (current.train_metric !== undefined && current.train_metric !== null) {
      if (current.val_metric - current.train_metric > 0.05) {
        return {
          status: 'SUSPEND',
          rule: 'Rule 2.1 / S1',
          reason: `Val metric (${current.val_metric.toFixed(4)}) is > 5% above train metric (${current.train_metric.toFixed(4)}).`,
          recommendation: 'SUSPEND SWEEP. Possible data leakage or train/val split bug. Check duplicate records across splits.'
        };
      }
    }
  }

  // S2: Near-perfect train + mediocre val at checkpoint 1
  if (checkpoints.length === 1) {
    const isTrainPerfect = current.train_loss < 0.01 || 
      ((run.primary_metric_name === 'accuracy' || run.primary_metric_name === 'f1' || run.primary_metric_name === 'pr_auc') && 
       current.train_metric !== undefined && current.train_metric !== null && current.train_metric > 0.99);
    const isValMediocre = (run.primary_metric_name === 'accuracy' || run.primary_metric_name === 'f1' || run.primary_metric_name === 'pr_auc') && 
      current.val_metric < 0.85;
    if (isTrainPerfect && isValMediocre) {
      return {
        status: 'SUSPEND',
        rule: 'Rule 2.1 / S2',
        reason: 'Train metric is near perfect (>99% or near-zero loss) while val metric is mediocre at checkpoint 1.',
        recommendation: 'SUSPEND SWEEP. Probable target leakage in feature columns (e.g. features recorded post-outcome).'
      };
    }
  }

  // S4: Underperforming baseline at 25% budget
  const budgetCheckpoints = 20; // assumed run budget
  if (checkpoints.length >= Math.ceil(budgetCheckpoints * 0.25)) {
    const direction = getMetricDirection(run.primary_metric_name);
    const isBetterThanBaseline = direction === 'maximize' 
      ? current.val_metric > run.baseline_metric 
      : current.val_metric < run.baseline_metric;
    
    // Check if ALL live runs are below baseline (we'll look at this run specifically)
    if (!isBetterThanBaseline) {
      const allActiveRuns = Object.values(runs);
      const runsAtLimit = allActiveRuns.filter(r => r.checkpoints && r.checkpoints.length >= Math.ceil(20 * 0.25) && r.status !== 'SUSPENDED');
      
      const allBelow = runsAtLimit.length > 0 && runsAtLimit.every(r => {
        const lastCp = r.checkpoints[r.checkpoints.length - 1];
        const rDirection = getMetricDirection(r.primary_metric_name);
        const rBetter = rDirection === 'maximize'
          ? lastCp.val_metric > r.baseline_metric
          : lastCp.val_metric < r.baseline_metric;
        return !rBetter;
      });

      if (allBelow) {
        const margin = Math.abs(current.val_metric - run.baseline_metric);
        if (margin > 0.1) {
          return {
            status: 'SUSPEND',
            rule: 'Rule 2.1 / S4',
            reason: `ALL runs (including ${run.run_name}) are significantly below their baselines after 25% of budget.`,
            recommendation: 'SUSPEND SWEEP. Return to Phase 1 redesign. Feature set or framing is likely flawed.'
          };
        }
      }
    }
  }

  // Rule 2.2 — Per-run Triage

  // K1: Val loss rising for 3 consecutive checkpoints
  if (checkpoints.length >= 4) {
    let rising = true;
    for (let i = checkpoints.length - 1; i >= checkpoints.length - 3; i--) {
      if (checkpoints[i].val_loss <= checkpoints[i - 1].val_loss) {
        rising = false;
        break;
      }
    }
    if (rising) {
      return {
        status: 'KILL',
        rule: 'Rule 2.2 / K1',
        reason: 'Validation loss has risen for 3 consecutive checkpoints.',
        recommendation: 'Kill run due to overfitting. Re-run with stronger dropout, weight decay, or early stopping.'
      };
    }
  }

  // K2: Train-Val Gap widening and > 15% relative
  if (checkpoints.length >= 3) {
    const computeGap = (c) => {
      const diff = Math.abs(c.train_loss - c.val_loss);
      return diff / (c.train_loss || 1e-5);
    };
    const gap0 = computeGap(checkpoints[checkpoints.length - 3]);
    const gap1 = computeGap(checkpoints[checkpoints.length - 2]);
    const gap2 = computeGap(checkpoints[checkpoints.length - 1]);
    
    if (gap2 > 0.15 && gap2 > gap1 && gap1 > gap0) {
      return {
        status: 'KILL',
        rule: 'Rule 2.2 / K2',
        reason: `Train-val loss gap (${(gap2 * 100).toFixed(1)}%) exceeds 15% and has been widening for 2 checkpoints.`,
        recommendation: 'Kill run. Tag as "needs regularization". Try adding dropout, L2 normalization, or reducing model capacity.'
      };
    }
  }

  // K5: No improvement over 5 checkpoints (for runs not in top-performing)
  if (checkpoints.length >= 6) {
    let flat = true;
    const direction = getMetricDirection(run.primary_metric_name);
    const bestBefore = checkpoints.slice(0, checkpoints.length - 5).reduce((best, c) => {
      return direction === 'maximize' ? Math.max(best, c.val_metric) : Math.min(best, c.val_metric);
    }, direction === 'maximize' ? -Infinity : Infinity);

    for (let i = checkpoints.length - 5; i < checkpoints.length; i++) {
      const val = checkpoints[i].val_metric;
      const improvement = direction === 'maximize' 
        ? (val - bestBefore) / (bestBefore || 1e-5)
        : (bestBefore - val) / (bestBefore || 1e-5);
      if (improvement >= 0.001) {
        flat = false;
        break;
      }
    }

    if (flat) {
      return {
        status: 'KILL',
        rule: 'Rule 2.2 / K5',
        reason: 'Validation metric improved by less than 0.1% over the last 5 checkpoints.',
        recommendation: 'Kill run due to plateauing. Training has converged.'
      };
    }
  }

  // C1: Continue and potentially extend if performing well
  const direction = getMetricDirection(run.primary_metric_name);
  const lastVal = current.val_metric;
  const prevVal = checkpoints.length > 1 ? checkpoints[checkpoints.length - 2].val_metric : lastVal;
  const stepImprovement = direction === 'maximize' 
    ? (lastVal - prevVal) / (prevVal || 1e-5)
    : (prevVal - lastVal) / (prevVal || 1e-5);
  
  if (stepImprovement >= 0.005) {
    return {
      status: 'CONTINUE',
      rule: 'Rule 2.2 / C1',
      reason: `Metric improving steadily (${(stepImprovement * 100).toFixed(2)}% this epoch).`,
      recommendation: 'Continue sweep. Eligible for budget extension if it becomes a top candidate.'
    };
  }

  // Default
  return {
    status: 'CONTINUE',
    rule: 'Rule 2.2 / Default',
    reason: 'Normal learning trajectory.',
    recommendation: 'Continue monitoring checkpoints.'
  };
}

// Phase 3 - Sweep Promotion and Final Verdict
function calculatePromotionVerdict(allRuns, defaultBaseline, evalSetSize = 300) {
  const activeRuns = Object.values(allRuns).filter(r => r.checkpoints && r.checkpoints.length > 0);
  if (activeRuns.length === 0) {
    return {
      verdict: 'BLOCKED',
      rule: '3.1 / No data',
      message: 'No completed or tracked runs with checkpoints.',
      action: 'Run at least one model sweep before asking for promotion.'
    };
  }

  // Sort runs by best validation metric
  const firstRun = activeRuns[0];
  const direction = getMetricDirection(firstRun.primary_metric_name);
  const baseline = firstRun.baseline_metric !== null && firstRun.baseline_metric !== undefined 
    ? firstRun.baseline_metric 
    : defaultBaseline;

  const getBestMetric = (run) => {
    const rDirection = getMetricDirection(run.primary_metric_name);
    return run.checkpoints.reduce((best, c) => {
      return rDirection === 'maximize' ? Math.max(best, c.val_metric) : Math.min(best, c.val_metric);
    }, rDirection === 'maximize' ? -Infinity : Infinity);
  };

  const runsWithScores = activeRuns.map(run => {
    const bestScore = getBestMetric(run);
    const bestCheckpoint = run.checkpoints.find(c => c.val_metric === bestScore);
    const paramCount = run.hyperparameters.param_count || 1000000;
    const latency = run.hyperparameters.inference_latency_ms || 10.0;
    const secondaryMetric = run.hyperparameters.secondary_metric_value || (run.checkpoints[run.checkpoints.length - 1].additional_metrics && run.checkpoints[run.checkpoints.length - 1].additional_metrics.secondary_metric) || 0.0;
    return { run, bestScore, bestCheckpoint, paramCount, latency, secondaryMetric };
  });

  // Sort descending for maximize, ascending for minimize
  runsWithScores.sort((a, b) => {
    return direction === 'maximize' ? b.bestScore - a.bestScore : a.bestScore - b.bestScore;
  });

  const winner = runsWithScores[0];
  const runnerUp = runsWithScores[1];

  // Rule 3.1: Blocks (Eval set < 200 samples)
  if (evalSetSize < 200) {
    return {
      verdict: 'BLOCKED',
      rule: 'Rule 3.1 / Small Eval',
      message: `Evaluation set size (${evalSetSize}) is below the minimum threshold of 200. Statistical resolution is impossible.`,
      action: 'Collect additional evaluation samples (at least 200, preferably 500+) before promoting any model.',
      bestModel: winner.run.run_name,
      score: winner.bestScore
    };
  }

  // Beats baseline by < 1% relative -> signals not found
  const relativeGainOverBaseline = direction === 'maximize'
    ? (winner.bestScore - baseline) / (baseline || 1e-5)
    : (baseline - winner.bestScore) / (baseline || 1e-5);

  if (relativeGainOverBaseline < 0.01) {
    return {
      verdict: 'KILLED',
      rule: 'Rule 3.1 / No Signal',
      message: `Best model (${winner.run.run_name}: ${winner.bestScore.toFixed(4)}) beats baseline (${baseline.toFixed(4)}) by only ${(relativeGainOverBaseline * 100).toFixed(2)}% (threshold is 1%). No signal found.`,
      action: 'Abort deployment. Refactor features, collect clean training data, or redesign experiments (Phase 1).',
      bestModel: winner.run.run_name,
      score: winner.bestScore
    };
  }

  // Rule 3.2: Promotion logic & Near Ties
  
  // Tiebreak check: if runnerUp exists and within 0.5% relative
  let tiebroken = false;
  let tiebreakReason = '';
  let finalWinner = winner;

  if (runnerUp) {
    const scoreDiff = direction === 'maximize'
      ? (winner.bestScore - runnerUp.bestScore) / (runnerUp.bestScore || 1e-5)
      : (runnerUp.bestScore - winner.bestScore) / (runnerUp.bestScore || 1e-5);
    
    if (scoreDiff <= 0.005) {
      // Near tie! Run deterministic tiebreak chain:
      // 1. Latency: faster wins if >= 2x faster
      if (runnerUp.latency <= winner.latency / 2) {
        finalWinner = runnerUp;
        tiebroken = true;
        tiebreakReason = `Latency tiebreak: ${runnerUp.run.run_name} is 2x+ faster (${runnerUp.latency}ms vs ${winner.latency}ms) despite score tie.`;
      } else if (winner.latency <= runnerUp.latency / 2) {
        finalWinner = winner;
        tiebroken = true;
        tiebreakReason = `Latency tiebreak: ${winner.run.run_name} is 2x+ faster (${winner.latency}ms vs ${runnerUp.latency}ms).`;
      } else {
        // 2. Model size: smaller wins if >= 5x smaller
        if (runnerUp.paramCount <= winner.paramCount / 5) {
          finalWinner = runnerUp;
          tiebroken = true;
          tiebreakReason = `Model size tiebreak: ${runnerUp.run.run_name} is 5x+ smaller (${runnerUp.paramCount} params vs ${winner.paramCount}).`;
        } else if (winner.paramCount <= runnerUp.paramCount / 5) {
          finalWinner = winner;
          tiebroken = true;
          tiebreakReason = `Model size tiebreak: ${winner.run.run_name} is 5x+ smaller (${winner.paramCount} params vs ${runnerUp.paramCount}).`;
        } else {
          // 3. Simplicity class: lower class wins (linear [1] > trees [2] > shallow NN [3] > deep NN [4])
          if (runnerUp.run.simplicity_class < winner.run.simplicity_class) {
            finalWinner = runnerUp;
            tiebroken = true;
            tiebreakReason = `Simplicity class tiebreak: ${runnerUp.run.run_name} belongs to Simplicity Class ${runnerUp.run.simplicity_class} vs Class ${winner.run.simplicity_class}.`;
          } else if (winner.run.simplicity_class < runnerUp.run.simplicity_class) {
            finalWinner = winner;
            tiebroken = true;
            tiebreakReason = `Simplicity class tiebreak: ${winner.run.run_name} belongs to Simplicity Class ${winner.run.simplicity_class} vs Class ${runnerUp.run.simplicity_class}.`;
          } else {
            // 4. Secondary metric comparison (higher is better)
            const aSecondary = winner.secondaryMetric;
            const bSecondary = runnerUp.secondaryMetric;
            if (bSecondary > aSecondary) {
              finalWinner = runnerUp;
              tiebroken = true;
              tiebreakReason = `Secondary metric tiebreak: ${runnerUp.run.run_name} has better secondary metric (${bSecondary} vs ${aSecondary}).`;
            } else {
              finalWinner = winner;
              tiebroken = true;
              tiebreakReason = `Standard winner: tiebreaker resolved in favor of primary candidate (${winner.run.run_name}) over secondary comparison.`;
            }
          }
        }
      }
    }
  }

  // Check for small eval set confidence caveat
  const flags = [];
  if (evalSetSize >= 200 && evalSetSize <= 500) {
    flags.push('SMALL-EVAL-CAVEAT');
  }

  // Check for marginal gain
  if (relativeGainOverBaseline >= 0.01 && relativeGainOverBaseline < 0.02) {
    flags.push('MARGINAL-GAIN');
  }

  const finalVerdict = flags.length > 0 ? 'PROMOTED-WITH-FLAGS' : 'PROMOTED';
  let message = `Model ${finalWinner.run.run_name} is selected. Score: ${finalWinner.bestScore.toFixed(4)} vs Baseline ${baseline.toFixed(4)}.`;
  if (tiebroken) {
    message += ` (Resolved via ${tiebreakReason})`;
  }

  let action = 'None required. Model is ready for serving pipelines.';
  if (flags.includes('SMALL-EVAL-CAVEAT')) {
    action = 'Attach confidence interval (1000 resample bootstrap) and review small-eval caveat before deployment.';
  } else if (flags.includes('MARGINAL-GAIN')) {
    action = 'Compare operational deployment cost with trivial baseline since gain is marginal (1-2%).';
  }

  return {
    verdict: finalVerdict,
    rule: 'Rule 3.2',
    message: message,
    flags: flags,
    action: action,
    promotedRunId: finalWinner.run.id,
    promotedRunName: finalWinner.run.run_name,
    score: finalWinner.bestScore
  };
}

// -------------------------------------------------------------
// Express API Route Handlers
// -------------------------------------------------------------

// Create a new run
app.post('/api/runs', (req, res) => {
  const { run_name, model_type, simplicity_class, hyperparameters, baseline_metric } = req.body;
  
  if (!run_name || !model_type) {
    return res.status(400).json({ error: 'run_name and model_type are required.' });
  }

  const id = generateId();
  runs[id] = {
    id: id,
    run_name: run_name,
    model_type: model_type,
    simplicity_class: simplicity_class || 4,
    hyperparameters: hyperparameters || {},
    start_time: Date.now() / 1000,
    end_time: null,
    status: 'RUNNING',
    verdict_message: null,
    checkpoints: [],
    triage_history: [],
    baseline_metric: baseline_metric !== undefined && baseline_metric !== null ? parseFloat(baseline_metric) : globalConfig.baseline_metric,
    primary_metric_name: req.body.primary_metric_name || 'loss'
  };

  console.log(`[Server] Registered run: ${run_name} (${id})`);
  return res.json({ run_id: id, message: 'Run registered successfully.' });
});

// Post a new checkpoint metric for a run
app.post('/api/runs/:runId/checkpoints', (req, res) => {
  const { runId } = req.params;
  const checkpoint = req.body; // { epoch, train_loss, val_loss, val_metric, lr, additional_metrics }

  const run = runs[runId];
  if (!run) {
    return res.status(404).json({ error: 'Run not found.' });
  }

  if (run.status !== 'RUNNING') {
    return res.status(400).json({ error: 'Run is not in RUNNING state.' });
  }

  // Run triage rules
  const triage = runPhase2Rules(run, checkpoint);
  
  // Record checkpoint and triage decision
  run.checkpoints.push({
    ...checkpoint,
    timestamp: Date.now() / 1000
  });

  run.triage_history.push(triage);

  if (triage.status === 'KILL') {
    run.status = 'KILLED';
    run.end_time = Date.now() / 1000;
    run.verdict_message = `KILLED: ${triage.reason}`;
    console.log(`[Server] Run ${run.run_name} KILLED: ${triage.reason}`);
  } else if (triage.status === 'SUSPEND') {
    run.status = 'SUSPENDED';
    run.end_time = Date.now() / 1000;
    run.verdict_message = `SUSPENDED: ${triage.reason}`;
    console.log(`[Server] Run ${run.run_name} SUSPENDED: ${triage.reason}`);
  }

  return res.json({ triage });
});

// Complete a run manually/cleanly
app.post('/api/runs/:runId/complete', (req, res) => {
  const { runId } = req.params;
  const { status, message, final_metrics } = req.body;

  const run = runs[runId];
  if (!run) {
    return res.status(404).json({ error: 'Run not found.' });
  }

  run.status = status || 'CONVERGED';
  run.end_time = Date.now() / 1000;
  run.verdict_message = message || 'Run completed cleanly.';
  if (final_metrics) {
    run.hyperparameters = { ...run.hyperparameters, ...final_metrics };
  }

  console.log(`[Server] Run completed: ${run.run_name} - Status: ${run.status}`);
  return res.json({ message: 'Run marked as complete.' });
});

// Get all runs
app.get('/api/runs', (req, res) => {
  return res.json(Object.values(runs));
});

// Get a single run
app.get('/api/runs/:runId', (req, res) => {
  const { runId } = req.params;
  const run = runs[runId];
  if (!run) {
    return res.status(404).json({ error: 'Run not found.' });
  }
  return res.json(run);
});

// Request global promotion verdict based on active runs
app.get('/api/promotion-verdict', (req, res) => {
  const evalSize = parseInt(req.query.eval_size) || 300;
  const verdict = calculatePromotionVerdict(runs, globalConfig.baseline_metric, evalSize);
  return res.json(verdict);
});

// Configure settings
app.post('/api/config', (req, res) => {
  const { baseline_metric, primary_metric_direction } = req.body;
  if (baseline_metric !== undefined) {
    globalConfig.baseline_metric = parseFloat(baseline_metric);
  }
  if (primary_metric_direction !== undefined) {
    globalConfig.primary_metric_direction = primary_metric_direction;
  }
  return res.json(globalConfig);
});

app.get('/api/config', (req, res) => {
  return res.json(globalConfig);
});

// Serve frontend SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`===============================================`);
  console.log(`   AutoML Experiment Pilot Dashboard Ready      `);
  console.log(`   URL: http://localhost:${PORT}                `);
  console.log(`===============================================`);
});
