let runs = [];
let activeRunId = null;
let lossChartInstance = null;
let currentConfig = { baseline_metric: 0.35, primary_metric_direction: 'maximize' };

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
  fetchConfig();
  fetchRuns();
  
  // Poll runs data every 2 seconds
  setInterval(fetchRuns, 2000);
  
  // Save configurations button
  document.getElementById('saveConfigBtn').addEventListener('click', saveConfig);
  
  // Generate verdict button
  document.getElementById('generateVerdictBtn').addEventListener('click', generateVerdict);
});

// Fetch server config
async function fetchConfig() {
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    currentConfig = data;
    document.getElementById('baselineInput').value = data.baseline_metric;
  } catch (err) {
    console.error('Error fetching config:', err);
  }
}

// Save config adjustments
async function saveConfig() {
  const baseline = parseFloat(document.getElementById('baselineInput').value);
  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ baseline_metric: baseline })
    });
    currentConfig = await res.json();
    showToast('Settings applied.');
  } catch (err) {
    console.error('Error saving config:', err);
  }
}

// Fetch all runs telemetry
async function fetchRuns() {
  try {
    const res = await fetch('/api/runs');
    runs = await res.json();
    updateStatsGrid();
    renderRunList();
    if (activeRunId) {
      updateRunDetail(activeRunId);
    }
  } catch (err) {
    console.error('Error fetching runs:', err);
  }
}

// Update the top statistics grid
function updateStatsGrid() {
  const activeCount = runs.filter(r => r.status === 'RUNNING').length;
  const completedCount = runs.filter(r => r.status === 'CONVERGED').length;
  const killedCount = runs.filter(r => r.status === 'KILLED').length;
  const suspendedCount = runs.filter(r => r.status === 'SUSPENDED' || r.status === 'BLOCKED').length;

  document.getElementById('statActive').innerText = activeCount;
  document.getElementById('statCompleted').innerText = completedCount;
  document.getElementById('statKilled').innerText = killedCount;
  document.getElementById('statSuspended').innerText = suspendedCount;
  document.getElementById('runCountBadge').innerText = `${runs.length} Sweeps`;
}

// Render the left sidebar list of runs
function renderRunList() {
  const listEl = document.getElementById('runList');
  if (runs.length === 0) {
    listEl.innerHTML = '<div class="empty-state">No sweeps logged yet. Run a template file in `library/` to begin.</div>';
    return;
  }

  let html = '';
  runs.forEach(run => {
    let statusClass = 'badge-info';
    if (run.status === 'CONVERGED') statusClass = 'badge-success';
    if (run.status === 'KILLED') statusClass = 'badge-danger';
    if (run.status === 'SUSPENDED') statusClass = 'badge-warning';
    
    const activeClass = run.id === activeRunId ? 'active' : '';
    const checkpointCount = run.checkpoints ? run.checkpoints.length : 0;
    const modelClassLabel = `Class ${run.simplicity_class}`;

    html += `
      <div class="run-item ${activeClass}" onclick="selectRun('${run.id}')">
        <div class="run-item-header">
          <span class="run-item-name">${run.run_name}</span>
          <span class="badge ${statusClass}">${run.status}</span>
        </div>
        <div class="run-item-meta">
          <span>${run.model_type} (${modelClassLabel})</span>
          <span>${checkpointCount} epochs</span>
        </div>
      </div>
    `;
  });
  listEl.innerHTML = html;
}

// Select a specific run
function selectRun(runId) {
  activeRunId = runId;
  document.getElementById('emptyDetailState').style.display = 'none';
  document.getElementById('detailContainer').style.display = 'block';
  
  // Force update list active class
  const items = document.querySelectorAll('.run-item');
  items.forEach(it => it.classList.remove('active'));
  
  fetchRuns(); // update current details
}

// Update the main panel details for selected run
function updateRunDetail(runId) {
  const run = runs.find(r => r.id === runId);
  if (!run) return;

  document.getElementById('detailRunName').innerText = run.run_name;
  document.getElementById('detailModelType').innerText = `${run.model_type} (Simplicity Class ${run.simplicity_class}) • Tracking metric: ${run.primary_metric_name}`;
  
  // Status badge update
  const badge = document.getElementById('detailStatusBadge');
  badge.className = 'badge';
  if (run.status === 'RUNNING') badge.classList.add('badge-info');
  else if (run.status === 'CONVERGED') badge.classList.add('badge-success');
  else if (run.status === 'KILLED') badge.classList.add('badge-danger');
  else badge.classList.add('badge-warning');
  badge.innerText = run.status;

  // Hyperparameters
  const paramsGrid = document.getElementById('detailParams');
  let paramsHtml = '';
  const entries = Object.entries(run.hyperparameters);
  if (entries.length === 0) {
    paramsHtml = '<div class="empty-state">No parameters declared.</div>';
  } else {
    entries.forEach(([key, val]) => {
      let formattedVal = typeof val === 'number' && val % 1 !== 0 ? val.toFixed(5) : val;
      if (typeof val === 'object') formattedVal = JSON.stringify(val);
      paramsHtml += `
        <div class="param-item">
          <span class="param-label">${key.replace(/_/g, ' ')}</span>
          <span class="param-val">${formattedVal}</span>
        </div>
      `;
    });
  }
  paramsGrid.innerHTML = paramsHtml;

  // Chart Rendering
  renderChart(run);

  // Recommendations and alerts
  updateTriageBox(run);

  // Telemetry Table
  const tbody = document.getElementById('telemetryTableBody');
  if (!run.checkpoints || run.checkpoints.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No checkpoints received yet.</td></tr>';
  } else {
    let tbodyHtml = '';
    run.checkpoints.forEach((cp, idx) => {
      const triage = run.triage_history[idx] || { status: 'CONTINUE', rule: 'N/A', reason: 'Tuning...' };
      
      let triageBadge = '<span class="badge badge-info">CONTINUE</span>';
      if (triage.status === 'KILL') triageBadge = '<span class="badge badge-danger">KILL</span>';
      if (triage.status === 'SUSPEND') triageBadge = '<span class="badge badge-warning">SUSPEND</span>';
      
      const lrStr = cp.lr ? cp.lr.toExponential(2) : 'N/A';

      tbodyHtml += `
        <tr>
          <td><strong>Epoch ${cp.epoch}</strong></td>
          <td>${cp.train_loss.toFixed(4)}</td>
          <td>${cp.val_loss.toFixed(4)}</td>
          <td>${cp.val_metric.toFixed(4)}</td>
          <td><code>${lrStr}</code></td>
          <td>${triageBadge} <span class="subtitle">(${triage.rule})</span></td>
        </tr>
      `;
    });
    tbody.innerHTML = tbodyHtml;
  }
}

// Render the train vs val loss curves using Chart.js
function renderChart(run) {
  const ctx = document.getElementById('lossChart').getContext('2d');
  
  const epochs = run.checkpoints.map(cp => cp.epoch);
  const trainLoss = run.checkpoints.map(cp => cp.train_loss);
  const valLoss = run.checkpoints.map(cp => cp.val_loss);

  if (lossChartInstance) {
    lossChartInstance.destroy();
  }

  lossChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Train Loss',
          data: trainLoss,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          borderWidth: 2,
          tension: 0.2,
          fill: true
        },
        {
          label: 'Val Loss',
          data: valLoss,
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          borderWidth: 2,
          tension: 0.2,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#f3f4f6' }
        }
      }
    }
  });
}

// Update the Triage Alert Recommendation boxes
function updateTriageBox(run) {
  const box = document.getElementById('triageAlertBox');
  const icon = document.getElementById('triageAlertIcon');
  const title = document.getElementById('triageAlertTitle');
  const reason = document.getElementById('triageAlertReason');
  const rule = document.getElementById('triageAlertRule');
  const prescText = document.getElementById('prescriptionText');

  box.className = 'triage-status-alert';

  if (!run.checkpoints || run.checkpoints.length === 0) {
    icon.innerText = '⚙️';
    title.innerText = 'Run Queued';
    reason.innerText = 'Model weights initialized. Telemetry has not begun reporting metrics.';
    rule.innerText = 'Rule 2.2 / Init';
    prescText.innerText = 'Start the Python runner file to stream metrics.';
    return;
  }

  // Get last triage status
  const lastTriage = run.triage_history[run.triage_history.length - 1];
  
  if (lastTriage.status === 'KILL') {
    box.classList.add('alert-kill');
    icon.innerText = '🛑';
    title.innerText = 'Run Terminated (Auto-ML Kill)';
  } else if (lastTriage.status === 'SUSPEND') {
    box.classList.add('alert-suspend');
    icon.innerText = '⚠️';
    title.innerText = 'Sweep Suspended (Leakage Audit Fired)';
  } else {
    box.classList.add('alert-continue');
    icon.innerText = '🟢';
    title.innerText = 'Run Active and Optimizing';
  }

  reason.innerText = lastTriage.reason;
  rule.innerText = lastTriage.rule;
  prescText.innerText = lastTriage.recommendation;
}

// Request the Phase 3 final promotion verdict
async function generateVerdict() {
  const evalSize = parseInt(document.getElementById('evalSizeInput').value) || 300;
  
  try {
    const res = await fetch(`/api/promotion-verdict?eval_size=${evalSize}`);
    const verdict = await res.json();
    
    const resultsEl = document.getElementById('verdictResults');
    const banner = document.getElementById('verdictBanner');
    const title = document.getElementById('verdictTitle');
    const ruleTag = document.getElementById('verdictRuleTag');
    const msgText = document.getElementById('verdictMessageText');
    const actText = document.getElementById('verdictActionText');

    resultsEl.style.display = 'block';
    
    banner.className = 'verdict-banner';
    if (verdict.verdict === 'PROMOTED') {
      banner.classList.add('verdict-promoted');
      title.innerText = 'PROMOTED';
    } else if (verdict.verdict === 'PROMOTED-WITH-FLAGS') {
      banner.classList.add('verdict-flags');
      title.innerText = `PROMOTED [WITH FLAGS: ${verdict.flags.join(', ')}]`;
    } else if (verdict.verdict === 'KILLED') {
      banner.classList.add('verdict-killed');
      title.innerText = 'KILLED: NO SIGNAL';
    } else {
      banner.classList.add('verdict-killed');
      title.innerText = 'BLOCKED: HARDBLOCK';
    }

    ruleTag.innerText = verdict.rule;
    msgText.innerText = verdict.message;
    actText.innerText = verdict.action;
    
    // Highlight in the run list if we have a winner
    if (verdict.promotedRunId) {
      showToast(`Winner Promoted: ${verdict.promotedRunName}!`);
    } else {
      showToast(`Verdict Generated: ${verdict.verdict}`);
    }

    // Scroll to the verdict panel
    resultsEl.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    console.error('Error generating verdict:', err);
  }
}

// Simple toast feedback helper
function showToast(message) {
  const toast = document.createElement('div');
  toast.style.position = 'fixed';
  toast.style.bottom = '20px';
  toast.style.right = '20px';
  toast.style.background = 'rgba(139, 92, 246, 0.9)';
  toast.style.border = '1px solid rgba(255, 255, 255, 0.2)';
  toast.style.color = '#fff';
  toast.style.padding = '0.75rem 1.5rem';
  toast.style.borderRadius = '8px';
  toast.style.backdropFilter = 'blur(10px)';
  toast.style.zIndex = '9999';
  toast.style.fontFamily = 'var(--font-body)';
  toast.style.fontWeight = '600';
  toast.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.5)';
  toast.style.opacity = '0';
  toast.style.transition = 'opacity 0.3s ease-out';
  
  toast.innerText = message;
  document.body.appendChild(toast);
  
  // Fade in
  setTimeout(() => toast.style.opacity = '1', 50);
  // Fade out and remove
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
