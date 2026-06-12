import urllib.request
import json
import time

class MLTracker:
    """
    Lightweight telemetry tracker for ML training loops.
    Sends metric checkpoints to the AutoML Experiment Pilot dashboard.
    Uses standard library urllib to ensure zero dependencies.
    """
    def __init__(self, run_name, model_type, simplicity_class=4, hyperparameters=None, baseline_metric=None, server_url="http://localhost:3000"):
        self.run_name = run_name
        self.model_type = model_type
        self.simplicity_class = simplicity_class
        self.hyperparameters = hyperparameters or {}
        self.baseline_metric = baseline_metric
        self.server_url = server_url.rstrip('/')
        self.run_id = None
        self._start_run()

    def _start_run(self):
        payload = {
            "run_name": self.run_name,
            "model_type": self.model_type,
            "simplicity_class": self.simplicity_class,
            "hyperparameters": self.hyperparameters,
            "primary_metric_name": self.hyperparameters.get("primary_metric", "loss")
        }
        if self.baseline_metric is not None:
            payload["baseline_metric"] = float(self.baseline_metric)
        
        req = urllib.request.Request(
            f"{self.server_url}/api/runs",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=3.0) as res:
                response = json.loads(res.read().decode('utf-8'))
                self.run_id = response.get("run_id")
                print(f"[Tracker] Connected to dashboard. Registered Run ID: {self.run_id}")
        except Exception as e:
            print(f"[Tracker] Warning: Could not register run at {self.server_url}. Dashboard is offline: {e}")

    def log_checkpoint(self, epoch, train_loss, val_loss, val_metric, train_metric=None, lr=None, additional_metrics=None):
        """
        Logs epoch metrics and returns triage decisions from the Auto-ML rules engine.
        Returns:
            dict: { status: 'CONTINUE'|'KILL'|'SUSPEND', reason: str, recommendation: str } or None
        """
        if not self.run_id:
            return None

        payload = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_metric": float(val_metric),
            "train_metric": float(train_metric) if train_metric is not None else None,
            "lr": float(lr) if lr is not None else None,
            "additional_metrics": additional_metrics or {}
        }

        req = urllib.request.Request(
            f"{self.server_url}/api/runs/{self.run_id}/checkpoints",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req, timeout=3.0) as res:
                response = json.loads(res.read().decode('utf-8'))
                triage = response.get("triage")
                if triage:
                    status = triage.get("status")
                    rule = triage.get("rule")
                    reason = triage.get("reason")
                    rec = triage.get("recommendation")
                    
                    if status in ('KILL', 'SUSPEND'):
                        print(f"\n[AutoML ALERT] [{rule}] [{status}] - {reason}")
                        print(f"[AutoML PRESCRIBED ACTION] - {rec}\n")
                    return triage
        except Exception as e:
            print(f"[Tracker] Warning: Failed to send checkpoint telemetry: {e}")
        return None

    def log_verdict(self, status="CONVERGED", message="Completed successfully.", final_metrics=None):
        """
        Signals run completion and final hyperparameters (e.g. parameter count, latency).
        """
        if not self.run_id:
            return

        payload = {
            "status": status,
            "message": message,
            "final_metrics": final_metrics or {}
        }

        req = urllib.request.Request(
            f"{self.server_url}/api/runs/{self.run_id}/complete",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req, timeout=3.0) as res:
                print(f"[Tracker] Run finalize message sent. Status: {status}")
        except Exception as e:
            print(f"[Tracker] Warning: Failed to send final run status: {e}")


if __name__ == "__main__":
    print("Testing MLTracker connection to dashboard at http://localhost:3000...")
    try:
        tracker = MLTracker("verification-run", "VerificationModel", simplicity_class=1, baseline_metric=0.5)
        if tracker.run_id:
            print("MLTracker connection verified successfully! Run ID:", tracker.run_id)
            tracker.log_verdict("CONVERGED", "Verification complete.")
        else:
            print("MLTracker connection failed: Dashboard returned invalid response or is offline.")
    except Exception as e:
        print(f"MLTracker connection failed: {e}")
