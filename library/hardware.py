import os
import sys
import torch
import gc

class HardwareOptimizer:
    """
    Utility class to automatically inspect host hardware and adjust training execution parameters.
    Handles device selection, optimal batch size estimation, mixed-precision settings,
    multi-process loading workers, and PyTorch CUDA Out-Of-Memory (OOM) recovery.
    """
    @staticmethod
    def inspect_system():
        """
        Gathers system CPU, RAM, and GPU info.
        """
        info = {
            "device_type": "cpu",
            "device_name": "CPU Only",
            "gpu_count": 0,
            "vram_gb": 0.0,
            "os": sys.platform,
            "cpu_cores": os.cpu_count() or 1,
            "supports_amp": False
        }

        if torch.cuda.is_available():
            info["device_type"] = "cuda"
            info["gpu_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            
            # Retrieve VRAM
            device_props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = device_props.total_memory / (1024 ** 3)
            
            # CUDA supports automatic mixed precision
            info["supports_amp"] = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device_type"] = "mps"
            info["device_name"] = "Apple Silicon GPU (MPS)"
            info["supports_amp"] = False # AMP support varies; usually disabled on MPS baseline

        return info

    def __init__(self, target_batch_size=64):
        self.sys_info = self.inspect_system()
        self.device = torch.device(self.sys_info["device_type"])
        self.target_batch_size = target_batch_size
        self.current_batch_size = target_batch_size
        self.use_amp = self.sys_info["supports_amp"]
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def get_dataloader_args(self):
        """
        Determines Windows-safe multiprocessing options for DataLoader.
        """
        # Multiprocessing on Windows can cause deadlocks/crashes in script main loops.
        # Safe baseline is 0. On Unix, we scale to min(4, CPU cores).
        if self.sys_info["os"].startswith("win"):
            num_workers = 0
            pin_memory = False
        else:
            num_workers = min(4, self.sys_info["cpu_cores"])
            pin_memory = self.sys_info["device_type"] == "cuda"
            
        return {
            "num_workers": num_workers,
            "pin_memory": pin_memory
        }

    def suggest_optimal_parameters(self, model_param_count=1000000):
        """
        Heuristically scales batch size and precision based on model parameters and VRAM size.
        """
        info = {
            "device": self.device,
            "use_amp": self.use_amp,
            "optimizer_precision": "float16" if self.use_amp else "float32"
        }

        # Suggest optimal batch size
        if self.sys_info["device_type"] == "cuda":
            vram = self.sys_info["vram_gb"]
            # Heuristic scaling: Adjust batch size based on available VRAM
            if vram < 4.0:
                self.current_batch_size = max(8, self.target_batch_size // 4)
            elif vram < 8.0:
                self.current_batch_size = max(16, self.target_batch_size // 2)
            else:
                self.current_batch_size = self.target_batch_size
        else:
            # CPU/MPS defaults to lower batch sizes to conserve memory
            self.current_batch_size = min(32, self.target_batch_size)

        info["batch_size"] = self.current_batch_size
        return info

    def clean_cache(self):
        """
        Releases unused PyTorch cache memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Clear mps cache if available
            pass

    def run_with_oom_guard(self, train_step_fn, *args, **kwargs):
        """
        Wraps a training epoch execution.
        If a CUDA Out-Of-Memory error occurs:
        1. Empties the PyTorch memory cache.
        2. Halves the current batch size.
        3. Returns a signal indicating that the batch size needs resizing.
        """
        try:
            return train_step_fn(*args, **kwargs), True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n[HardwareOptimizer] [OOM WARNING] Out-of-memory error detected during execution!")
                self.clean_cache()
                self.current_batch_size = max(4, self.current_batch_size // 2)
                print(f"[HardwareOptimizer] Memory cleared. Suggested smaller batch size: {self.current_batch_size}")
                return None, False
            else:
                raise e
