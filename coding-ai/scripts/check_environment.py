#!/usr/bin/env python3
"""Report the machine's capability and verify that GPU ops actually run.

Run this first on the target computer:

    python scripts/check_environment.py

It exits non-zero only when torch itself is missing or broken; a CPU-only
machine is a supported (slower) configuration, not an error.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time


def human_gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def main() -> int:
    info: dict = {"platform": platform.platform(), "python": sys.version.split()[0],
                  "machine": platform.machine(), "processor": platform.processor()}
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed.\n"
              "  CPU only:   pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
              "  GPU (cu121):pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
              "See docs/TECHNICAL_PLAN.md for the GTX 1060 (Pascal sm_61) recipe.")
        return 2

    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["threads"] = torch.get_num_threads()
    gpu_ok = False

    print(f"python      : {info['python']} ({info['platform']})")
    print(f"torch       : {torch.__version__}  (CUDA build: {torch.version.cuda})")
    print(f"cpu threads : {torch.get_num_threads()}")

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            info.update({"gpu": name, "compute_capability": f"{cap[0]}.{cap[1]}",
                         "vram": props.total_memory})
            print(f"gpu         : {name}  compute capability {cap[0]}.{cap[1]}  "
                  f"VRAM {human_gb(props.total_memory)}")
            # Actually execute an op on the GPU instead of trusting the listing.
            a = torch.randn(1024, 1024, device="cuda")
            b = torch.randn(1024, 1024, device="cuda")
            for _ in range(3):
                c = a @ b
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                c = a @ b
            torch.cuda.synchronize()
            gflops = 20 * 2 * 1024 ** 3 / (time.time() - t0) / 1e9
            info["gpu_matmul_gflops"] = round(gflops, 1)
            print(f"gpu matmul  : {gflops:.0f} GFLOPS (fp32)  -> GPU execution verified")
            gpu_ok = True
        except Exception as e:  # pragma: no cover - hardware dependent
            print(f"gpu         : listed but unusable: {e}")
            info["gpu_error"] = str(e)
    else:
        print("gpu         : none available - CPU inference/training path will be used")

    # CPU throughput probe (a 1024^3 matmul loop)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    for _ in range(3):
        a @ b
    t0 = time.time()
    for _ in range(50):
        a @ b
    print(f"cpu matmul  : {50 * 2 * 512 ** 3 / (time.time() - t0) / 1e9:.0f} GFLOPS (fp32)")

    try:
        import psutil  # optional
        mem = psutil.virtual_memory()
        print(f"ram         : {human_gb(mem.total)} total, {human_gb(mem.available)} available")
        info["ram"] = mem.total
    except ImportError:
        pass
    usage = shutil.disk_usage(".")
    print(f"disk        : {human_gb(usage.free)} free of {human_gb(usage.total)}")
    info["disk_free"] = usage.free

    print("\nrecommendation:")
    if gpu_ok:
        print("  GPU available: run inference with --device cuda --dtype float16 "
              "(verify output quality; fall back to float32 if unstable).")
    else:
        print("  CPU only: the shipped 17M-parameter model decodes at roughly "
              "30-70 tokens/s on 2 cores and much faster on a modern desktop CPU.")
    info["json"] = True
    print("\n" + json.dumps({k: (str(v) if not isinstance(v, (int, float, str)) else v)
                             for k, v in info.items() if k != "json"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
