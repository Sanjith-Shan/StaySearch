"""Every measurement is one JSON line in results/*.jsonl, stamped with the machine and its load."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess

from staysearch.config import RESULTS


def machine() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(), "cores": os.cpu_count()}
    if platform.system() == "Darwin":
        q = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        info["cpu"] = q("machdep.cpu.brand_string")
        info["ram_gb"] = round(int(q("hw.memsize")) / 2**30, 1)
    else:
        info["cpu"] = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    return info


def load() -> dict:
    try:
        la = os.getloadavg()
        return {"loadavg_1m": round(la[0], 2), "loadavg_5m": round(la[1], 2)}
    except (OSError, AttributeError):
        try:
            import psutil  # Windows has no loadavg
            return {"cpu_percent": psutil.cpu_percent(interval=0.5)}
        except ImportError:
            return {}


def append(name: str, row: dict) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), **row,
           "machine": machine(), "load": load()}
    # A second machine writing through the file mirror uses its own file to avoid sync conflicts.
    suffix = os.environ.get("STAYSEARCH_RESULTS_SUFFIX")
    fname = f"{name}.{suffix}.jsonl" if suffix else f"{name}.jsonl"
    with open(RESULTS / fname, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return rec


def reset(name: str) -> None:
    p = RESULTS / f"{name}.jsonl"
    if p.exists():
        p.unlink()


def require_disk(min_gb: float = 3.0) -> None:
    """Refuse to start heavy work on a nearly full disk (the build machine is shared and small)."""
    import shutil
    from pathlib import Path
    free = shutil.disk_usage(Path.home()).free / 1e9
    if free < min_gb:
        raise SystemExit(f"stop: {free:.1f} GB free < {min_gb} GB; run this step on the mini PC instead")
