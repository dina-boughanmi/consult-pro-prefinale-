import os
import shutil
import subprocess
import time
from collections.abc import Callable


def run_timed_step(
    label: str,
    operation: Callable,
    timings: list[tuple[str, float]],
):
    started = time.perf_counter()
    print(f"[TIMER] {label} started", flush=True)
    try:
        result = operation()
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"[TIMER] {label} failed after {elapsed:.2f}s", flush=True)
        timings.append((f"{label} (failed)", elapsed))
        raise
    elapsed = time.perf_counter() - started
    print(f"[TIMER] {label} finished in {elapsed:.2f}s", flush=True)
    timings.append((label, elapsed))
    return result


def print_timing_summary(
    job_id: str,
    timings: list[tuple[str, float]],
    total_seconds: float,
) -> None:
    step_width = max([len("Step"), len("TOTAL"), *(len(label) for label, _ in timings)])
    time_width = max(len("Time (s)"), 10)
    border = f"+{'-' * (step_width + 2)}+{'-' * (time_width + 2)}+"

    print(f"\n[TIMING SUMMARY] Analysis job {job_id}", flush=True)
    print(border, flush=True)
    print(f"| {'Step':<{step_width}} | {'Time (s)':>{time_width}} |", flush=True)
    print(border, flush=True)
    for label, elapsed in timings:
        print(f"| {label:<{step_width}} | {elapsed:>{time_width}.2f} |", flush=True)
    print(border, flush=True)
    print(f"| {'TOTAL':<{step_width}} | {total_seconds:>{time_width}.2f} |", flush=True)
    print(border, flush=True)


def convert_to_wav_16k_mono(src_path: str) -> str:
    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg introuvable, conversion ignorée.", flush=True)
        return src_path

    dst_path = os.path.splitext(src_path)[0] + "_16k.wav"
    if os.path.abspath(dst_path) == os.path.abspath(src_path):
        dst_path = os.path.splitext(src_path)[0] + "_16k_converted.wav"

    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst_path,
    ]
    try:
        subprocess.run(command, check=True, timeout=600, capture_output=True)
        return dst_path
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[WARN] Conversion ffmpeg échouée ({error}), fichier d'origine utilisé.", flush=True)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return src_path
