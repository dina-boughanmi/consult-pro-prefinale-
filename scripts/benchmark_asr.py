import argparse
import gc
import time
from pathlib import Path

from faster_whisper import WhisperModel


def benchmark(audio_path: Path, model_path: str, device: str, compute_type: str, beam_size: int) -> None:
    model = WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=0 if device == "cuda" else 2,
    )
    started = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language="fr",
        beam_size=beam_size,
        vad_filter=True,
    )
    segment_count = sum(1 for _ in segments)
    elapsed = time.perf_counter() - started
    duration = info.duration
    print(
        f"device={device} compute={compute_type} beam={beam_size} "
        f"audio={duration:.1f}s inference={elapsed:.2f}s "
        f"real_time_factor={elapsed / duration:.3f} segments={segment_count}"
    )
    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare faster-whisper inference on the same local, non-sensitive audio file."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local CTranslate2 model directory")
    parser.add_argument("--devices", default="cuda,cpu", help="Comma-separated devices to compare")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--cpu-compute-type", default="int8")
    parser.add_argument("--gpu-compute-type", default="float16")
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.beam_size < 1:
        parser.error("--beam-size must be at least 1")
    if not Path(args.model).is_dir():
        parser.error("--model must point to a local CTranslate2 model directory")

    for device in (value.strip() for value in args.devices.split(",")):
        if device not in {"cpu", "cuda"}:
            parser.error(f"Unsupported device: {device}")
        compute_type = args.gpu_compute_type if device == "cuda" else args.cpu_compute_type
        benchmark(args.audio, args.model, device, compute_type, args.beam_size)


if __name__ == "__main__":
    main()
