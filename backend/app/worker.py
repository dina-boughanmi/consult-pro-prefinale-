import logging
import os
import threading
import time

from app.services.analysis_processing import (
    convert_to_wav_16k_mono,
    print_timing_summary,
    run_timed_step,
)
from app.services.database import (
    claim_next_analysis,
    heartbeat_analysis,
    initialize_database,
    save_analysis_result,
    update_analysis_progress,
    update_analysis_status,
)
from app.services.diarization_service import load_asr_model, run_diarization_pipeline
from app.services.coaching_service import generate_coaching


logger = logging.getLogger("consultpro.worker")
POLL_INTERVAL_SECONDS = float(os.environ.get("CONSULTPRO_WORKER_POLL_SECONDS", "2"))
HEARTBEAT_INTERVAL_SECONDS = 30


def _process_analysis(job: dict) -> None:
    job_id = job["id"]
    audio_path = job["audio_path"]
    converted_path: str | None = None
    timings: list[tuple[str, float]] = []
    job_started = time.perf_counter()
    heartbeat_stop = threading.Event()

    def heartbeat_loop() -> None:
        while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            try:
                heartbeat_analysis(job_id)
            except Exception:
                logger.exception("Could not update worker heartbeat for analysis %s", job_id)

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    try:
        update_analysis_progress(job_id, "Préparation de l'audio...")
        converted_path = run_timed_step(
            "Audio conversion",
            lambda: convert_to_wav_16k_mono(audio_path),
            timings,
        )
        if converted_path is None:
            raise RuntimeError("Audio conversion returned no processing path.")

        update_analysis_progress(job_id, "Diarisation et transcription en cours...")
        report = run_timed_step(
            "Diarization and transcription",
            lambda: run_diarization_pipeline(converted_path, timings=timings),
            timings,
        )

        update_analysis_progress(job_id, "Calcul du coaching...")
        coaching = run_timed_step(
            "Coaching generation",
            lambda: generate_coaching(report),
            timings,
        )
        save_analysis_result(
            job_id,
            {"report": report, "coaching": coaching},
            report.get("duration_s"),
            coaching.get("svs", {}).get("global"),
        )
    except Exception:
        logger.exception("Analysis job %s failed", job_id)
        update_analysis_status(
            job_id,
            "error",
            "Processing failed",
            progress=None,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL_SECONDS + 1)
        for path in {audio_path, converted_path}:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.exception("Could not remove temporary audio for analysis %s", job_id)
        print_timing_summary(job_id, timings, time.perf_counter() - job_started)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    initialize_database()
    model_path = os.environ.get(
        "CONSULTPRO_WHISPER_MODEL_PATH",
        "base",
    )
    if (
        os.environ.get("CONSULTPRO_ENVIRONMENT", "development").lower() == "production"
        and not os.path.isdir(model_path)
    ):
        raise RuntimeError(
            "En production, CONSULTPRO_WHISPER_MODEL_PATH doit pointer vers un modèle local."
        )
    device = os.environ.get("CONSULTPRO_ASR_DEVICE", "cpu")
    compute_type = os.environ.get("CONSULTPRO_ASR_COMPUTE_TYPE", "int8")
    cpu_threads = int(os.environ.get("CONSULTPRO_ASR_CPU_THREADS", "2"))
    load_asr_model(model_path, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    logger.info("ConsultPro analysis worker ready (device=%s, compute_type=%s)", device, compute_type)

    while True:
        try:
            job = claim_next_analysis()
            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            _process_analysis(job)
        except KeyboardInterrupt:
            logger.info("Analysis worker stopping")
            return
        except Exception:
            logger.exception("Worker queue polling failed")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
