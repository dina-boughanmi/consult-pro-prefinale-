import io
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np
from starlette.datastructures import UploadFile

from app.routers import analysis_router
from app.services import diarization_service
from app import worker


class UploadSecurityTests(unittest.TestCase):
    def test_audio_signatures_match_expected_containers(self):
        examples = {
            ".wav": b"RIFF1234WAVE",
            ".ogg": b"OggS" + b"\x00" * 8,
            ".mp3": b"ID3" + b"\x00" * 8,
            ".mp4": b"\x00\x00\x00\x18ftypisom",
            ".m4a": b"\x00\x00\x00\x18ftypisom",
        }
        for extension, header in examples.items():
            with self.subTest(extension=extension):
                self.assertTrue(analysis_router._matches_audio_signature(extension, header))
        self.assertFalse(analysis_router._matches_audio_signature(".wav", b"not an audio file"))

    def test_upload_at_limit_is_accepted(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            with (
                patch.object(analysis_router, "UPLOAD_DIR", upload_dir),
                patch.object(analysis_router, "MAX_UPLOAD_BYTES", 12),
                patch.object(analysis_router, "create_analysis"),
            ):
                upload = UploadFile(filename="sample.wav", file=io.BytesIO(b"RIFF1234WAVE"))
                response = analysis_router.upload_consultation_audio(
                    upload, {"id": "test-user"}
                )
                saved_path = Path(upload_dir, f"{response.job_id}.wav")
                self.assertTrue(saved_path.is_file())
                saved_path.unlink()
                self.assertEqual(response.status.value, "pending")

    def test_stream_over_limit_is_rejected_and_partial_file_removed(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            with (
                patch.object(analysis_router, "UPLOAD_DIR", upload_dir),
                patch.object(analysis_router, "MAX_UPLOAD_BYTES", 12),
                patch.object(analysis_router, "create_analysis"),
            ):
                upload = UploadFile(
                    filename="large.wav",
                    file=io.BytesIO(b"RIFF1234WAVE!"),
                    size=None,
                )
                with self.assertRaises(HTTPException) as raised:
                    analysis_router.upload_consultation_audio(
                        upload, {"id": "test-user"}
                    )
                self.assertEqual(raised.exception.status_code, 413)
                self.assertEqual(list(Path(upload_dir).iterdir()), [])

    def test_cuda_load_failure_falls_back_to_cpu_int8(self):
        attempts = []

        class FakeWhisperModel:
            def __init__(self, _model_path, device, compute_type, cpu_threads):
                attempts.append((device, compute_type, cpu_threads))
                if device == "cuda":
                    raise RuntimeError("CUDA unavailable")

        diarization_service._asr_model = None
        try:
            with patch.dict(
                sys.modules,
                {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
            ):
                diarization_service.load_asr_model(
                    "local-model",
                    device="cuda",
                    compute_type="float16",
                    cpu_threads=2,
                )
            self.assertEqual(
                attempts,
                [("cuda", "float16", 0), ("cpu", "int8", 2)],
            )
        finally:
            diarization_service._asr_model = None

    def test_cuda_inference_failure_retries_on_cpu_int8(self):
        attempts = []

        class FakeWhisperModel:
            def __init__(self, _model_path, device, compute_type, cpu_threads):
                attempts.append((device, compute_type, cpu_threads))
                self.device = device

            def transcribe(self, *_args, **_kwargs):
                if self.device == "cuda":
                    def failed_segments():
                        raise RuntimeError("Library cublas64_12.dll is not found")
                        yield

                    return failed_segments(), None
                return iter([SimpleNamespace(start=0.0, end=1.0, text="Bonjour")]), None

        diarization_service._asr_model = None
        try:
            with patch.dict(
                sys.modules,
                {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
            ):
                diarization_service.load_asr_model(
                    "local-model",
                    device="cuda",
                    compute_type="float16",
                    cpu_threads=2,
                )
                result = diarization_service._transcribe_full_audio("test.wav")

            self.assertEqual(
                attempts,
                [("cuda", "float16", 0), ("cpu", "int8", 2)],
            )
            self.assertEqual(result[0]["text"], "Bonjour")
        finally:
            diarization_service._asr_model = None
            diarization_service._asr_model_path = None
            diarization_service._asr_device = None

    def test_worker_processes_and_removes_queued_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir, "queued.wav")
            audio_path.write_bytes(b"audio")
            report = {"duration_s": 12.0}
            coaching = {"svs": {"global": 80}}
            progress_updates = []
            saved_results = []

            def timed_step(_label, operation, timings):
                result = operation()
                timings.append(("mock step", 0.01))
                return result

            with (
                patch.object(worker, "run_timed_step", side_effect=timed_step),
                patch.object(worker, "convert_to_wav_16k_mono", return_value=str(audio_path)),
                patch.object(worker, "run_diarization_pipeline", return_value=report),
                patch.object(worker, "generate_coaching", return_value=coaching),
                patch.object(
                    worker,
                    "update_analysis_progress",
                    side_effect=lambda *args: progress_updates.append(args),
                ),
                patch.object(
                    worker,
                    "save_analysis_result",
                    side_effect=lambda *args: saved_results.append(args),
                ),
                patch.object(worker, "heartbeat_analysis"),
            ):
                worker._process_analysis({"id": "job-id", "audio_path": str(audio_path)})

            self.assertFalse(audio_path.exists())
            self.assertEqual(len(progress_updates), 3)
            self.assertEqual(len(saved_results), 1)

    def test_pipeline_reports_individual_stage_timings(self):
        timings = []
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir, "test.wav")
            audio_path.write_bytes(b"test")
            with (
                patch.object(
                    diarization_service.librosa,
                    "load",
                    return_value=(np.ones(200, dtype=np.float32), 16000),
                ),
                patch.object(diarization_service.librosa, "get_duration", return_value=1.0),
                patch.object(diarization_service, "_segment_by_silence", return_value=[(0, 200)]),
                patch.object(
                    diarization_service,
                    "_compute_segment_embedding",
                    return_value=([1.0], {"pitch_std": 0.1, "energy_mean": 0.2}),
                ),
                patch.object(
                    diarization_service,
                    "_transcribe_full_audio",
                    return_value=[{"start": 0.0, "end": 0.01, "text": "Bonjour"}],
                ),
            ):
                diarization_service.run_diarization_pipeline(
                    str(audio_path),
                    timings=timings,
                )

        labels = {label for label, _elapsed in timings}
        self.assertIn("Audio loading", labels)
        self.assertIn("Speaker diarization and clustering", labels)
        self.assertIn("Whisper transcription", labels)
        self.assertIn("Transcript alignment and postprocessing", labels)


if __name__ == "__main__":
    unittest.main()
