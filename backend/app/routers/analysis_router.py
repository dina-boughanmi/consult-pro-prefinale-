import os
import stat
import uuid
from datetime import datetime
from pathlib import Path
from enum import Enum

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.routers.auth_router import require_user
from app.services.database import (
    create_analysis,
    get_analysis_for_user,
    list_analyses_for_user,
)


router = APIRouter(prefix="/api/analysis", tags=["analysis"])

DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"
UPLOAD_DIR = Path(os.environ.get("CONSULTPRO_UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR))).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
if os.name != "nt":
    os.chmod(UPLOAD_DIR, stat.S_IRWXU)

MAX_UPLOAD_BYTES = int(os.environ.get("CONSULTPRO_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024
if MAX_UPLOAD_BYTES <= 0:
    raise RuntimeError("CONSULTPRO_MAX_UPLOAD_BYTES doit être supérieur à zéro.")


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: str | None = None
    result: dict | None = None
    error: str | None = None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _matches_audio_signature(extension: str, header: bytes) -> bool:
    if extension == ".wav":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if extension == ".ogg":
        return header.startswith(b"OggS")
    if extension in {".mp4", ".m4a"}:
        return len(header) >= 8 and header[4:8] == b"ftyp"
    if extension == ".mp3":
        return header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        )
    return False


@router.post("/upload", response_model=JobStatusResponse)
def upload_consultation_audio(
    file: UploadFile = File(...),
    user: dict = Depends(require_user),
):
    allowed_extensions = {".wav", ".mp3", ".mp4", ".m4a", ".ogg"}
    filename = os.path.basename(file.filename or "")
    extension = Path(filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            400,
            f"Format non supporté. Formats acceptés : {', '.join(sorted(allowed_extensions))}",
        )
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Le fichier dépasse la limite de {MAX_UPLOAD_BYTES} octets.")

    job_id = uuid.uuid4().hex
    audio_path = Path(UPLOAD_DIR) / f"{job_id}{extension}"
    written = 0
    try:
        with audio_path.open("xb") as destination:
            if os.name != "nt":
                os.chmod(audio_path, stat.S_IRUSR | stat.S_IWUSR)
            while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
                if written == 0 and not _matches_audio_signature(extension, chunk[:16]):
                    raise HTTPException(400, "Le contenu du fichier ne correspond pas à un audio accepté.")
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"Le fichier dépasse la limite de {MAX_UPLOAD_BYTES} octets."
                    )
                destination.write(chunk)
        if written == 0:
            raise HTTPException(400, "Le fichier audio est vide.")
    except Exception:
        audio_path.unlink(missing_ok=True)
        raise

    try:
        create_analysis(job_id, user["id"], _now(), str(audio_path))
    except Exception:
        audio_path.unlink(missing_ok=True)
        raise

    return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)


@router.get("/history")
def get_analysis_history(user: dict = Depends(require_user)):
    return list_analyses_for_user(user["id"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_analysis_status(job_id: str, user: dict = Depends(require_user)):
    stored = get_analysis_for_user(job_id, user["id"])
    if stored is None:
        raise HTTPException(404, "Analyse introuvable.")
    return JobStatusResponse(
        job_id=job_id,
        status=stored["status"],
        progress=stored.get("progress"),
        result=stored["result"],
        error=stored["error"],
    )
