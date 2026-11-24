"""
diarization_service.py
-----------------------
Portage du notebook `diarisation_hybrid_versionF.ipynb` en module Python
appelable depuis le backend FastAPI (au lieu d'un notebook Colab).

Différences par rapport au notebook :
- Plus de Google Drive / Colab : chemins passés en paramètres.
- Transcription via `faster-whisper` (CTranslate2) plutôt que `transformers` :
  bien plus rapide sur CPU, ce qui est le cas de figure sur un VPS Hostinger
  KVM (pas de GPU). Voir load_asr_model() plus bas.
- Le modèle est chargé UNE SEULE FOIS au démarrage du serveur (voir main.py
  -> lifespan), pas à chaque requête.
- Retourne un dict Python (équivalent de report.json) au lieu d'écrire des
  fichiers de rapport.

MODIF (intégration coaching praticien — phonétique/diction) :
- `_compute_segment_embedding` retourne maintenant un tuple
  (embedding, features) au lieu d'un simple vecteur. `features` contient
  pitch_std et energy_mean, déjà calculés pour le clustering mais jusque-là
  jetés après usage.
- Chaque segment du report final porte désormais :
    - "pitch_std"             : variation du ton (voix monotone / tremblante)
    - "energy_mean"           : intensité vocale moyenne (log-énergie)
    - "debit_mots_par_sec"    : débit de parole, calculé après transcription
  Ces trois champs alimentent coaching_service.py pour couvrir l'axe
  "phonétique et diction" du cahier des charges (débit, articulation
  indirecte via variation de ton), qui n'était pas exposé auparavant.
- `_merge_consecutive` moyenne pitch_std/energy_mean au prorata de la durée
  quand deux segments consécutifs du même locuteur sont fusionnés, au lieu
  de garder arbitrairement les valeurs du premier segment.

IMPORTANT — conversion du modèle :
`whisper-dental-fr-final` est un modèle HuggingFace fine-tuné. Pour l'utiliser
avec faster-whisper, il doit d'abord être converti au format CTranslate2 :

    pip install ctranslate2 transformers[torch]
    ct2-transformers-converter \\
        --model /chemin/vers/whisper-dental-fr-final \\
        --output_dir /chemin/vers/whisper-dental-fr-final-ct2 \\
        --quantization int8

Cette conversion se fait UNE FOIS (en local ou sur le VPS), pas à chaque
déploiement. C'est `whisper-dental-fr-final-ct2` qu'on charge ensuite ici.
"""

import os
import re
import uuid
import warnings

import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Paramètres (identiques au notebook)
# ─────────────────────────────────────────────────────────────────────
TARGET_SR = 16000
SILENCE_THRESHOLD_DB = -40
MIN_SILENCE_MS = 400
MIN_SEGMENT_MS = 800
N_MFCC = 40
N_MELS = 80
N_SPEAKERS = 2
MERGE_GAP_MS = 1500

KEYWORDS_DOCTOR = [
    r"\border\w*nance\b", r"\bprescrire\b", r"\bprescris\b", r"\bprescription\b",
    r"\bdiagnostic\b", r"\btraitement\b", r"\bexamen\b", r"\bradiographie\b",
    r"\banalyse\b", r"\bdosage\b", r"\bantibiotique\b", r"\bconsultation\b",
    r"\btenez\b", r"\brespirer\b", r"\bauscult\w+\b", r"\bpression artérielle\b",
    r"\bje vous (conseille|recommande|prescris|propose)\b",
    r"\bvous devez\b", r"\bfaut (faire|prendre|consulter)\b",
]
KEYWORDS_PATIENT = [
    r"\bj'ai mal\b", r"\bça fait mal\b", r"\bdepuis (hier|ce matin|lundi|une semaine)\b",
    r"\bj'ai (de la fièvre|de la toux|du mal|des douleurs|des nausées)\b",
    r"\bje me sens\b", r"\bje souffre\b", r"\bje dors mal\b",
    r"\bmon (médecin|docteur|généraliste)\b",
    r"\bc'est (grave|normal|inquiétant)\b",
    r"\bqu'est-ce que j'ai\b",
]


# ─────────────────────────────────────────────────────────────────────
# Chargement du modèle Whisper (une seule fois, au démarrage du serveur)
# ─────────────────────────────────────────────────────────────────────
_asr_model = None
_asr_model_path: str | None = None
_asr_device: str | None = None
_asr_cpu_threads = 2
ASR_BEAM_SIZE = int(os.environ.get("CONSULTPRO_ASR_BEAM_SIZE", "1"))


def load_asr_model(model_path: str, device: str = "cuda", compute_type: str = "float16", cpu_threads: int = 0):
    """
    Charge le modèle Whisper en priorité sur GPU, avec fallback CPU si nécessaire.
    Cela évite de rester bloqué sur un environnement sans CUDA alors que le PC
    a bien une carte graphique disponible.
    """
    global _asr_model, _asr_model_path, _asr_device, _asr_cpu_threads
    if _asr_model is not None:
        return _asr_model

    from faster_whisper import WhisperModel

    if device in {"", "auto"}:
        candidate_devices = ["cuda", "cpu"]
    elif device == "cuda":
        candidate_devices = ["cuda", "cpu"]
    elif device == "cpu":
        candidate_devices = ["cpu"]
    else:
        raise ValueError(f"Device ASR non supporté : {device}")

    last_error = None
    for candidate in candidate_devices:
        try:
            candidate_compute_type = compute_type
            if candidate == "cpu" and compute_type not in {
                "int8", "int8_float32", "float32"
            }:
                candidate_compute_type = "int8"
            _asr_model = WhisperModel(
                model_path,
                device=candidate,
                compute_type=candidate_compute_type,
                cpu_threads=cpu_threads if candidate == "cpu" else 0,
            )
            _asr_model_path = model_path
            _asr_device = candidate
            _asr_cpu_threads = cpu_threads
            print(
                f"WhisperModel loaded on device={candidate} "
                f"(compute_type={candidate_compute_type})",
                flush=True,
            )
            return _asr_model
        except Exception as exc:  # pragma: no cover - dépend de l'environnement GPU/CPU
            last_error = exc
            print(f"Whisper loading on {candidate} failed: {exc}", flush=True)

    raise RuntimeError(
        f"Impossible de charger le modèle Whisper sur {candidate_devices}."
    ) from last_error


def _transcribe(wav_path: str) -> str:
    if _asr_model is None:
        raise RuntimeError("Modèle ASR non chargé — appelle load_asr_model() au démarrage.")
    segments, _info = _asr_model.transcribe(
        wav_path, language="fr", beam_size=ASR_BEAM_SIZE, vad_filter=True
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def _transcribe_full_audio(audio_path: str) -> list[dict]:
    """Transcrit tout le fichier une seule fois et renvoie les segments de temps."""
    try:
        return _transcribe_full_audio_with_loaded_model(audio_path)
    except Exception as exc:
        message = str(exc).lower()
        cuda_runtime_failure = any(
            marker in message
            for marker in ("cuda", "cublas", "cudnn", "nvrtc")
        )
        if _asr_device != "cuda" or not cuda_runtime_failure:
            raise

        print(
            f"Whisper inference failed on CUDA ({exc}); retrying this job on CPU int8.",
            flush=True,
        )
        global _asr_model
        _asr_model = None
        load_asr_model(
            _asr_model_path or "",
            device="cpu",
            compute_type="int8",
            cpu_threads=_asr_cpu_threads,
        )
        return _transcribe_full_audio_with_loaded_model(audio_path)


def _transcribe_full_audio_with_loaded_model(audio_path: str) -> list[dict]:
    if _asr_model is None:
        raise RuntimeError("Modèle ASR non chargé — appelle load_asr_model() au démarrage.")

    segments, _info = _asr_model.transcribe(
        audio_path,
        language="fr",
        beam_size=ASR_BEAM_SIZE,
        vad_filter=True,
        word_timestamps=False,
    )

    result = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        result.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text,
        })
    return result


# ─────────────────────────────────────────────────────────────────────
# Segmentation par silence
# ─────────────────────────────────────────────────────────────────────
def _segment_by_silence(y, sr, threshold_db, min_silence_ms, min_segment_ms):
    hop_ms = 20
    hop_len = int(sr * hop_ms / 1000)
    frame_len = hop_len * 2

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    is_speech = rms_db > threshold_db

    min_sil_frames = max(1, int(min_silence_ms / hop_ms))
    is_speech = median_filter(is_speech.astype(float), size=min_sil_frames) > 0.5

    segments, in_seg, seg_start = [], False, 0
    for i, speech in enumerate(is_speech):
        if speech and not in_seg:
            seg_start, in_seg = i, True
        elif not speech and in_seg:
            start_s, end_s = seg_start * hop_len, i * hop_len
            if (end_s - start_s) / sr * 1000 >= min_segment_ms:
                segments.append((start_s, min(end_s, len(y))))
            in_seg = False
    if in_seg:
        start_s, end_s = seg_start * hop_len, len(y)
        if (end_s - start_s) / sr * 1000 >= min_segment_ms:
            segments.append((start_s, end_s))
    return segments


def _compute_segment_embedding(y_seg, sr, n_mfcc=40, n_mels=80):
    """
    Retourne (embedding, features) :
    - embedding : vecteur utilisé pour le clustering (inchangé)
    - features  : dict avec pitch_std et energy_mean, exposés pour le
                  coaching praticien (axe phonétique/diction).
      Retourne (None, None) si le segment est trop court.
    """
    hop = 160
    if len(y_seg) < hop * 4:
        return None, None

    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop, n_fft=512)
    d1 = librosa.feature.delta(mfcc)

    # librosa.pyin est coûteux lorsqu'il est exécuté sur chaque segment.
    # librosa.yin est plus léger et suffit ici pour une mesure indicative du pitch.
    try:
        f0 = librosa.yin(
            y_seg,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=512,
            hop_length=hop,
        )
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0) > 1:
            f0_mean = float(np.mean(f0))
            f0_std = float(np.std(f0))
            f0_med = float(np.median(f0))
            voiced_ratio = float(len(f0) / max(1, int(len(y_seg) / hop)))
        elif len(f0) == 1:
            f0_mean = f0_med = float(f0[0])
            f0_std = 0.0
            voiced_ratio = 1.0
        else:
            f0_mean = f0_std = f0_med = voiced_ratio = 0.0
    except Exception as exc:
        print(f"⚠️ [DIARIZATION] Pitch indisponible : {exc}", flush=True)
        f0_mean = f0_std = f0_med = voiced_ratio = 0.0

    rms = librosa.feature.rms(y=y_seg, hop_length=hop)[0]
    log_e = np.log(rms + 1e-10)

    embedding = np.concatenate([
        mfcc.mean(1), mfcc.std(1),
        d1.mean(1), d1.std(1),
        [f0_mean, f0_std, f0_med, voiced_ratio],
        [log_e.mean(), log_e.std()],
    ])

    features = {
        "pitch_std": float(f0_std),
        "energy_mean": float(log_e.mean()),
    }

    return embedding, features


def _cluster_embeddings(embs, n_spk):
    """Clusterise les segments en évitant PCA/clustering sur trop peu d'échantillons."""
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2 or len(embs) == 0:
        raise ValueError("Aucun embedding vocal valide pour la diarisation.")
    if len(embs) == 1:
        return np.zeros(1, dtype=int)

    n_clusters = min(int(n_spk), len(embs))
    X = StandardScaler().fit_transform(embs)
    n_comp = max(1, min(30, X.shape[1], X.shape[0] - 1))
    if X.shape[1] > n_comp and X.shape[0] > 2:
        X = PCA(n_components=n_comp, random_state=0).fit_transform(X)
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)
    return labels


def _merge_consecutive(segments, merge_gap_ms):
    """
    Fusionne les segments consécutifs du même locuteur. Pour pitch_std et
    energy_mean, moyenne pondérée par la durée plutôt que de garder
    arbitrairement les valeurs du premier segment de la fusion.
    """
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if prev["speaker_id"] == seg["speaker_id"] and seg["start_ms"] - prev["end_ms"] <= merge_gap_ms:
            prev_dur = prev["duration_ms"]
            seg_dur = seg["duration_ms"]
            total_dur = prev_dur + seg_dur
            if total_dur > 0:
                for key in ("pitch_std", "energy_mean"):
                    if key in prev and key in seg:
                        prev[key] = (prev[key] * prev_dur + seg[key] * seg_dur) / total_dur
            prev["end_ms"] = seg["end_ms"]
            prev["duration_ms"] = prev["end_ms"] - prev["start_ms"]
        else:
            merged.append(seg.copy())
    return merged


def _keyword_score(text, patterns):
    text_low = text.lower()
    return sum(1 for p in patterns if re.search(p, text_low))


def _question_score(text):
    return text.count("?") + len(re.findall(
        r"\b(depuis quand|où avez-vous|est-ce que vous|avez-vous|comment allez|"
        r"avez-vous pris|est-ce que ça|qu'est-ce qui|pouvez-vous)\b",
        text.lower(),
    ))


def _identify_roles(segments):
    scores = {"SPEAKER_0": {"doctor": 0, "patient": 0},
              "SPEAKER_1": {"doctor": 0, "patient": 0}}
    for seg in segments:
        spk, text = seg["speaker_id"], seg.get("text", "")
        if not text.strip():
            continue
        scores[spk]["doctor"] += _keyword_score(text, KEYWORDS_DOCTOR)
        scores[spk]["patient"] += _keyword_score(text, KEYWORDS_PATIENT)
        scores[spk]["doctor"] += _question_score(text) * 0.5

    for spk in ["SPEAKER_0", "SPEAKER_1"]:
        ss = [s for s in segments if s["speaker_id"] == spk]
        avg = np.mean([s["duration_ms"] for s in ss]) if ss else 0
        scores[spk]["doctor"] += avg / 5000

    net_0 = scores["SPEAKER_0"]["doctor"] - scores["SPEAKER_0"]["patient"]
    net_1 = scores["SPEAKER_1"]["doctor"] - scores["SPEAKER_1"]["patient"]

    if net_0 >= net_1:
        role_map = {"SPEAKER_0": "Médecin", "SPEAKER_1": "Patient"}
    else:
        role_map = {"SPEAKER_0": "Patient", "SPEAKER_1": "Médecin"}
    return role_map


# ─────────────────────────────────────────────────────────────────────
# Fonction principale — équivalent de tout le notebook 1
# ─────────────────────────────────────────────────────────────────────
def run_diarization_pipeline(
    audio_path: str,
    work_dir: str | None = None,
    timings: list[tuple[str, float]] | None = None,
) -> dict:
    """Exécute la diarisation et la transcription, avec logs de progression."""
    import time

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(os.path.abspath(audio_path)), "consultpro_work")
    os.makedirs(work_dir, exist_ok=True)

    started = time.perf_counter()
    print(f"🎧 [DIARIZATION] Début du pipeline : {audio_path}", flush=True)

    stage_started = time.perf_counter()
    print("🎧 [DIARIZATION] Chargement audio...", flush=True)
    audio_data, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    if audio_data.size == 0:
        raise ValueError("Le fichier audio est vide ou illisible.")
    duration = float(librosa.get_duration(y=audio_data, sr=sr))
    print(f"✅ [DIARIZATION] Audio chargé : {duration:.1f}s, SR={sr}", flush=True)
    if timings is not None:
        timings.append(("Audio loading", time.perf_counter() - stage_started))

    stage_started = time.perf_counter()
    print("🔊 [DIARIZATION] Détection des zones de parole...", flush=True)
    speech_segments = _segment_by_silence(
        audio_data, sr, SILENCE_THRESHOLD_DB, MIN_SILENCE_MS, MIN_SEGMENT_MS
    )
    print(f"✅ [DIARIZATION] {len(speech_segments)} zones détectées", flush=True)
    if len(speech_segments) == 0:
        raise ValueError("Aucune parole détectée. Vérifiez le volume et le fichier audio.")

    embeddings, valid_segs, seg_features = [], [], []
    total = len(speech_segments)
    print("🧠 [DIARIZATION] Calcul des embeddings (pitch simplifié)...", flush=True)
    for i, (start_s, end_s) in enumerate(speech_segments, start=1):
        emb, feats = _compute_segment_embedding(
            audio_data[start_s:end_s], sr, N_MFCC, N_MELS
        )
        if emb is not None:
            embeddings.append(emb)
            valid_segs.append((start_s, end_s))
            seg_features.append(feats)
        if i == 1 or i % 5 == 0 or i == total:
            print(f"🧠 [DIARIZATION] Embedding {i}/{total}", flush=True)

    if not embeddings:
        raise ValueError("Les zones détectées sont trop courtes pour analyser les voix.")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"✅ [DIARIZATION] {len(embeddings)} embeddings calculés", flush=True)

    print("🧩 [DIARIZATION] Clustering des locuteurs...", flush=True)
    if len(embeddings) >= 2:
        seg_labels = _cluster_embeddings(embeddings, N_SPEAKERS)
    else:
        seg_labels = np.zeros(len(embeddings), dtype=int)
    print("✅ [DIARIZATION] Clustering terminé", flush=True)
    if timings is not None:
        timings.append(("Speaker diarization and clustering", time.perf_counter() - stage_started))

    raw_segs = []
    for lbl, (start_s, end_s), feats in zip(seg_labels, valid_segs, seg_features):
        start_ms = int(start_s / sr * 1000)
        end_ms = int(end_s / sr * 1000)
        raw_segs.append({
            "speaker_id": f"SPEAKER_{int(lbl)}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": max(1, end_ms - start_ms),
            "pitch_std": float(feats.get("pitch_std", 0.0)),
            "energy_mean": float(feats.get("energy_mean", 0.0)),
            "text": "",
        })
    segments = _merge_consecutive(raw_segs, MERGE_GAP_MS)

    # Transcrire le fichier complet une seule fois. Faster-whisper renvoie un
    # générateur : le calcul réel se produit pendant l'itération ci-dessous.
    print("🎙️ [WHISPER] Début de transcription globale...", flush=True)
    transcription_started = time.perf_counter()
    full_transcript = _transcribe_full_audio(audio_path)
    print(
        f"✅ [WHISPER] Transcription terminée : {len(full_transcript)} morceaux "
        f"en {time.perf_counter() - transcription_started:.1f}s",
        flush=True,
    )
    if timings is not None:
        timings.append(("Whisper transcription", time.perf_counter() - transcription_started))
    if not full_transcript:
        raise RuntimeError(
            "Whisper n'a retourné aucun texte. Vérifiez l'audio, le modèle et les logs du backend."
        )

    stage_started = time.perf_counter()
    # Affecter chaque morceau transcrit au segment de parole avec le plus grand
    # chevauchement temporel, pour éviter de dupliquer le texte entre locuteurs.
    for chunk in full_transcript:
        best_seg = None
        best_overlap = 0.0
        chunk_start = float(chunk["start"])
        chunk_end = float(chunk["end"])
        for seg in segments:
            seg_start = seg["start_ms"] / 1000.0
            seg_end = seg["end_ms"] / 1000.0
            overlap = max(0.0, min(seg_end, chunk_end) - max(seg_start, chunk_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_seg = seg
        if best_seg is not None:
            best_seg["text"] = (best_seg.get("text", "") + " " + chunk["text"]).strip()

    for seg in segments:
        text = seg.get("text", "").strip()
        duration_s = max(seg["duration_ms"] / 1000.0, 0.001)
        seg["debit_mots_par_sec"] = round(len(text.split()) / duration_s, 2)

    if any(seg.get("text", "").strip() for seg in segments) and len({s["speaker_id"] for s in segments}) > 1:
        role_map = _identify_roles(segments)
    else:
        role_map = {"SPEAKER_0": "Médecin", "SPEAKER_1": "Patient"}

    for seg in segments:
        seg["speaker"] = role_map.get(seg["speaker_id"], seg["speaker_id"])
    if timings is not None:
        timings.append(("Transcript alignment and postprocessing", time.perf_counter() - stage_started))

    result = {
        "file": os.path.basename(audio_path),
        "duration_s": duration,
        "n_speakers": len({s["speaker_id"] for s in segments}),
        "n_segments": len(segments),
        "role_map": role_map,
        "segments": segments,
    }
    print(
        f"🏁 [DIARIZATION] Pipeline terminé en {time.perf_counter() - started:.1f}s "
        f"({len(segments)} segments)",
        flush=True,
    )
    return result
