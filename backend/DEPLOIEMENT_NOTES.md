# Notes de déploiement — analyse audio

## Dépendances et modèle

Depuis le dossier `backend`, installe les dépendances :

```bash
pip install -r requirements.txt
```

Le service utilise `faster-whisper` avec un modèle local CTranslate2. Convertis le
checkpoint Hugging Face une seule fois, en environnement de développement :

```bash
pip install ctranslate2 "transformers[torch]"
ct2-transformers-converter \
    --model /chemin/vers/whisper-dental-fr-final \
    --output_dir /var/consultpro/models/whisper-dental-fr-final-ct2 \
    --quantization int8
```

Ne télécharge pas le modèle depuis un worker à chaque démarrage; déploie-le localement
et configure son chemin.

## Configuration et processus

Configure les variables suivantes via systemd et un gestionnaire de secrets :

```bash
CONSULTPRO_ENVIRONMENT=production
CONSULTPRO_DATABASE_URL=postgresql://consultpro:URL_ENCODED_PASSWORD@127.0.0.1:5432/consultpro
CONSULTPRO_DB_SSLMODE=require
CONSULTPRO_CORS_ORIGINS=https://consult-pro-praticien.vercel.app
CONSULTPRO_ALLOWED_HOSTS=api.example.com
CONSULTPRO_ENABLE_HSTS=true
CONSULTPRO_MAX_UPLOAD_BYTES=26214400
CONSULTPRO_WHISPER_MODEL_PATH=/var/consultpro/models/whisper-dental-fr-final-ct2
CONSULTPRO_ASR_DEVICE=cuda
CONSULTPRO_ASR_COMPUTE_TYPE=float16
CONSULTPRO_ASR_CPU_THREADS=2
CONSULTPRO_ASR_BEAM_SIZE=1
CONSULTPRO_UPLOAD_DIR=/var/consultpro/uploads
```

Lance deux services avec ces mêmes variables, depuis le dossier `backend` :

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
python -m app.worker
```

L'API web ne charge pas le modèle. Le worker dédié le charge une fois et consomme les
jobs persistés dans PostgreSQL (`FOR UPDATE SKIP LOCKED`), avec heartbeat pour détecter
un worker arrêté. Commence avec un worker par GPU pour éviter plusieurs copies du
modèle en VRAM. `CONSULTPRO_ASR_DEVICE=cuda` tente CUDA puis retombe sur CPU `int8`
si CUDA échoue au chargement du modèle ou pendant l'inférence. Si GPU/driver/CUDA ne
sont pas compatibles, tu peux aussi forcer `cpu` et `int8`.

## Limites et sécurité opérationnelle

- L'upload audio est limité à 25 Mio; configure le reverse proxy pour environ 27 Mio
  afin d'inclure le multipart. Termine HTTPS au proxy et limite le débit sur login,
  inscription et upload (voir la configuration Nginx dans le README racine).
- Le compte PostgreSQL doit être dédié, non-superutilisateur; ne publie pas le port
  5432 sur Internet.
- Configure les véritables domaines CORS et hôtes autorisés, sans joker. Le processus
  de production refuse HSTS désactivé et les origines HTTP.
- Chiffre les disques et sauvegardes, définis la conservation des rapports/transcriptions,
  limite l'accès aux uploads, et teste régulièrement les restaurations.

## Mesure de performance

Les logs du worker séparent le chargement audio, la diarisation/clustering, la
transcription Whisper et le post-traitement. Pour comparer le temps d'inférence CPU et
GPU, utilise un fichier de test autorisé et non sensible :

```bash
python ../scripts/benchmark_asr.py \
  --audio /chemin/vers/audio-test.wav \
  --model /var/consultpro/models/whisper-dental-fr-final-ct2 \
  --devices cuda,cpu
```

Le benchmark indique le facteur temps réel (durée inférence / durée audio) et exclut
le chargement du modèle. Fais plusieurs exécutions identiques et contrôle également la
qualité des transcriptions avant de modifier le beam size.
