# ConsultPro — Guide de démarrage local

Plateforme d'analyse vocale IA et de coaching praticien pour les consultations dentaires.

## Prérequis

- **Python 3.11 (64-bit)** — installé et accessible via `py -3.11`
- **Node.js** (avec `npm`)
- **PostgreSQL 14+** — serveur accessible depuis le backend
- **PowerShell** (Windows)
- Le modèle Whisper fine-tuné converti au format CTranslate2 (`whisper-dental-fr-final-ct2`), disponible localement — voir [Modèle Whisper](#modèle-whisper) ci-dessous

## Structure du projet

```
consultpro/
├── backend/          # API FastAPI (analyse audio, transcription, scoring)
│   └── venv/         # environnement virtuel Python (créé automatiquement)
├── frontend/         # Interface React/Vite/Tailwind
├── model/            # Modèle Whisper fine-tuné (converti CTranslate2)
└── scripts/
    └── run_dev.ps1   # script de démarrage local (backend + frontend)
```

## Installation & démarrage

### 1. Cloner le projet

```powershell
git clone <url-du-repo> consultpro
cd consultpro
```

### 2. Modèle Whisper

Le backend a besoin du modèle Whisper fine-tuné au format **CTranslate2** (pas le format HuggingFace brut).

- Si tu as déjà le dossier converti (`whisper-dental-fr-final-ct2` contenant `config.json`, `model.bin`, `vocabulary.json`), place-le à l'emplacement suivant :
  ```
  C:\consultpro\model\whisper-dental-fr-final-ct2
  ```
- Si tu ne l'as pas, le script utilisera par défaut le modèle Whisper public `base` (moins précis, mais fonctionnel pour tester rapidement).

> Pour reconvertir le modèle depuis un checkpoint HuggingFace brut, voir la section [Conversion du modèle](#conversion-du-modèle-si-nécessaire).

### 3. Lancer le projet

Configure d'abord la connexion à PostgreSQL dans le terminal (adapte l'utilisateur,
le mot de passe et le nom de la base) :

```powershell
$env:CONSULTPRO_DATABASE_URL = "postgresql://consultpro:mot_de_passe@localhost:5432/consultpro"
$env:CONSULTPRO_DB_SSLMODE = "prefer"
```

Le compte PostgreSQL de l'application doit être dédié et ne doit pas être
superutilisateur. Pour la production, utilise un mot de passe fort, `sslmode=require`
ou `verify-full`, et un gestionnaire de secrets plutôt qu'un fichier commité.

Depuis la racine du projet (`C:\consultpro`), dans PowerShell :

```powershell
cd C:\consultpro
.\scripts\run_dev.ps1
```

Ce script :
- crée le venv Python et installe les dépendances backend si besoin (première exécution seulement)
- exige `CONSULTPRO_DATABASE_URL`; le schéma PostgreSQL est initialisé au démarrage
- installe les dépendances frontend (`npm install`) si besoin
- lance l’**API FastAPI** sur `http://localhost:8000`
- lance un **worker d’analyse séparé** qui charge le modèle une seule fois
- lance le **frontend Vite** sur `http://localhost:5173`

⚠️ **Première exécution** : si PowerShell refuse d'exécuter le script (erreur "n'est pas signé numériquement"), débloque-le une fois :
```powershell
Unblock-File -Path .\scripts\run_dev.ps1
```

Une fois les deux services démarrés, ouvre **http://localhost:5173** dans le navigateur.

Pour arrêter : `Ctrl+C` dans le terminal, ou ferme les fenêtres ouvertes par le script.

### 4. Utiliser ton propre modèle fine-tuné

Si le modèle n'est pas au chemin par défaut `C:\consultpro\model\whisper-dental-fr-final-ct2`, définis la variable d'environnement avant de lancer le script :

```powershell
$env:CONSULTPRO_WHISPER_MODEL_PATH = "chemin\vers\ton\modele-ct2"
.\scripts\run_dev.ps1
```

## Variables d'environnement (backend)

| Variable | Description | Valeur par défaut (local) |
|---|---|---|
| `CONSULTPRO_WHISPER_MODEL_PATH` | Chemin vers le modèle Whisper (format CTranslate2) ou nom d'un modèle public (`base`, `small`, ...) | `C:\consultpro\model\whisper-dental-fr-final-ct2` |
| `CONSULTPRO_ASR_DEVICE` | Device d'inférence (`cpu` ou `cuda`) | `cpu` |
| `CONSULTPRO_ASR_COMPUTE_TYPE` | Type de calcul (`int8`, `float16`, ...) | `int8` |
| `CONSULTPRO_ASR_CPU_THREADS` | Nombre de threads CPU pour l'inférence | `2` |
| `CONSULTPRO_ASR_BEAM_SIZE` | Beam size Whisper (1 privilégie la vitesse) | `1` |
| `CONSULTPRO_UPLOAD_DIR` | Dossier de stockage des fichiers audio uploadés | `backend\uploads` |
| `CONSULTPRO_WORKER_POLL_SECONDS` | Intervalle d'attente de la file PostgreSQL | `2` |
| `CONSULTPRO_DATABASE_URL` | URL PostgreSQL requise | aucune (à configurer) |
| `CONSULTPRO_DB_SSLMODE` | Mode TLS PostgreSQL (`prefer` local, `require` ou `verify-full` en production) | `prefer` ou `require` selon l'environnement |
| `CONSULTPRO_ENVIRONMENT` | `development` ou `production` | `development` |
| `CONSULTPRO_CORS_ORIGINS` | Liste séparée par virgules des origines frontend autorisées | localhost et Vercel |
| `CONSULTPRO_ALLOWED_HOSTS` | Liste séparée par virgules des noms d'hôtes API autorisés | localhost |
| `CONSULTPRO_ENABLE_HSTS` | Active HSTS; obligatoire en production derrière HTTPS | `false` |
| `CONSULTPRO_MAX_UPLOAD_BYTES` | Taille maximale du fichier audio (25 Mio par défaut) | `26214400` |

Un exemple de configuration sans secret réel se trouve dans [.env.example](./.env.example).
Le backend ne charge pas automatiquement les fichiers `.env` : injecte ces variables
dans l'environnement du processus (ou dans le gestionnaire de secrets de l'hébergeur).

### Base SQLite existante → PostgreSQL

La migration est explicite et ne supprime pas le fichier SQLite source. Fais d'abord
une sauvegarde de la base et configure `CONSULTPRO_DATABASE_URL`, puis exécute depuis
la racine :

```powershell
.\backend\venv\Scripts\python.exe .\scripts\migrate_sqlite_to_postgres.py
```

Les enregistrements existants (comptes, sessions et analyses) sont copiés; une relance
ne duplique pas les identifiants déjà migrés. Vérifie le nombre de lignes et l'accès
aux analyses avant de mettre l'ancienne base hors service.

### File de traitement et performance

L'API web ne charge plus le modèle IA. Elle écrit les uploads valides dans la file
PostgreSQL; un processus `python -m app.worker` dédié charge Whisper une seule fois,
traite les analyses et met à jour leur état. PostgreSQL distribue les tâches avec
`FOR UPDATE SKIP LOCKED`; un heartbeat permet de reprendre une tâche si un worker
s'arrête. Lance au moins un worker. Sur une seule GPU, garde un seul worker pour éviter
de charger plusieurs copies du modèle en VRAM. Des workers CPU supplémentaires ne
doivent être ajoutés qu'après mesure des ressources et de la capacité DB.

Le rapport du worker sépare désormais les durées de chargement audio, diarisation /
clustering, transcription Whisper et post-traitement. Compare CPU et CUDA sur un
enregistrement de test autorisé et non sensible, avec le même modèle et les mêmes
paramètres :

```powershell
.\backend\venv\Scripts\python.exe .\scripts\benchmark_asr.py `
  --audio .\chemin\vers\audio-test.wav `
  --model .\model\whisper-dental-fr-final-ct2 `
  --devices cuda,cpu
```

Le script affiche le temps d'inférence et le facteur temps réel (temps de traitement /
durée audio). Il exécute uniquement la transcription; il ne mesure pas diarisation,
upload, ni latence totale. Fais plusieurs exécutions, compare les résultats, et ne
change `CONSULTPRO_ASR_BEAM_SIZE` que si tu as aussi vérifié l'exactitude des
transcriptions. Le chargement du modèle n'est pas compté dans le temps d'inférence.

Le build Vite sépare maintenant framework, routeur, graphiques et autres dépendances
en morceaux cacheables; cela réduit le JavaScript initial à télécharger, sans garantir
une réduction du temps de compilation.

### Sécurité et déploiement

- Termine HTTPS sur un reverse proxy ou un service d'hébergement fiable. En production,
  `CONSULTPRO_ENVIRONMENT=production` active TLS PostgreSQL par défaut et refuse HSTS
  désactivé. Les cookies ne sont pas utilisés; l'API authentifie les requêtes avec un
  jeton Bearer.
- Définis `CONSULTPRO_CORS_ORIGINS` et `CONSULTPRO_ALLOWED_HOSTS` aux seuls domaines
  réellement utilisés (pas de joker). N'expose pas directement le port PostgreSQL.
- Configure aussi le reverse proxy avec une limite de corps légèrement supérieure à
  26 Mio pour inclure l'enveloppe multipart, par exemple `client_max_body_size 27m`
  sous Nginx. Applique une limitation de débit aux routes `/api/auth/login`,
  `/api/auth/register` et `/api/analysis/upload`; fais-le au proxy/WAF ou dans un
  service partagé, car un limiteur en mémoire serait contournable avec plusieurs workers.
- Valeurs Nginx à adapter (zones dans le bloc `http`, limites dans les locations HTTPS) :

  ```nginx
  limit_req_zone $binary_remote_addr zone=cp_login:10m rate=5r/m;
  limit_req_zone $binary_remote_addr zone=cp_register:10m rate=2r/m;
  limit_req_zone $binary_remote_addr zone=cp_upload:10m rate=6r/m;
  ```

  ```nginx
  client_max_body_size 27m;
  limit_req_status 429;
  location /api/ {
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_pass http://127.0.0.1:8000;
  }
  location = /api/auth/login {
      limit_req zone=cp_login burst=5 nodelay;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_pass http://127.0.0.1:8000;
  }
  location = /api/auth/register {
      limit_req zone=cp_register burst=2 nodelay;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_pass http://127.0.0.1:8000;
  }
  location = /api/analysis/upload {
      limit_req zone=cp_upload burst=2 nodelay;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_pass http://127.0.0.1:8000;
  }
  ```
- Les fichiers audio sont limités à 25 Mio, contrôlés par extension et signature,
  écrits sous un nom généré et supprimés après traitement. Le proxy doit aussi limiter
  le corps des requêtes afin de bloquer les uploads trop volumineux avant le parsing.
- Les résultats et transcriptions restent dans PostgreSQL. Définis une politique de
  conservation/suppression adaptée aux données de consultation, chiffre les volumes et
  sauvegardes, restreins l'accès à la base, et teste régulièrement la restauration.
- Le jeton Bearer du frontend est conservé dans `localStorage`; une politique CSP stricte
  et la prévention des injections côté frontend restent nécessaires. Avant un usage réel
  de données de santé, fais valider l'hébergement, la conformité réglementaire applicable
  et les paramètres de sécurité par les responsables compétents.

## Conversion du modèle (si nécessaire)

Si tu pars d'un checkpoint HuggingFace brut (`whisper-dental-fr-final`) et non déjà converti :

```bash
pip install ctranslate2 "transformers[torch]"

ct2-transformers-converter \
    --model /chemin/vers/whisper-dental-fr-final \
    --output_dir /chemin/vers/whisper-dental-fr-final-ct2 \
    --quantization int8
```

Place ensuite le dossier `whisper-dental-fr-final-ct2/` obtenu dans `model/` à la racine du projet.

## Vérification rapide de l'API

```powershell
curl -X POST http://localhost:8000/api/analysis/upload -F "file=@test.wav"
# -> {"job_id": "...", "status": "pending"}

curl http://localhost:8000/api/analysis/<job_id>
# -> statut, puis résultat complet une fois "done"
```

## Points d'attention

- Le chargement du modèle Whisper au démarrage du backend prend plusieurs secondes (normal, une seule fois au lancement).
- Une consultation de ~15 min peut prendre plusieurs minutes à transcrire sur CPU. Le frontend informe l'utilisateur de ce délai (pattern job asynchrone : upload → polling du statut).
- En déploiement (VPS), ces mêmes variables d'environnement sont définies via le service `systemd` plutôt qu'en ligne de commande.
