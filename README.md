# Chatterbox Turbo Local

Local AI voice cloning web app (Flask + [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox)).

## Run it (Windows)

1. Install [Python 3.11](https://www.python.org/downloads/windows/) (tick "Add to PATH").
2. Double-click `start.bat`.
3. Open http://127.0.0.1:7860

First run downloads the model (~2 GB) in the background; the page shows progress.

## Using the app

- **Reference voice**: at least **5 seconds** of clean speech from one speaker (10–20s is ideal).
  WAV/MP3/M4A/WebM/OGG all work — files are converted automatically (bundled ffmpeg, no manual install).
- **Text**: up to 2000 characters per generation.
- Download generated audio as WAV or MP3.

## If something breaks

- The page now shows the real error (model load failures, unreadable reference audio, etc.) instead of spinning forever.
- Delete the `.venv` folder and run `start.bat` again — `requirements.txt` pins a verified-working
  stack and pip will rebuild it exactly.

## Files

- `app.py` — Flask server (voice generation, audio normalization, WAV/MP3 export)
- `index.html` — web UI
- `requirements.txt` — pinned, verified-compatible dependency versions
- `start.bat` — creates/updates the venv and launches the app
