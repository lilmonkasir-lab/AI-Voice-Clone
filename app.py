import os, uuid, traceback, threading, subprocess
from pathlib import Path
import torch
import torchaudio as ta
from flask import Flask, jsonify, request, send_from_directory
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Use the bundled ffmpeg binary (imageio-ffmpeg) so MP3 export and reading
# phone-recorded audio (m4a/webm/ogg) works without installing FFmpeg manually.
FFMPEG = None
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
if not FFMPEG:
    FFMPEG = "ffmpeg"  # fall back to a system install, if any


def ffmpeg_convert(src: Path, dst: Path, *extra_args):
    """Run ffmpeg and raise with a readable message on failure."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), *extra_args, str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg conversion failed")

BASE = Path(__file__).resolve().parent
UPLOAD = BASE / "uploads"; OUTPUT = BASE / "outputs"
UPLOAD.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
app = Flask(__name__)

MODEL = None
LOAD_ERROR = None          # last model-load error, surfaced to the UI
MODEL_LOCK = threading.Lock()
LOCK = threading.Lock()

MIN_REF_SECONDS = 5      # hard requirement of the model for voice cloning
MAX_REF_SECONDS = 30     # longer refs are trimmed (model only uses ~10-15s anyway)
MAX_TEXT_CHARS = 2000    # quality degrades on very long inputs


def get_model():
    """Load the model once, thread-safely. Raises after storing the error for /api/status."""
    global MODEL, LOAD_ERROR
    with MODEL_LOCK:
        if MODEL is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Chatterbox Turbo on {device}... (downloads ~2GB on first run)", flush=True)
            try:
                MODEL = ChatterboxTurboTTS.from_pretrained(device=device)
                LOAD_ERROR = None
                print("Model ready.", flush=True)
            except Exception as e:
                LOAD_ERROR = e
                traceback.print_exc()
                raise
        return MODEL


def normalize_reference(src: Path) -> Path:
    """Decode ANY audio file (wav/mp3/m4a/webm/ogg/flac...) and re-encode it as
    a clean 16 kHz mono WAV, trimmed to MAX_REF_SECONDS. Converts phone/VOIP
    recordings into a format the model can always read."""
    dst = src.parent / f"{src.stem}_norm.wav"
    ffmpeg_convert(src, dst,
                   "-vn", "-ac", "1", "-ar", "16000",
                   "-acodec", "pcm_s16le", "-t", str(MAX_REF_SECONDS))
    return dst


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/api/status")
def status():
    payload = {
        "ready": MODEL is not None,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loading": MODEL is None and LOAD_ERROR is None,
        "error": str(LOAD_ERROR) if MODEL is None and LOAD_ERROR else None,
    }
    return jsonify(payload)


@app.post("/api/generate")
def generate():
    global LOAD_ERROR
    text = request.form.get("text", "").strip()
    ref = request.files.get("reference")
    if not text:
        return jsonify(error="Enter some text first."), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify(error=f"Text is too long ({len(text)} chars). Keep it under {MAX_TEXT_CHARS} characters per generation."), 400
    if not ref or not ref.filename:
        return jsonify(error="Choose a reference audio file (5+ seconds of clear speech)."), 400

    try:
        get_model()  # may raise if the model failed to load
    except Exception as e:
        return jsonify(error=f"Model failed to load: {e}"), 500

    job = uuid.uuid4().hex
    suffix = Path(ref.filename).suffix.lower() or ".audio"
    refpath = UPLOAD / f"{job}{suffix}"
    refpath_norm = None
    out = OUTPUT / f"chatterbox_{job}.wav"
    ref.save(refpath)

    # The model needs audio it can decode - normalize everything to WAV first.
    # If conversion is unavailable (no ffmpeg), fall back to the raw upload.
    try:
        try:
            refpath_norm = normalize_reference(refpath)
        except Exception as conv_err:
            print(f"[warn] audio conversion failed ({conv_err}); passing original file to model", flush=True)
            refpath_norm = refpath

        with LOCK:
            model = get_model()
            try:
                wav = model.generate(text, audio_prompt_path=str(refpath_norm))
            except AssertionError as e:
                # e.g. "Audio prompt must be longer than 5 seconds!"
                return jsonify(error=str(e)), 400
            ta.save(str(out), wav, model.sr)
        return jsonify(ok=True, file=f"/outputs/{out.name}")
    except Exception as e:
        traceback.print_exc()
        msg = str(e)
        low = msg.lower()
        if any(k in low for k in ("libsndfile", "audioread", "soundfile", "format not recognised", "no backend")):
            msg = ("Couldn't read that reference audio file. Use a WAV/MP3/M4A recording of "
                   "5+ seconds of clear speech (install FFmpeg for other formats).")
        return jsonify(error=msg), 500
    finally:
        refpath.unlink(missing_ok=True)
        if refpath_norm and refpath_norm != refpath:
            refpath_norm.unlink(missing_ok=True)


@app.post("/api/mp3")
def mp3():
    name = Path((request.json or {}).get("file", "")).name
    if not name.lower().endswith(".wav"):
        return jsonify(error="Invalid WAV."), 400
    wav = OUTPUT / name
    if not wav.exists():
        return jsonify(error="WAV not found."), 404
    out = wav.with_suffix(".mp3")
    try:
        ffmpeg_convert(wav, out, "-vn", "-c:a", "libmp3lame", "-b:a", "192k")
        return jsonify(ok=True, file=f"/outputs/{out.name}")
    except Exception as e:
        return jsonify(error=f"MP3 conversion failed: {e}"), 500


@app.get("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT, name, as_attachment=False)


if __name__ == "__main__":
    host = os.environ.get("CHATTERBOX_HOST", "127.0.0.1")
    port = int(os.environ.get("CHATTERBOX_PORT", "7860"))
    print(f"Open http://{host}:{port}")
    # Start the web server immediately and load the model in the background,
    # so the page opens right away and any load error is shown in the UI.
    threading.Thread(target=get_model, daemon=True).start()
    app.run(host, port, debug=False, threaded=True)
