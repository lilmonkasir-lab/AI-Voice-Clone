import os, uuid, traceback, threading, subprocess, re
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
UPLOAD = BASE / "uploads"; OUTPUT = BASE / "outputs"; PREVIEWS = BASE / "tag_previews"
UPLOAD.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True); PREVIEWS.mkdir(exist_ok=True)
app = Flask(__name__)

MODEL = None
LOAD_ERROR = None          # last model-load error, surfaced to the UI
MODEL_LOCK = threading.Lock()
LOCK = threading.Lock()

MIN_REF_SECONDS = 5      # hard requirement of the model for voice cloning
MAX_REF_SECONDS = 30     # longer refs are trimmed (model only uses ~10-15s anyway)
MAX_TEXT_CHARS = 2000    # quality degrades on very long inputs

# Full paralinguistic tag catalog from the model's added_tokens.json (token IDs 50257-50275).
# "sfx"   = vocal sound effects (confirmed by Resemble) - injected sound at the tag position.
# "style" = delivery/emotion tags (experimental) - color the whole sentence; effect varies by voice.
TAGS = [
    # tag,            label,          emoji, description,                                                category, preview script
    ("laugh",        "Laugh",        "😂", "A full, hearty laugh.",                                     "sfx",   "Oh that is hilarious [laugh], I haven't laughed like that in years."),
    ("chuckle",      "Chuckle",      "😏", "A soft, brief laugh.",                                      "sfx",   "Well, that's one way to fix it [chuckle], I guess we'll never know."),
    ("sigh",         "Sigh",         "😮‍💨", "A tired or resigned exhale.",                               "sfx",   "It's been such a long week [sigh], but we'll get there."),
    ("gasp",         "Gasp",         "😲", "A sharp, shocked intake of breath.",                        "sfx",   "Did you see that [gasp], it nearly took my head off."),
    ("groan",        "Groan",        "😩", "A frustrated, pained sound.",                               "sfx",   "Not this again [groan], why does this always happen to me."),
    ("cough",        "Cough",        "🤧", "A single cough.",                                           "sfx",   "Excuse me [cough], sorry, where was I."),
    ("sniff",        "Sniff",        "👃", "A quick nasal sniff.",                                      "sfx",   "I must be coming down with something [sniff], anyway, as I was saying."),
    ("shush",        "Shush",        "🤫", "A 'shhh' hushing sound.",                                   "sfx",   "Everyone quiet [shush], the show is about to start."),
    ("clear throat", "Clear throat", "🙂", "An 'ahem' throat clear.",                                   "sfx",   "Ahem [clear throat], ladies and gentlemen, your attention please."),
    ("happy",        "Happy",        "😄", "Bright, upbeat delivery.",                                  "style", "[happy] We just hit a million subscribers, can you believe it!"),
    ("angry",        "Angry",        "😠", "Hot, intense delivery.",                                    "style", "[angry] This is completely unacceptable and I want it fixed today."),
    ("sarcastic",    "Sarcastic",    "🙃", "Dry, ironic, deadpan delivery.",                            "style", "[sarcastic] Oh sure, that went absolutely perfectly."),
    ("surprised",    "Surprised",    "😳", "Wide-eyed, disbelieving delivery.",                         "style", "[surprised] Wait, you actually finished the whole thing already?"),
    ("fear",         "Fear",         "😨", "Tense, uneasy delivery.",                                   "style", "[fear] Something is moving in the hallway, and I do not like it."),
    ("crying",       "Crying",       "😢", "Choked-up, tearful delivery.",                              "style", "[crying] I cannot believe they are really gone."),
    ("whispering",   "Whispering",   "🤫", "Lowered, hushed delivery - secrets and tension.",           "style", "[whispering] Nobody is supposed to know about this, keep it down."),
    ("dramatic",     "Dramatic",     "🎬", "Big, theatrical trailer-style delivery.",                   "style", "[dramatic] This changes everything we thought we knew."),
    ("narration",    "Narration",    "📖", "Steady documentary/storyteller delivery.",                  "style", "[narration] The year was nineteen sixty nine, and the world held its breath."),
    ("advertisement","Advertisement","📣", "Energetic promo / announcer delivery.",                     "style", "[advertisement] Try it free for thirty days, cancel anytime."),
]
TAG_CATEGORIES = {
    "sfx":   "🔊 Vocal sound effects — confirmed by Resemble",
    "style": "🎭 Delivery & emotion (experimental — effect varies by voice)",
}
TAG_LOOKUP = {f"[{t[0]}]": {"tag": f"[{t[0]}]", "label": t[1], "emoji": t[2],
                            "desc": t[3], "category": t[4], "preview": t[5]}
              for t in TAGS}

PRESET_SCRIPTS = [
    ("📞 Support callback", "Hi there, this is Sarah from MochaFone [clear throat], I'm just calling you back about the billing issue [sigh], it's a bit of a long story but I'm here to help."),
    ("👻 Horror trailer", "[whispering] Nobody was supposed to open that door [gasp] and now [dramatic] something is coming this way."),
    ("🎉 Excited podcaster", "[happy] Okay wow, we just crossed a million subscribers [laugh], I honestly cannot believe it!"),
    ("😮‍💨 Tired coworker", "[sigh] It's Monday again [groan], and the coffee machine is broken [cough], so this should be fun [sarcastic]."),
    ("📚 Audiobook opening", "[narration] The year was eighteen forty two [clear throat], and London was about to change forever [dramatic]."),
    ("😈 Cartoon villain", "[dramatic] You thought you could stop me [chuckle] oh, that's adorable [laugh]."),
    ("🤫 Sharing a secret", "[shush] [whispering] Meet me behind the old library at midnight [sniff], and tell absolutely no one."),
]


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


# ---------- Paralinguistic tags ----------

@app.get("/api/tags")
def api_tags():
    return jsonify(
        tags=list(TAG_LOOKUP.values()),
        categories=TAG_CATEGORIES,
        presets=[{"label": p[0], "text": p[1]} for p in PRESET_SCRIPTS],
    )


@app.post("/api/tag-preview")
def tag_preview():
    tag = (request.json or {}).get("tag", "").strip().lower()
    info = TAG_LOOKUP.get(tag)
    if not info:
        return jsonify(error=f"Unknown tag: {tag or '(none)'}"), 400

    slug = re.sub(r"[^a-z0-9]+", "_", info["tag"].strip("[]")).strip("_")
    f = PREVIEWS / f"tag_{slug}.wav"
    if f.exists():
        return jsonify(ok=True, file=f"/tag-previews/{f.name}", cached=True)

    try:
        model = get_model()
    except Exception as e:
        return jsonify(error=f"Model failed to load: {e}"), 500
    if getattr(model, "conds", None) is None:
        return jsonify(error="Built-in preview voice unavailable in this model build — use Generate with a reference voice instead."), 503

    try:
        with LOCK:
            wav = model.generate(info["preview"])  # no reference -> built-in default voice
            ta.save(str(f), wav, model.sr)
        return jsonify(ok=True, file=f"/tag-previews/{f.name}")
    except Exception as e:
        traceback.print_exc()
        f.unlink(missing_ok=True)
        return jsonify(error=f"Preview failed: {e}"), 500


@app.get("/tag-previews/<path:name>")
def serve_tag_previews(name):
    return send_from_directory(PREVIEWS, name, as_attachment=False)


if __name__ == "__main__":
    host = os.environ.get("CHATTERBOX_HOST", "127.0.0.1")
    port = int(os.environ.get("CHATTERBOX_PORT", "7860"))
    print(f"Open http://{host}:{port}")
    # Start the web server immediately and load the model in the background,
    # so the page opens right away and any load error is shown in the UI.
    threading.Thread(target=get_model, daemon=True).start()
    app.run(host, port, debug=False, threaded=True)
