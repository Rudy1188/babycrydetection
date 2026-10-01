# -*- coding: utf-8 -*-
"""Local test page for the baby-cry model."""

import csv
import json
import os
import pickle
import sys
import tempfile
import warnings
import webbrowser
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baby_cry_detection.rpi_methods.feature_engineer import FeatureEngineer
from baby_cry_detection.rpi_methods.majority_voter import MajorityVoter

MODEL_DIR = ROOT / "output" / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
PERFORMANCE_PATH = MODEL_DIR / "performance.json"
SAMPLES_PATH = MODEL_DIR / "samples.json"
SAMPLE_RATE = 44100
OTHERS_PER_CLASS = 3
WINDOW_SECONDS = 5
MAX_WINDOWS = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

_model = None
_engineer = FeatureEngineer()
_catalog = None


def load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            return None
        with open(MODEL_PATH, "rb") as handle:
            _model = pickle.load(handle)
    return _model


def cry_probability(model, features):
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    index = classes.index("Crying baby")
    return float(probabilities[index])


def slice_windows(audio):
    """Cut a clip the same way the original predictor does.

    A 9 second recording becomes five overlapping 5 second windows.
    Shorter clips use as many full windows as they contain, or the whole clip.
    """
    window = WINDOW_SECONDS * SAMPLE_RATE
    hop = SAMPLE_RATE
    if len(audio) < int(0.5 * SAMPLE_RATE):
        raise ValueError("Audio is too short. Use at least 1 second.")

    if len(audio) < window:
        return [audio], [0.0]

    audio = audio[: 9 * SAMPLE_RATE]
    clips = []
    starts = []
    for start in range(0, len(audio) - window + 1, hop):
        clips.append(audio[start:start + window])
        starts.append(start / SAMPLE_RATE)
        if len(clips) == MAX_WINDOWS:
            break
    return clips, starts


def predict_audio(audio):
    model = get_model()
    if model is None:
        raise FileNotFoundError("Model file is missing. Run run_ui.bat once so it can train.")

    clips, starts = slice_windows(audio)
    window_results = []
    votes = []
    probabilities = []

    for index, (clip, start) in enumerate(zip(clips, starts), start=1):
        features = _engineer.feature_engineer(clip)
        probability = cry_probability(model, features)
        vote = 1 if probability >= 0.5 else 0
        votes.append(vote)
        probabilities.append(probability)
        window_results.append({
            "index": index,
            "start_sec": round(start, 2),
            "cry_probability": round(probability, 4),
            "label": "Baby cry" if vote == 1 else "Not a cry",
        })

    majority = int(MajorityVoter(votes).vote())
    mean_probability = float(np.mean(probabilities))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    warning = None
    if rms < 0.005:
        warning = "This clip is very quiet. Move the microphone closer and try again."

    return {
        "is_baby_cry": bool(majority),
        "label": "Baby cry detected" if majority else "Not a baby cry",
        "confidence": round(mean_probability if majority else 1.0 - mean_probability, 4),
        "cry_probability": round(mean_probability, 4),
        "windows": window_results,
        "duration_sec": round(len(audio) / SAMPLE_RATE, 2),
        "warning": warning,
    }


def load_upload(storage):
    import librosa

    suffix = Path(storage.filename or "clip.wav").suffix.lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}:
        raise ValueError("Use a WAV, MP3, OGG, FLAC, M4A, or WebM file.")

    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temporary.close()
    try:
        storage.save(temporary.name)
        audio, _ = librosa.load(temporary.name, sr=SAMPLE_RATE, mono=True)
    finally:
        if os.path.exists(temporary.name):
            os.remove(temporary.name)

    if audio.size == 0:
        raise ValueError("Could not read any audio from that file.")
    return audio


def pretty_category(name):
    if name == "crying_baby":
        return "Baby cry"
    return name.replace("_", " ").title()


def dataset_paths():
    csv_files = list((ROOT / "data").glob("ESC-50*/meta/esc50.csv"))
    if not csv_files:
        return None, None
    csv_path = csv_files[0]
    audio_dir = csv_path.parent.parent / "audio"
    if not audio_dir.is_dir():
        return None, None
    return audio_dir, csv_path


def load_catalog():
    """Baby-cry clips plus a few examples of every other sound."""
    global _catalog
    if _catalog is not None:
        return _catalog

    audio_dir, csv_path = dataset_paths()
    if audio_dir is None:
        _catalog = {"audio_dir": None, "clips": {}, "groups": []}
        return _catalog

    grouped = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["category"], []).append(row["filename"])

    clips = {}
    groups = []
    categories = ["crying_baby"] + sorted(name for name in grouped if name != "crying_baby")
    for category in categories:
        filenames = grouped.get(category, [])
        if category != "crying_baby":
            filenames = filenames[:OTHERS_PER_CLASS]
        group_name = pretty_category(category)
        is_cry = category == "crying_baby"
        group_clips = []
        for index, filename in enumerate(filenames, start=1):
            meta = {
                "id": filename,
                "title": "{0} {1}".format(group_name, index),
                "category": category,
                "category_label": group_name,
                "is_cry": is_cry,
            }
            clips[filename] = meta
            group_clips.append(meta)
        if group_clips:
            groups.append({
                "id": category,
                "name": group_name,
                "is_cry": is_cry,
                "clips": group_clips,
            })

    _catalog = {"audio_dir": audio_dir, "clips": clips, "groups": groups}
    return _catalog


def resolve_clip(clip_id):
    catalog = load_catalog()
    meta = catalog["clips"].get(clip_id)
    audio_dir = catalog["audio_dir"]
    if meta is None or audio_dir is None:
        raise FileNotFoundError("Unknown sample.")
    root = audio_dir.resolve()
    path = (root / clip_id).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError("Sample file is missing.")
    return path, meta


def sample_path(kind):
    samples = load_json(SAMPLES_PATH) or {}
    path = samples.get(kind)
    if not path:
        raise FileNotFoundError("Sample clips are not available until training finishes.")
    resolved = Path(path).resolve()
    data_root = (ROOT / "data").resolve()
    if data_root not in resolved.parents or not resolved.exists():
        raise FileNotFoundError("Sample file is missing: {0}".format(path))
    return resolved


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    performance = load_json(PERFORMANCE_PATH) or {}
    samples = load_json(SAMPLES_PATH) or {}
    cry_ok = bool(samples.get("cry")) and Path(samples["cry"]).exists()
    other_ok = bool(samples.get("other")) and Path(samples["other"]).exists()
    return jsonify({
        "model_ready": MODEL_PATH.exists(),
        "samples_ready": cry_ok and other_ok,
        "performance": {
            "accuracy": performance.get("accuracy"),
            "cry_recall": performance.get("cry_recall"),
            "cry_precision": performance.get("cry_precision"),
            "n_cry": performance.get("n_cry"),
            "n_other": performance.get("n_other"),
        },
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "Choose an audio file first."}), 400
    upload = request.files["audio"]
    if not upload.filename:
        return jsonify({"error": "Choose an audio file first."}), 400
    try:
        result = predict_audio(load_upload(upload))
    except Exception as exc:
        message = str(exc)
        if "ffmpeg" in message.lower() or "NoBackendError" in type(exc).__name__ or "audioread" in message.lower():
            message = "Could not decode that file. Save it as WAV, or install ffmpeg to open MP3 files."
        return jsonify({"error": message}), 400
    return jsonify(result)


@app.route("/api/samples")
def samples():
    catalog = load_catalog()
    groups = []
    for group in catalog["groups"]:
        groups.append({
            "id": group["id"],
            "name": group["name"],
            "is_cry": group["is_cry"],
            "clips": [
                {
                    "id": clip["id"],
                    "title": clip["title"],
                    "is_cry": clip["is_cry"],
                    "category_label": clip["category_label"],
                }
                for clip in group["clips"]
            ],
        })
    return jsonify({"groups": groups, "count": sum(len(group["clips"]) for group in groups)})


@app.route("/api/audio/<clip_id>")
def audio(clip_id):
    try:
        path, _meta = resolve_clip(clip_id)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    return send_file(path, mimetype="audio/wav", conditional=True, max_age=3600)


@app.route("/api/test/<clip_id>")
def test_clip(clip_id):
    try:
        import librosa

        path, meta = resolve_clip(clip_id)
        audio_data, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        result = predict_audio(audio_data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    result["source"] = meta["title"]
    result["expected_cry"] = meta["is_cry"]
    result["expected_label"] = meta["category_label"]
    result["correct"] = bool(result["is_baby_cry"]) == bool(meta["is_cry"])
    result["detection_accuracy"] = result["confidence"]
    return jsonify(result)


@app.route("/api/sample/<kind>")
def predict_sample(kind):
    if kind not in {"cry", "other"}:
        return jsonify({"error": "Unknown sample."}), 404
    try:
        import librosa

        audio, _ = librosa.load(str(sample_path(kind)), sr=SAMPLE_RATE, mono=True)
        result = predict_audio(audio)
        result["source"] = "Built-in baby cry clip" if kind == "cry" else "Built-in dog bark clip"
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


def main():
    if not MODEL_PATH.exists():
        print("No trained model yet. Run:  python scripts\\prepare_and_train.py")
    host = os.environ.get("BABY_CRY_HOST", "127.0.0.1")
    port = int(os.environ.get("BABY_CRY_PORT", "5000"))
    url = "http://{0}:{1}".format(host, port)
    print("Test page: {0}".format(url))
    if os.environ.get("BABY_CRY_NO_BROWSER") != "1":
        webbrowser.open("http://127.0.0.1:{0}".format(port))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
