# -*- coding: utf-8 -*-
"""Download ESC-50 (if needed) and train a baby-cry vs other-sound model.

Uses the same 18 audio features as the original project (zero-crossing rate,
RMS, 13 MFCCs, spectral centroid, rolloff, bandwidth) and a scaled SVM.
"""

import argparse
import json
import logging
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "esc50.zip"
MODEL_DIR = ROOT / "output" / "model"
FEATURE_CACHE = ROOT / "output" / "dataset" / "esc50_features.npz"
DOWNLOAD_URL = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip"
MIN_ZIP_BYTES = 500_000_000


def os_cpu_count():
    import os
    return os.cpu_count() or 2


def find_dataset():
    """Return (audio_dir, csv_path) for an extracted ESC-50 tree."""
    candidates = list(DATA_DIR.glob("ESC-50*/meta/esc50.csv"))
    candidates += list(DATA_DIR.glob("*/meta/esc50.csv"))
    for csv_path in candidates:
        audio_dir = csv_path.parent.parent / "audio"
        if audio_dir.is_dir():
            return audio_dir, csv_path
    return None, None


def download_zip():
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size >= MIN_ZIP_BYTES:
        print("Dataset zip already downloaded.")
        return

    print("Downloading ESC-50 (about 600 MB). This runs once.")
    tmp = ZIP_PATH.with_suffix(".zip.part")
    urllib.request.urlretrieve(DOWNLOAD_URL, tmp)
    tmp.replace(ZIP_PATH)
    print("Downloaded {0:.0f} MB".format(ZIP_PATH.stat().st_size / 1e6))


def extract_zip():
    audio_dir, csv_path = find_dataset()
    if audio_dir is not None:
        print("Dataset already extracted: {0}".format(audio_dir))
        return audio_dir, csv_path

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < MIN_ZIP_BYTES:
        raise SystemExit("ESC-50 zip is missing or incomplete: {0}".format(ZIP_PATH))

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(DATA_DIR)

    audio_dir, csv_path = find_dataset()
    if audio_dir is None:
        raise SystemExit("Could not find ESC-50 audio after extraction.")
    return audio_dir, csv_path


def read_metadata(csv_path):
    import csv

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def add_noise(signal, snr_db, rng):
    power = float(np.mean(signal ** 2)) + 1e-12
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return (signal + noise).astype(signal.dtype, copy=False)


def _extract_file(task):
    """Compute features for one clip. Runs in a worker process."""
    index, path, is_cry = task
    import librosa
    from baby_cry_detection.pc_methods.feature_engineer import FeatureEngineer

    logging.disable(logging.INFO)
    audio, _ = librosa.load(path, sr=44100, mono=True, duration=5.0)
    signals = [audio]
    if is_cry:
        rng = np.random.default_rng(index)
        signals.extend([
            add_noise(audio, 20, rng),
            add_noise(audio, 10, rng),
            audio * 0.6,
            audio * 1.4,
        ])

    engineer = FeatureEngineer()
    label = "Crying baby" if is_cry else "Not crying"
    features = [engineer.feature_engineer(signal)[0].ravel() for signal in signals]
    return features, [label] * len(features)


def select_training_rows(rows):
    """Use every baby-cry clip and four examples of each other ESC-50 sound.

    The full set is 2,000 files. Four clips from each other class keeps the
    model balanced and finishes training on a normal PC.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)

    selected = []
    for category, items in grouped.items():
        if category == "crying_baby":
            selected.extend(items)
        else:
            selected.extend(items[:4])
    return selected


def extract_features(audio_dir, rows):
    if FEATURE_CACHE.exists():
        print("Loading cached features: {0}".format(FEATURE_CACHE))
        cached = np.load(FEATURE_CACHE, allow_pickle=True)
        return cached["X"], cached["y"], str(cached["cry_file"]), str(cached["other_file"])

    cry_file = None
    other_file = None
    tasks = []
    for index, row in enumerate(rows, start=1):
        path = audio_dir / row["filename"]
        is_cry = row["category"] == "crying_baby"
        if is_cry and cry_file is None:
            cry_file = str(path)
        if (not is_cry) and row["category"] == "dog" and other_file is None:
            other_file = str(path)
        tasks.append((index, str(path), is_cry))

    total = len(tasks)
    print("Reading {0} clips and computing features...".format(total))
    features = []
    labels = []
    done = 0
    workers = min(2, os_cpu_count())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_file, task) for task in tasks]
        for future in as_completed(futures):
            file_features, file_labels = future.result()
            features.extend(file_features)
            labels.extend(file_labels)
            done += 1
            if done % 100 == 0 or done == total:
                print("  {0}/{1}".format(done, total), flush=True)

    if other_file is None:
        for row in rows:
            if row["category"] != "crying_baby":
                other_file = str(audio_dir / row["filename"])
                break

    x_values = np.vstack(features).astype(np.float32)
    y_values = np.array(labels)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_CACHE,
        X=x_values,
        y=y_values,
        cry_file=cry_file or "",
        other_file=other_file or "",
    )
    print("Saved features: {0}".format(FEATURE_CACHE))
    return x_values, y_values, cry_file, other_file


def train(x_values, y_values):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=0.25,
        random_state=0,
        stratify=y_values,
    )

    pipeline = Pipeline([
        ("scl", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced")),
    ])
    param_grid = [
        {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10, 100]},
        {
            "clf__kernel": ["rbf"],
            "clf__C": [1, 10, 100],
            "clf__gamma": ["scale", 0.1, 0.01],
        },
    ]
    print("Training SVM (this can take a few minutes)...")
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_macro", n_jobs=2)
    search.fit(x_train, y_train)
    model = search.best_estimator_
    predictions = model.predict(x_test)

    performance = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions, average="macro")),
        "precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="macro")),
        "cry_recall": float(recall_score(y_test, predictions, pos_label="Crying baby", zero_division=0)),
        "cry_precision": float(precision_score(y_test, predictions, pos_label="Crying baby", zero_division=0)),
        "n_samples": int(len(y_values)),
        "n_cry": int(np.sum(y_values == "Crying baby")),
        "n_other": int(np.sum(y_values == "Not crying")),
        "best_params": {key: (value if isinstance(value, (int, float, str)) else str(value))
                        for key, value in search.best_params_.items()},
    }
    return model, performance


def save_model(model, performance, cry_file, other_file):
    import pickle

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    with open(MODEL_DIR / "performance.json", "w", encoding="utf-8") as handle:
        json.dump(performance, handle, indent=2)

    samples = {
        "cry": cry_file,
        "other": other_file,
    }
    with open(MODEL_DIR / "samples.json", "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)

    print("Saved model: {0}".format(model_path))
    print("Held-out accuracy: {0:.1%}".format(performance["accuracy"]))
    print("Baby-cry recall: {0:.1%}".format(performance["cry_recall"]))
    print("Baby-cry precision: {0:.1%}".format(performance["cry_precision"]))


def self_check(model, cry_file, other_file):
    import librosa
    from baby_cry_detection.rpi_methods.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()
    print("Quick check on two known clips:")
    for label, path in (("baby cry", cry_file), ("dog bark", other_file)):
        if not path or not Path(path).exists():
            print("  skipped {0}: file missing".format(label))
            continue
        audio, _ = librosa.load(path, sr=44100, mono=True, duration=5.0)
        features = engineer.feature_engineer(audio)
        predicted = model.predict(features)[0]
        print("  {0} -> {1}".format(label, predicted))


def main():
    parser = argparse.ArgumentParser(description="Prepare ESC-50 data and train the baby-cry model.")
    parser.add_argument("--force", action="store_true", help="Retrain even if model.pkl already exists.")
    args = parser.parse_args()

    model_path = MODEL_DIR / "model.pkl"
    if model_path.exists() and not args.force:
        print("Model already exists at {0}".format(model_path))
        print("Start the test page with run_ui.bat, or retrain with --force.")
        return

    try:
        download_zip()
    except Exception as exc:
        audio_dir, _ = find_dataset()
        if audio_dir is None and (not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < MIN_ZIP_BYTES):
            raise SystemExit("Could not download ESC-50: {0}".format(exc))

    audio_dir, csv_path = extract_zip()
    rows = select_training_rows(read_metadata(csv_path))
    print("Using {0} clips ({1} baby-cry).".format(
        len(rows),
        sum(1 for row in rows if row["category"] == "crying_baby"),
    ))
    x_values, y_values, cry_file, other_file = extract_features(audio_dir, rows)
    model, performance = train(x_values, y_values)
    save_model(model, performance, cry_file, other_file)
    self_check(model, cry_file, other_file)


if __name__ == "__main__":
    main()
