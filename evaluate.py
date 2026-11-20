"""Evaluate the desktop detector against a manifest of labeled videos."""

import argparse
import csv
import json
from pathlib import Path

from processor import AnalysisError, DEFAULT_THRESHOLD, MODEL_PATH, classify_video


def read_manifest(path):
    videos = {}
    with path.open(encoding="utf-8-sig", newline="") as manifest:
        rows = csv.DictReader(manifest, strict=True)
        if not rows.fieldnames or not {"path", "label"}.issubset(rows.fieldnames):
            raise ValueError("The manifest must have path and label columns.")
        for row in rows:
            filename = (row["path"] or "").strip()
            label = (row["label"] or "").strip().lower()
            if not filename or label not in {"real", "fake"}:
                raise ValueError(f"Invalid path or label on CSV line {rows.line_num}.")
            video = (path.parent / filename).resolve()
            if video in videos:
                raise ValueError(f"Duplicate video on CSV line {rows.line_num}: {video}")
            videos[video] = label

    if set(videos.values()) != {"real", "fake"}:
        raise ValueError("The manifest must contain both real and fake videos.")
    return videos


def evaluate(videos, model_path, threshold):
    counts = {
        "real": {"real": 0, "fake": 0},
        "fake": {"real": 0, "fake": 0},
    }
    for video, actual in videos.items():
        try:
            result = classify_video(video, model_path=model_path)
        except AnalysisError as exc:
            raise AnalysisError(f"{video}: {exc}") from exc
        predicted = "fake" if result.fake_score >= threshold else "real"
        counts[actual][predicted] += 1

    true_positive = counts["fake"]["fake"]
    false_positive = counts["real"]["fake"]
    false_negative = counts["fake"]["real"]
    predicted_fake = true_positive + false_positive
    actual_fake = true_positive + false_negative
    return {
        "model": str(model_path),
        "threshold": threshold,
        "videos": len(videos),
        "confusion_matrix_actual_then_predicted": counts,
        "fake_precision": true_positive / predicted_fake if predicted_fake else 0.0,
        "fake_recall": true_positive / actual_fake,
        "fake_f1": 2 * true_positive / (actual_fake + predicted_fake),
        "false_positive_rate": false_positive / sum(counts["real"].values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with path and label columns.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a Keras model.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    try:
        videos = read_manifest(args.manifest)
        report = evaluate(videos, args.model, args.threshold)
    except (AnalysisError, OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
