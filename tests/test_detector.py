import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np

from evaluate import evaluate
from processor import AnalysisError, VideoAnalysis, classify_video


def capture_of(frames):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    return capture


class DetectorTests(unittest.TestCase):
    def test_same_frames_produce_same_score_under_different_names(self):
        frame = np.full((8, 10, 3), (12, 24, 48), dtype=np.uint8)
        captures = [capture_of([frame]), capture_of([frame])]
        model = Mock(return_value=np.array([[0.25]]))
        with (
            patch("processor.cv2.VideoCapture", side_effect=captures),
            patch("processor._load_model", return_value=model),
        ):
            neutral = classify_video("holiday.mp4")
            renamed = classify_video("fake_synthesis.mp4")

        self.assertEqual(neutral, VideoAnalysis(0.25, 1))
        self.assertEqual(neutral, renamed)
        self.assertEqual(neutral.label, "real")
        self.assertEqual(VideoAnalysis(0.5, 16).label, "fake")
        model_args, model_kwargs = model.call_args
        batch = model_args[0]
        self.assertEqual(batch.shape, (1, 16, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertFalse(model_kwargs["training"])
        np.testing.assert_allclose(batch[0, -1, 0, 0], np.array([48, 24, 12]) / 255)
        for capture in captures:
            capture.release.assert_called_once()

    def test_empty_video_fails_before_loading_the_model(self):
        capture = capture_of([])
        with (
            patch("processor.cv2.VideoCapture", return_value=capture),
            patch("processor._load_model") as load,
        ):
            with self.assertRaisesRegex(AnalysisError, "no readable frames"):
                classify_video("empty.mp4")
        capture.release.assert_called_once()
        load.assert_not_called()

    def test_missing_model_produces_an_explicit_error(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("processor.cv2.VideoCapture", return_value=capture_of([frame])),
        ):
            model_path = Path(directory) / "missing.keras"
            with self.assertRaisesRegex(AnalysisError, "Model file not found"):
                classify_video("valid.mp4", model_path=model_path)

    def test_loads_complete_keras_archive_with_variable_sequence_length(self):
        from tensorflow import keras

        model = keras.Sequential([
            keras.Input((None, 224, 224, 3)),
            keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D()),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros"),
        ])
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("processor.cv2.VideoCapture", return_value=capture_of([frame])),
        ):
            model_path = Path(directory) / "fixture.keras"
            model.save(model_path)
            self.assertEqual(
                classify_video("clip.mp4", model_path=model_path),
                VideoAnalysis(0.5, 1),
            )

    def test_loads_legacy_single_tensor_encoder_and_preserves_weights(self):
        from tensorflow import keras

        frame_input = keras.Input((224, 224, 3))
        frame_output = keras.layers.GlobalAveragePooling2D()(frame_input)
        encoder = keras.Model(frame_input, frame_output)
        model = keras.Sequential([
            keras.Input((16, 224, 224, 3)),
            keras.layers.TimeDistributed(encoder),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros"),
        ])
        model.layers[-1].bias.assign([1.0])
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("processor.cv2.VideoCapture", return_value=capture_of([frame])),
        ):
            original = Path(directory) / "modern.keras"
            legacy = Path(directory) / "legacy.keras"
            model.save(original)
            with ZipFile(original) as source, ZipFile(legacy, "w") as target:
                config = json.loads(source.read("config.json"))
                input_layer, wrapper, *_ = config["config"]["layers"]
                input_layer["config"]["batch_shape"][1] = None
                config["build_config"]["input_shape"][1] = None
                wrapper["build_config"]["input_shape"][1] = None
                encoder_config = wrapper["config"]["layer"]["config"]
                for key in ("input_layers", "output_layers"):
                    encoder_config[key] = [encoder_config[key]]
                target.writestr("config.json", json.dumps(config))
                for entry in ("metadata.json", "model.weights.h5"):
                    target.writestr(entry, source.read(entry))
            result = classify_video("clip.mp4", model_path=legacy)
            self.assertAlmostEqual(result.fake_score, 0.7310586, places=6)

    def test_nonfinite_score_fails_instead_of_returning_a_label(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        model = Mock(return_value=np.array([[np.nan]]))
        with (
            patch("processor.cv2.VideoCapture", return_value=capture_of([frame])),
            patch("processor._load_model", return_value=model),
        ):
            with self.assertRaisesRegex(AnalysisError, "invalid score"):
                classify_video("clip.mp4")

    def test_evaluation_counts_false_positives_and_false_negatives(self):
        videos = {
            Path("a.mp4"): "real",
            Path("b.mp4"): "real",
            Path("c.mp4"): "fake",
            Path("d.mp4"): "fake",
        }
        analyses = [VideoAnalysis(score, 16) for score in (0.1, 0.8, 0.3, 0.9)]
        with patch("evaluate.classify_video", side_effect=analyses):
            report = evaluate(videos, Path("Model.keras"), 0.5)
        self.assertEqual(
            report["confusion_matrix_actual_then_predicted"],
            {"real": {"real": 1, "fake": 1}, "fake": {"real": 1, "fake": 1}},
        )
        self.assertEqual(report["fake_f1"], 0.5)
        self.assertEqual(report["false_positive_rate"], 0.5)
