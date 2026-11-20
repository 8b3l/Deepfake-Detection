"""Video sampling and inference for the released deepfake detector."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().with_name("Model.keras")
SEQUENCE_LENGTH = 16
FRAME_STRIDE = 5
FRAME_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.5


class AnalysisError(RuntimeError):
    """An analysis could not produce a valid result."""


@dataclass(frozen=True)
class VideoAnalysis:
    fake_score: float
    sampled_frames: int

    @property
    def label(self) -> str:
        return "fake" if self.fake_score >= DEFAULT_THRESHOLD else "real"


def _load_legacy_model(keras, path: Path):
    """Adapt the released Keras 3.3 archive to current Keras serializers."""
    with ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    if config["class_name"] != "Sequential":
        raise ValueError("Expected a legacy Sequential model.")

    input_layer, wrapper, *_ = config["config"]["layers"]
    shape = input_layer["config"]["batch_shape"]
    encoder = wrapper["config"]["layer"]
    if (
        input_layer["class_name"] != "InputLayer"
        or shape != [None, None, *FRAME_SIZE, 3]
        or wrapper["class_name"] != "TimeDistributed"
        or encoder["class_name"] != "Functional"
    ):
        raise ValueError("Unexpected architecture in the legacy model.")

    # The old serializer wrapped the encoder's single input and output in lists.
    # TimeDistributed expects a single tensor shape from the frame encoder.
    endpoints = encoder["config"]
    for key in ("input_layers", "output_layers"):
        refs = endpoints[key]
        if len(refs) != 1 or not isinstance(refs[0], list) or len(refs[0]) != 3:
            raise ValueError("Unexpected input or output in the legacy frame encoder.")
        endpoints[key] = refs[0]

    fixed_shape = [None, SEQUENCE_LENGTH, *FRAME_SIZE, 3]
    input_layer["config"]["batch_shape"] = fixed_shape
    for entry in (config, wrapper):
        build_config = entry.get("build_config", {})
        if build_config.get("input_shape") == shape:
            build_config["input_shape"] = fixed_shape

    config.pop("compile_config", None)
    model = keras.models.model_from_json(json.dumps(config))
    model.load_weights(path, skip_mismatch=False)
    return model


@lru_cache(maxsize=1)
def _load_model(path: Path):
    if not path.is_file():
        raise AnalysisError(f"Model file not found: {path.name}. See the setup instructions.")

    try:
        # TensorFlow startup and deserialization can take time on a desktop CPU.
        from tensorflow import keras

        try:
            model = keras.models.load_model(path, compile=False)
        except ValueError:
            model = _load_legacy_model(keras, path)
    except Exception as exc:
        raise AnalysisError(f"Could not load {path.name}. Check the model file and try again.") from exc

    input_shape = tuple(model.inputs[0].shape) if len(model.inputs) == 1 else ()
    if (
        len(input_shape) != 5
        or input_shape[0] not in (None, 1)
        or input_shape[1] not in (None, SEQUENCE_LENGTH)
        or input_shape[2:] != (*FRAME_SIZE, 3)
        or len(model.outputs) != 1
        or tuple(model.outputs[0].shape[1:]) != (1,)
    ):
        raise AnalysisError("The model has an incompatible video input or classification output.")
    return model


def _video_batch(video_path: str | Path) -> tuple[np.ndarray, int]:
    frames = []
    try:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise AnalysisError("Could not open the video. Check the file and its codec.")

            for index in range(1 + (SEQUENCE_LENGTH - 1) * FRAME_STRIDE):
                ok, frame = capture.read()
                if not ok:
                    break
                if index % FRAME_STRIDE == 0:
                    resized = cv2.resize(frame, FRAME_SIZE)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    frames.append(rgb)
        finally:
            capture.release()
    except cv2.error as exc:
        raise AnalysisError("Could not decode the video. Check the file and its codec.") from exc

    if not frames:
        raise AnalysisError("The video contains no readable frames.")

    sampled_frames = len(frames)
    frames.extend([frames[-1]] * (SEQUENCE_LENGTH - sampled_frames))
    batch = np.asarray(frames, dtype=np.float32)[None, ...] / 255.0
    return batch, sampled_frames


def classify_video(
    video_path: str | Path,
    *,
    model_path: str | Path = MODEL_PATH,
    on_stage: Callable[[str], None] | None = None,
) -> VideoAnalysis:
    """Return the model's raw score and the number of frames sampled before padding."""
    if on_stage:
        on_stage("Reading video frames")
    batch, sampled_frames = _video_batch(video_path)

    if on_stage:
        on_stage("Preparing model")
    model = _load_model(Path(model_path).resolve())

    if on_stage:
        on_stage("Running model")
    try:
        scores = np.asarray(model(batch, training=False))
    except Exception as exc:
        raise AnalysisError("The model could not analyze this video.") from exc

    if scores.shape != (1, 1) or not 0 <= scores[0, 0] <= 1:
        raise AnalysisError("The model returned an invalid score.")
    return VideoAnalysis(float(scores[0, 0]), sampled_frames)
