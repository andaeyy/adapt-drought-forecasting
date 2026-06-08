from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="Custom")
class TakeLastTimestep(layers.Layer):
    def call(self, x):
        return x[:, -1]

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="Custom")
class TileToHorizon(layers.Layer):
    def __init__(self, horizon=7, **kwargs):
        super().__init__(**kwargs)
        self.horizon = int(horizon)

    def call(self, x):
        return tf.tile(tf.expand_dims(x, axis=1), [1, self.horizon, 1, 1, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"horizon": self.horizon})
        return cfg


CUSTOM_OBJECTS = {
    "TakeLastTimestep": TakeLastTimestep,
    "Custom>TakeLastTimestep": TakeLastTimestep,
    "TileToHorizon": TileToHorizon,
    "Custom>TileToHorizon": TileToHorizon,
}


@dataclass
class LoadedModelBundle:
    sm_model: keras.Model
    et_model: keras.Model
    sm_norms: Dict[str, Any]
    et_norms: Dict[str, Any]
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def resolve_timescale_model_dir(base_dir: str, parent_dirs: list[str], best_arch_folder: str) -> str:
    """Find the model location for specific timescale."""
    for p in parent_dirs:
        cand = os.path.join(base_dir, p, best_arch_folder)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find model folder. Tried: "
        + ", ".join(os.path.join(base_dir, p, best_arch_folder) for p in parent_dirs)
    )


def _load_norms(path: str) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def load_models(model_dir: str) -> LoadedModelBundle:
    """Load models and normalization files."""
    sm_model_path = os.path.join(model_dir, "keras_convlstm_sm_best.keras")
    et_model_path = os.path.join(model_dir, "keras_convlstm_et_best.keras")
    sm_norms_path = os.path.join(model_dir, "keras_convlstm_sm_norms.npz")
    et_norms_path = os.path.join(model_dir, "keras_convlstm_et_norms.npz")

    for p in [sm_model_path, et_model_path, sm_norms_path, et_norms_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    sm_model = keras.models.load_model(sm_model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
    et_model = keras.models.load_model(et_model_path, compile=False, custom_objects=CUSTOM_OBJECTS)

    sm_norms = _load_norms(sm_norms_path)
    et_norms = _load_norms(et_norms_path)

    return LoadedModelBundle(
        sm_model=sm_model,
        et_model=et_model,
        sm_norms=sm_norms,
        et_norms=et_norms,
    )
