from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np

from config import (
    DEFAULT_COPULA_TAU,
    GPU_CONFIG_SOURCE,
    GPU_DEVICE_ID,
    GPU_ENV_VAR,
)

import tensorflow as tf
from models import load_models
from data_local import get_model_grid_from_local_forcing
from data_earthaccess import (
    fetch_nldas_forb_hourly_to_cache,
    open_forb_hourly_as_xarray,
    to_daily_forcing_on_target_grid,
)
from drought_index import drought_from_zscores


def _require_tf_gpu() -> str:
    tf.config.set_soft_device_placement(False)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return "/GPU:0"
    raise RuntimeError(
        "TensorFlow cannot see a GPU. This app only runs on GPU. "
        "Start the app inside a GPU session, then set "
        f"{GPU_ENV_VAR} to the assigned GPU ID or rely on the session's "
        "CUDA_VISIBLE_DEVICES value. "
        f"Current selection: {GPU_DEVICE_ID!r} from {GPU_CONFIG_SOURCE}."
    )


def _enable_tf_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def _as_np_date(d: Any) -> np.datetime64:
    if isinstance(d, np.datetime64):
        return d.astype("datetime64[D]")
    if isinstance(d, Date):
        return np.datetime64(d.isoformat())
    if isinstance(d, str):
        return np.datetime64(d)
    raise TypeError(f"Unsupported date type: {type(d)}")


def _broadcast_mu_sd(mu: np.ndarray, sd: np.ndarray, C: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu).astype(np.float32)
    sd = np.asarray(sd).astype(np.float32)
    if mu.ndim == 1:
        mu = mu.reshape((1, 1, 1, -1))
    if sd.ndim == 1:
        sd = sd.reshape((1, 1, 1, -1))
    if mu.shape[-1] != C or sd.shape[-1] != C:
        raise ValueError(f"Norm channel mismatch. Expected C={C} but got mu={mu.shape}, sd={sd.shape}")
    return mu, sd


def _model_in_channels(model: tf.keras.Model) -> int:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[-1])


def _model_seq_len(model: tf.keras.Model) -> int:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1])


@lru_cache(maxsize=8)
def _get_loaded_models(model_dir: str):
    return load_models(model_dir)


@lru_cache(maxsize=8)
def _get_model_grid_cached(forcing_dir: str, et_path: str, sm_path: str, model_grid_path: str | None):
    return get_model_grid_from_local_forcing(forcing_dir, et_path, sm_path, model_grid_path=model_grid_path)


def _load_earthaccess_daily_forcing(
    *,
    start_day: np.datetime64,
    as_of: np.datetime64,
    cache_dir: str | None,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    start_iso = f"{str(start_day)}T00:00:00Z"
    # End-exclusive so the as-of day is included.
    end_iso = f"{str((as_of + np.timedelta64(1, 'D')).astype('datetime64[D]'))}T00:00:00Z"

    paths, used = fetch_nldas_forb_hourly_to_cache(
        start_iso=start_iso,
        end_iso=end_iso,
        cache_dir=cache_dir,
    )
    ds_hourly = open_forb_hourly_as_xarray(paths)
    try:
        X_phys, day_keys = to_daily_forcing_on_target_grid(ds_hourly, lat2d, lon2d)
    finally:
        ds_hourly.close()
    return X_phys, day_keys, used


def _predict_single(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    """Run singular model forecasting window and return 2D forecast field."""
    with tf.device(_require_tf_gpu()):
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        y = model(x_tf, training=False)
    y = np.asarray(y)

    if y.ndim == 5:
        y = y[:, -1]
    if y.ndim == 4:
        y = y[0, :, :, 0]
    elif y.ndim == 3:
        y = y[0]
    else:
        raise ValueError(f"Unexpected model output shape: {y.shape}")
    return y.astype(np.float32)


def _build_sm_pred_feature_window(
    *,
    sm_model: tf.keras.Model,
    X_base_norm: np.ndarray,
    sm_y_mu: np.ndarray,
    sm_y_sd: np.ndarray,
    et_in_mu_feat: float,
    seq_len_sm: int,
    horizon: int,
    et_start_idx: int,
    et_end_idx: int,
) -> np.ndarray:
    """Build ET model's soil-moisture forecast feature."""
    t_indices = np.arange(et_start_idx, et_end_idx + 1, dtype=np.int64)
    end_indices = t_indices - horizon

    valid = end_indices >= (seq_len_sm - 1)

    H, W = X_base_norm.shape[1], X_base_norm.shape[2]
    sm_feat = np.full((t_indices.size, H, W), float(et_in_mu_feat), dtype=np.float32)

    if not np.any(valid):
        return sm_feat

    starts = end_indices[valid] - (seq_len_sm - 1)
    ends = end_indices[valid]

    xb_list = []
    for s, e in zip(starts, ends):
        xb_list.append(X_base_norm[s : e + 1])
    xb = np.stack(xb_list, axis=0).astype(np.float32)

    with tf.device(_require_tf_gpu()):
        x_tf = tf.convert_to_tensor(xb, dtype=tf.float32)
        y = sm_model(x_tf, training=False)
    y = np.asarray(y)

    if y.ndim == 5:
        y = y[:, -1]
    if y.ndim == 4:
        y = y[..., 0]
    elif y.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected SM model output shape: {y.shape}")

    sm_y_mu = np.asarray(sm_y_mu).astype(np.float32).reshape(-1)[0]
    sm_y_sd = np.asarray(sm_y_sd).astype(np.float32).reshape(-1)[0]
    y_phys = y.astype(np.float32) * sm_y_sd + sm_y_mu

    sm_feat[valid] = y_phys
    return sm_feat


@dataclass
class ForecastResult:
    target_day: np.datetime64
    horizon_days: int
    lat2d: np.ndarray
    lon2d: np.ndarray
    sm_pred: np.ndarray
    et_pred: np.ndarray
    z_sm: np.ndarray
    z_et: np.ndarray
    pdry: np.ndarray
    category: np.ndarray
    debug: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def run_forecast(
    *,
    bundle: Dict[str, Any],
    timescale: str,
    as_of_day: Any,
    history_days: int = 140,
    tau: float = DEFAULT_COPULA_TAU,
    drought_sensitivity: float = 1.0,
) -> ForecastResult:
    """Retrieve NLDAS Forcing + run inference +  classification via drought index."""
    _enable_tf_gpu_memory_growth()
    tf_device = _require_tf_gpu()

    model_dir = bundle["model_dir"]
    forcing_dir = bundle.get("forcing_dir")
    et_path = bundle.get("et_path")
    sm_path = bundle.get("sm_path")
    model_grid_path = bundle.get("model_grid_path")
    cache_dir = bundle.get("cache_dir")

    if forcing_dir is None or et_path is None or sm_path is None:
        raise ValueError("bundle must include forcing_dir, et_path, sm_path")

    as_of = _as_np_date(as_of_day)

    loaded = _get_loaded_models(model_dir)
    sm_model = loaded.sm_model
    et_model = loaded.et_model
    sm_norms = loaded.sm_norms
    et_norms = loaded.et_norms

    seq_len_sm = _model_seq_len(sm_model)
    seq_len_et = _model_seq_len(et_model)

    C_sm = _model_in_channels(sm_model)
    C_et = _model_in_channels(et_model)

    if C_sm != 7:
        raise ValueError(f"SM model expected 7 channels but got {C_sm}")

    horizon_days = int(bundle["horizon_days"])

    # Include enough history for the SM window and any ET soil-moisture feature window.
    min_needed = seq_len_et + seq_len_sm + horizon_days + 10
    hist = max(int(history_days), int(min_needed))

    lat2d, lon2d = _get_model_grid_cached(forcing_dir, et_path, sm_path, model_grid_path)

    start_day = (as_of - np.timedelta64(hist - 1, "D")).astype("datetime64[D]")
    X_base_phys, day_keys, used = _load_earthaccess_daily_forcing(
        start_day=start_day,
        as_of=as_of,
        cache_dir=cache_dir,
        lat2d=lat2d,
        lon2d=lon2d,
    )
    forcing_source = "earthaccess_hourly_resampled_daily"
    forcing_meta: Dict[str, Any] = {"earthaccess_used": used}

    if day_keys.size == 0:
        raise RuntimeError("No daily forcing days returned for  requested range.")

    idx_asof = np.where(day_keys == as_of)[0]
    if idx_asof.size == 0:
        raise RuntimeError(
            f"as_of_day {as_of} not present in Earthaccess daily forcing. "
            f"Available range: {day_keys[0]} .. {day_keys[-1]}"
        )
    idx_asof = int(idx_asof[-1])

    sm_in_mu, sm_in_sd = _broadcast_mu_sd(sm_norms["input_mu"], sm_norms["input_sd"], 7)
    sm_y_mu = sm_norms["target_mu"]
    sm_y_sd = sm_norms["target_sd"]

    X_base_norm_for_sm = (X_base_phys - sm_in_mu) / sm_in_sd
    X_base_norm_for_sm = np.nan_to_num(X_base_norm_for_sm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    sm_start = idx_asof - (seq_len_sm - 1)
    if sm_start < 0:
        raise RuntimeError(f"Not enough history for SM model: need {seq_len_sm} days before {as_of}")

    xb_sm = X_base_norm_for_sm[sm_start : idx_asof + 1][None, ...]
    sm_pred_norm = _predict_single(sm_model, xb_sm)
    sm_y_mu0 = np.asarray(sm_y_mu).reshape(-1)[0]
    sm_y_sd0 = np.asarray(sm_y_sd).reshape(-1)[0]
    sm_pred_phys = sm_pred_norm * float(sm_y_sd0) + float(sm_y_mu0)

    target_day = (as_of + np.timedelta64(horizon_days, "D")).astype("datetime64[D]")

    et_start = idx_asof - (seq_len_et - 1)
    if et_start < 0:
        raise RuntimeError(f"Not enough history for ET model: need {seq_len_et} days before {as_of}")

    X_for_et_phys = X_base_phys[et_start : idx_asof + 1]

    et_in_mu = et_norms["input_mu"]
    et_in_sd = et_norms["input_sd"]
    et_t_mu = et_norms["target_mu"]
    et_t_sd = et_norms["target_sd"]

    if C_et == 7:
        X_et_phys = X_for_et_phys
        et_in_mu_b, et_in_sd_b = _broadcast_mu_sd(et_in_mu, et_in_sd, 7)
    elif C_et == 8:
        et_in_mu_b8, et_in_sd_b8 = _broadcast_mu_sd(et_in_mu, et_in_sd, 8)
        et_feat_mu = float(et_in_mu_b8.reshape(-1)[7])

        sm_feat_window = _build_sm_pred_feature_window(
            sm_model=sm_model,
            X_base_norm=X_base_norm_for_sm,
            sm_y_mu=sm_y_mu,
            sm_y_sd=sm_y_sd,
            et_in_mu_feat=et_feat_mu,
            seq_len_sm=seq_len_sm,
            horizon=horizon_days,
            et_start_idx=et_start,
            et_end_idx=idx_asof,
        )

        X_et_phys = np.concatenate([X_for_et_phys, sm_feat_window[..., None]], axis=-1).astype(np.float32)
        et_in_mu_b, et_in_sd_b = et_in_mu_b8, et_in_sd_b8
    else:
        raise ValueError(f"ET model expected 7 or 8 channels but got {C_et}")

    X_et_norm = (X_et_phys - et_in_mu_b) / et_in_sd_b
    X_et_norm = np.nan_to_num(X_et_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xb_et = X_et_norm[None, ...]

    et_pred_norm = _predict_single(et_model, xb_et)

    et_t_mu0 = np.asarray(et_t_mu).reshape(-1)[0]
    et_t_sd0 = np.asarray(et_t_sd).reshape(-1)[0]
    et_pred_phys = et_pred_norm * float(et_t_sd0) + float(et_t_mu0)

    z_sm = (sm_pred_phys - float(sm_y_mu0)) / float(sm_y_sd0)
    z_et = (et_pred_phys - float(et_t_mu0)) / float(et_t_sd0)

    pdry, cat = drought_from_zscores(
        z_et=z_et,
        z_sm=z_sm,
        tau=tau,
        sensitivity=drought_sensitivity,
    )

    debug = {
        "as_of_day": str(as_of),
        "target_day": str(target_day),
        "horizon_days": horizon_days,
        "model_dir": model_dir,
        "forcing_dir": forcing_dir,
        "model_grid_path": model_grid_path,
        "forcing_source": forcing_source,
        "forcing_start_day": str(start_day),
        "forcing_end_day": str(as_of),
        "forcing_meta": forcing_meta,
        "seq_len_sm": seq_len_sm,
        "seq_len_et": seq_len_et,
        "C_sm": C_sm,
        "C_et": C_et,
        "history_days_used": hist,
        "tau": float(tau),
        "drought_sensitivity": float(drought_sensitivity),
        "configured_system_gpu": GPU_DEVICE_ID,
        "gpu_config_source": GPU_CONFIG_SOURCE,
        "tensorflow_device": tf_device,
        "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
    }

    return ForecastResult(
        target_day=target_day,
        horizon_days=horizon_days,
        lat2d=lat2d,
        lon2d=lon2d,
        sm_pred=sm_pred_phys.astype(np.float32),
        et_pred=et_pred_phys.astype(np.float32),
        z_sm=z_sm.astype(np.float32),
        z_et=z_et.astype(np.float32),
        pdry=pdry.astype(np.float32),
        category=cat.astype(np.int8),
        debug=debug,
    )
