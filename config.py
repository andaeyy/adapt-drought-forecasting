from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass(frozen=True)
class TimescaleSpec:
    name: str
    parent_dirs: List[str]
    best_arch_folder: str
    horizon_days: int


APP_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
BASE_DIR = os.environ.get("NLDAS_BASE_DIR", REPO_ROOT)
STREAMLIT_MODEL_ARTIFACTS_DIR = os.environ.get(
    "STREAMLIT_MODEL_ARTIFACTS_DIR",
    os.path.join(APP_DIR, "model_artifacts"),
)
MODEL_GRID_PATH = os.environ.get(
    "MODEL_GRID_PATH",
    os.path.join(STREAMLIT_MODEL_ARTIFACTS_DIR, "grid", "model_grid.npz"),
)
FORCING_DIR = os.environ.get("NLDAS_FORCING_DIR", os.path.join(BASE_DIR, "yearly"))

# ELM target grids
ELM_ET_PATH = os.environ.get("ELM_ET_PATH", os.path.join(BASE_DIR, "ELM_EVAPOTRANSPIRATION_2000_2019.nc"))
ELM_SM_PATH = os.environ.get("ELM_SM_PATH", os.path.join(BASE_DIR, "ELM_SM_2000-2019.nc"))

NLDAS_CACHE_DIR = os.environ.get("NLDAS_CACHE_DIR", os.path.join(BASE_DIR, "droughtapp", "droughtapp_cache"))
os.makedirs(NLDAS_CACHE_DIR, exist_ok=True)

DEFAULT_HISTORY_DAYS = int(os.environ.get("DEFAULT_HISTORY_DAYS", "140"))
DEFAULT_COPULA_TAU = float(os.environ.get("DEFAULT_COPULA_TAU", "0.40"))


# Best-performing model depending on forecast horizon
TIMESCALES: Dict[str, TimescaleSpec] = {
    "Weekly": TimescaleSpec(
        name="Weekly",
        parent_dirs=[os.path.join(STREAMLIT_MODEL_ARTIFACTS_DIR, "Weekly")],
        best_arch_folder="Seq2seqconvlstm",
        horizon_days=7,
    ),
    "Monthly": TimescaleSpec(
        name="Monthly",
        parent_dirs=[os.path.join(STREAMLIT_MODEL_ARTIFACTS_DIR, "Monthly")],
        best_arch_folder="DEconvlstm",
        horizon_days=30,
    ),
    "Seasonal": TimescaleSpec(
        name="Seasonal",
        parent_dirs=[os.path.join(STREAMLIT_MODEL_ARTIFACTS_DIR, "Seasonal")],
        best_arch_folder="DEconvlstm",
        horizon_days=90,
    ),
}

# Other compatible file names depending on app version
ET_FILE_DEFAULT = ELM_ET_PATH
SM_FILE_DEFAULT = ELM_SM_PATH
APP_CACHE_DIR = NLDAS_CACHE_DIR

TIMESCALE_TO_HORIZON_DAYS = {k: v.horizon_days for k, v in TIMESCALES.items()}

DEFAULT_BUNDLE_KEY = "default"
BUNDLES: Dict[str, Dict[str, Any]] = {
    DEFAULT_BUNDLE_KEY: {
        "forcing_dir": FORCING_DIR,
        "et_path": ELM_ET_PATH,
        "sm_path": ELM_SM_PATH,
        "model_grid_path": MODEL_GRID_PATH,
        "cache_dir": NLDAS_CACHE_DIR,
    }
}
