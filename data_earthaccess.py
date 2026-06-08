from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr


# FORA includes LWdown for FLDS. FORB can be used with FLDS filled as zero.
DEFAULT_NLDAS_SHORTNAME = os.environ.get("NLDAS_SHORTNAME", "NLDAS_FORA0125_H")

NEEDED_NLDAS_VARS = ["SWdown", "Rainf", "Tair", "Qair", "PSurf", "Wind_E", "Wind_N"]


def _parse_iso_to_utc_dt(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _xr_open_dataset_safe(path: str, **kwargs) -> xr.Dataset:
    """Open NetCDF with fallback engines."""
    engines = [None, "netcdf4", "h5netcdf", "scipy"]
    last_err = None
    for eng in engines:
        try:
            if eng is None:
                return xr.open_dataset(path, **kwargs)
            return xr.open_dataset(path, engine=eng, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to open {path} with xarray. Last error: {last_err}")


def _open_many_hourly(paths: List[str]) -> xr.Dataset:
    """Open and concatenate hourly NLDAS granules."""
    dsets = []
    for p in paths:
        ds = _xr_open_dataset_safe(p, decode_cf=True)
        dsets.append(ds)

    ds_all = xr.concat(dsets, dim="time", data_vars="minimal", coords="minimal", compat="override")

    for ds in dsets:
        try:
            ds.close()
        except Exception:
            pass

    return ds_all


def _subset_bbox(ds: xr.Dataset, lat2d: np.ndarray, lon2d: np.ndarray, pad_deg: float = 0.5) -> xr.Dataset:
    """Subset NLDAS on native 1D lat/lon bounds."""
    lat_min = float(np.nanmin(lat2d)) - pad_deg
    lat_max = float(np.nanmax(lat2d)) + pad_deg
    lon_min = float(np.nanmin(lon2d)) - pad_deg
    lon_max = float(np.nanmax(lon2d)) + pad_deg

    # Match the dataset longitude convention before slicing.
    ds_lon = ds["lon"].values
    if np.nanmax(ds_lon) <= 180.0 and lon_min > 180.0:
        lon_min -= 360.0
        lon_max -= 360.0
    elif np.nanmax(ds_lon) > 180.0 and lon_min < 0.0:
        lon_min += 360.0
        lon_max += 360.0

    return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def _precompute_bilinear_weights(
    lat_src_1d: np.ndarray,
    lon_src_1d: np.ndarray,
    lat_tgt_2d: np.ndarray,
    lon_tgt_2d: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Precompute bilinear weights from source grid to target grid."""
    lat_src = np.asarray(lat_src_1d).astype(np.float64)
    lon_src = np.asarray(lon_src_1d).astype(np.float64)

    lat_flip = False
    lon_flip = False
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        lat_flip = True
    if lon_src[0] > lon_src[-1]:
        lon_src = lon_src[::-1]
        lon_flip = True

    lat_t = np.asarray(lat_tgt_2d).astype(np.float64)
    lon_t = np.asarray(lon_tgt_2d).astype(np.float64)

    i1 = np.searchsorted(lat_src, lat_t, side="right")
    j1 = np.searchsorted(lon_src, lon_t, side="right")

    i1 = np.clip(i1, 1, lat_src.size - 1)
    j1 = np.clip(j1, 1, lon_src.size - 1)

    i0 = i1 - 1
    j0 = j1 - 1

    lat0 = lat_src[i0]
    lat1 = lat_src[i1]
    lon0 = lon_src[j0]
    lon1 = lon_src[j1]

    denom_lat = np.where((lat1 - lat0) == 0, 1.0, (lat1 - lat0))
    denom_lon = np.where((lon1 - lon0) == 0, 1.0, (lon1 - lon0))

    wy = (lat_t - lat0) / denom_lat
    wx = (lon_t - lon0) / denom_lon

    w00 = (1 - wy) * (1 - wx)
    w01 = (1 - wy) * wx
    w10 = wy * (1 - wx)
    w11 = wy * wx

    return {
        "i0": i0.astype(np.int64),
        "i1": i1.astype(np.int64),
        "j0": j0.astype(np.int64),
        "j1": j1.astype(np.int64),
        "w00": w00.astype(np.float32),
        "w01": w01.astype(np.float32),
        "w10": w10.astype(np.float32),
        "w11": w11.astype(np.float32),
        "lat_flip": np.array(lat_flip),
        "lon_flip": np.array(lon_flip),
    }


def _bilinear_regrid_3d(data_tyx: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply precomputed bilinear weights to a (time, y, x) array."""
    x = data_tyx
    if bool(w["lat_flip"]):
        x = x[:, ::-1, :]
    if bool(w["lon_flip"]):
        x = x[:, :, ::-1]

    i0, i1, j0, j1 = w["i0"], w["i1"], w["j0"], w["j1"]
    w00, w01, w10, w11 = w["w00"], w["w01"], w["w10"], w["w11"]

    v00 = x[:, i0, j0]
    v01 = x[:, i0, j1]
    v10 = x[:, i1, j0]
    v11 = x[:, i1, j1]

    return (v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11).astype(np.float32)


def fetch_nldas_forb_hourly_to_cache(
    start_iso: str,
    end_iso: str,
    cache_dir: Optional[str] = None,
    short_name: Optional[str] = None,
    quiet: bool = True,
) -> Tuple[List[str], Dict]:
    """Download hourly NLDAS granules to cache."""
    default_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NLDAS_Cache"))
    cache_dir = cache_dir or os.environ.get("NLDAS_CACHE_DIR", default_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    short_name = short_name or DEFAULT_NLDAS_SHORTNAME

    import earthaccess

    try:
        earthaccess.login(persist=True)
    except Exception:
        pass

    t0 = _parse_iso_to_utc_dt(start_iso)
    t1 = _parse_iso_to_utc_dt(end_iso)

    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(t0.isoformat(), t1.isoformat()),
        count=-1,
    )
    if not results:
        raise RuntimeError(f"No NLDAS granules found for {short_name} in {t0} .. {t1}")

    downloaded = earthaccess.download(results, local_path=cache_dir)

    paths = []
    for p in downloaded:
        if not p:
            continue
        paths.append(os.path.abspath(p))

    used = {
        "short_name": short_name,
        "cache_dir": os.path.abspath(cache_dir),
        "n_granules": len(results),
        "n_downloaded_paths": len(paths),
        "start_iso": start_iso,
        "end_iso": end_iso,
    }
    return paths, used


def open_forb_hourly_as_xarray(paths: List[str]) -> xr.Dataset:
    """Open downloaded FORB hourly files."""
    if not paths:
        raise ValueError("open_forb_hourly_as_xarray retrieved empty paths list")
    return _open_many_hourly(sorted(paths))


def to_daily_forcing_on_target_grid(
    ds_hourly: xr.Dataset,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    pad_deg: float = 0.5,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert hourly NLDAS forcing to daily data on the target grid. Acceptable format to feed into model"""
    for v in NEEDED_NLDAS_VARS:
        if v not in ds_hourly:
            raise KeyError(f"Missing {v} in NLDAS dataset. Available: {list(ds_hourly.data_vars)}")

    has_lwdown = "LWdown" in ds_hourly

    ds = _subset_bbox(ds_hourly, lat2d, lon2d, pad_deg=pad_deg)

    wind_mag = np.sqrt(ds["Wind_E"] ** 2 + ds["Wind_N"] ** 2)

    rain_units = (ds["Rainf"].attrs.get("units") or "").lower()
    rain_is_rate = ("s-1" in rain_units) or ("/s" in rain_units)

    ds_mean = xr.Dataset(
        {
            "SWdown": ds["SWdown"],
            "Tair": ds["Tair"],
            "Qair": ds["Qair"],
            "PSurf": ds["PSurf"],
            "WIND": wind_mag,
        }
    )
    if has_lwdown:
        ds_mean["LWdown"] = ds["LWdown"]

    ds_mean = ds_mean.resample(time="1D").mean(keep_attrs=True)

    if rain_is_rate:
        ds_rain = xr.Dataset({"Rainf": ds["Rainf"]}).resample(time="1D").mean(keep_attrs=True)
        prect_mms = ds_rain["Rainf"].astype("float32")
    else:
        ds_rain = xr.Dataset({"Rainf": ds["Rainf"]}).resample(time="1D").sum(keep_attrs=True)
        prect_mms = (ds_rain["Rainf"] / 86400.0).astype("float32")

    ds_daily = xr.merge([ds_mean, ds_rain])

    lat_src = ds_daily["lat"].values
    lon_src = ds_daily["lon"].values
    w = _precompute_bilinear_weights(lat_src, lon_src, lat2d, lon2d)

    def rg(var_da: xr.DataArray) -> np.ndarray:
        arr = var_da.transpose("time", "lat", "lon").astype("float32").values
        return _bilinear_regrid_3d(arr, w)

    FSDS = rg(ds_daily["SWdown"])
    TBOT = rg(ds_daily["Tair"])
    QBOT = rg(ds_daily["Qair"])
    PSRF = rg(ds_daily["PSurf"])
    WIND = rg(ds_daily["WIND"])

    if has_lwdown:
        FLDS = rg(ds_daily["LWdown"])
    else:
        FLDS = np.zeros_like(FSDS, dtype=np.float32)

    PRECT = _bilinear_regrid_3d(
        prect_mms.transpose("time", "lat", "lon").values.astype("float32"),
        w,
    )

    X = np.stack([PRECT, TBOT, WIND, QBOT, PSRF, FSDS, FLDS], axis=-1).astype(np.float32)
    day_keys = ds_daily["time"].values.astype("datetime64[D]")

    return X, day_keys
