# ADAPT Drought Forecasting Streamlit App

This repository contains a Streamlit app for running ADAPT drought forecasts with bundled model artifacts, model normalizations, and a precomputed model grid.

The app downloads recent NLDAS forcing data at runtime through NASA Earthdata using `earthaccess`. Each user must authenticate with their own NASA Earthdata account.

## Requirements

- Python 3.10 or newer
- Git LFS, needed to download the files in `model_artifacts/`
- A free NASA Earthdata account: <https://urs.earthdata.nasa.gov/users/new>

## Clone

Install Git LFS before cloning or immediately after:

```bash
git lfs install
git clone <repo-url>
cd <repo-folder>
git lfs pull
```

Check that model files were downloaded correctly:

```bash
git lfs ls-files
find model_artifacts -type f
```

## Python Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Earthdata Login

Run this once on your own machine:

```bash
python -c "import earthaccess; earthaccess.login(persist=True)"
```

This stores your Earthdata credentials locally. The repository does not include or use the project author's credentials.

## Run

```bash
streamlit run app.py
```

or:

```bash
./run.sh
```

The first forecast run will download NLDAS forcing files into a local cache. Later runs reuse cached files when possible.

## Bundled Artifacts

Runtime model assets live in `model_artifacts/`:

- `Weekly/Seq2seqconvlstm`
- `Monthly/DEconvlstm`
- `Seasonal/DEconvlstm`
- `grid/model_grid.npz`

All files under `model_artifacts/` are tracked with Git LFS.

`grid/model_grid.npz` is the extracted grid needed at runtime. The original raw files are not required for normal app use:

- `yearly/clmforc.nldas.YYYY.nc`
- `ELM_EVAPOTRANSPIRATION_2000_2019.nc`
- `ELM_SM_2000-2019.nc`

If `model_artifacts/grid/model_grid.npz` is missing, the app can still reconstruct the grid from those raw NetCDF files when `NLDAS_BASE_DIR`, `NLDAS_FORCING_DIR`, `ELM_ET_PATH`, and `ELM_SM_PATH` point to valid local data.

## Configuration

The defaults should work after cloning with Git LFS and logging into Earthdata. These environment variables are available for custom deployments:

- `STREAMLIT_MODEL_ARTIFACTS_DIR`: alternate model artifact directory
- `MODEL_GRID_PATH`: alternate precomputed grid `.npz`
- `NLDAS_CACHE_DIR`: local cache for Earthdata downloads
- `NLDAS_BASE_DIR`, `NLDAS_FORCING_DIR`, `ELM_ET_PATH`, `ELM_SM_PATH`: fallback raw-data paths for rebuilding the model grid

Example:

```bash
export NLDAS_CACHE_DIR="$PWD/droughtapp_cache"
streamlit run app.py
```

## Troubleshooting

- If model loading fails, run `git lfs pull` and confirm the files in `model_artifacts/` are not tiny pointer files.
- If Earthdata download fails, re-run `python -c "import earthaccess; earthaccess.login(persist=True)"`.
- If TensorFlow is slow or no GPU is available, the app will still run on CPU, but forecasts may take longer.
- If Streamlit starts but the browser cannot connect on a remote server, use `./run.sh` or pass `--server.address 0.0.0.0`.
