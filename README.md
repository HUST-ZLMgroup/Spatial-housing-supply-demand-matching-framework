# Spatial housing supply-demand matching framework

**Quantifying Housing Vacancy, Supply-Demand Mismatch, and Associated Carbon Emissions Across China Using Multi-Source Geospatial Data**

---

## Overview

This repository contains the full analysis pipeline for estimating building-scale housing vacancy rates (HVR), quantifying housing supply-demand spatial mismatch (SDMI), projecting future construction carbon emissions under SSP scenarios, and identifying key drivers through causal and sensitivity analysis.

The study covers mainland China (31 provinces, 367 cities) from 2013 to 2023, with future projections to 2050 under five SSP scenarios.

---

## Repository Structure

```
housing-vacancy-carbon/
│
├── configs/
│   ├── base_config.yaml        # Paths and parameters for modules 00–02
│   └── ssp_config.yaml         # SSP scenario paths, carbon intensity table, XGBoost params
│
├── data/
│   ├── Data source.txt         # Data sources and download instructions
│
├── src/
│   ├── 00_preprocess/          # Building data preprocessing
│   ├── 01_hvr/                 # Housing Vacancy Rate estimation
│   ├── 02_sdmi/                # Supply-demand mismatch
│   ├── 03_carbon/              # Carbon emission projection
│   ├── 04_causal/              # Causal analysis & sensitivity
│   ├── 05_optimization/        # Multi-objective optimization 
│   └── utils/                  # Shared utilities (ArcPy env management)
│
├── scripts/                    # Entry-point scripts (run in order)
│   ├── run_00_preprocess.py
│   ├── run_01_hvr.py
│   ├── run_02_sdmi.py
│   ├── run_03_carbon.py
│   ├── run_04_causal.py
│   └── run_05_optimization.py
│
├── result/                  
├── .gitignore
└── README.md
```


### Spatial modules (00–02, require ArcGIS Pro)

```bash
# Activate the ArcGIS Pro conda environment
conda activate arcgispro-py3

# Install additional packages into the ArcPy environment
pip install lingam SALib pymoo openpyxl PyYAML xgboost scikit-learn
```

---

## Configuration

Before running, edit the two config files to match your local paths:

**`configs/base_config.yaml`** — paths for raw data, building GDBs, boundaries, and output directories for modules 00–02.

**`configs/ssp_config.yaml`** — SSP scenario CSV directories, model output paths, carbon intensity lookup table, and XGBoost hyperparameters for modules 03–05.

Both files are annotated; search for paths starting with `data/` or `output/` and update them to your actual locations.

---

## Running the Pipeline

Run modules in order. Each script accepts `--config` to override the default config path.

### Step 0 — Building data preprocessing
```bash
python scripts/run_00_preprocess.py                        # All provinces
python scripts/run_00_preprocess.py --provinces 广东 湖北  # Specific provinces
```

### Step 1 — Housing Vacancy Rate (HVR)
```bash
python scripts/run_01_hvr.py                               # All provinces × all years
python scripts/run_01_hvr.py --years 2020 2021             # Specific years
python scripts/run_01_hvr.py --provinces 广东 --years 2023
```

### Step 2 — Supply-Demand Mismatch (SDMI)
```bash
python scripts/run_02_sdmi.py                  # All 6 steps, all years
python scripts/run_02_sdmi.py --steps 1 2 3    # Only rigid/improvement/renewal demand
python scripts/run_02_sdmi.py --years 2020 2021
```

Steps: `1` rigid demand, `2` improvement demand, `3` renewal demand, `4` merge demand, `5` potential supply, `6` SDMI index.

### Step 3 — Carbon Emission Projection
```bash
# XGBoost forecasting (run for each task and scenario)
python scripts/run_03_carbon.py --task demand    --scenario ssp2
python scripts/run_03_carbon.py --task supply    --scenario ssp2
python scripts/run_03_carbon.py --task new_area  --scenario ssp2

# Compute new_supply = max(supply - old_supply, 0) via ArcPy raster
python scripts/run_03_carbon.py --task new_supply

# Calculate carbon emissions (Eq.16, 5-year rolling window)
python scripts/run_03_carbon.py --task carbon    --scenario ssp2

# Replay prediction with saved pkl (no retraining)
python scripts/run_03_carbon.py --task demand --scenario ssp2 --replay
```

### Step 4 — Causal Analysis & Sensitivity
```bash
# DirectLiNGAM causal discovery (all SSP × year combinations)
python scripts/run_04_causal.py --task lingam

# One-at-a-time sensitivity analysis (Eq.15)
python scripts/run_04_causal.py --task ovat

# Sobol global sensitivity analysis (Eqs.16-17)
python scripts/run_04_causal.py --task sobol --ssp ssp2   # Single scenario
python scripts/run_04_causal.py --task sobol               # All scenarios
```

### Step 5 — Multi-objective Optimization (NSGA-II)
```bash
# Single governance strategy
python scripts/run_05_optimization.py --strategy cooperative   --scenario ssp2
python scripts/run_05_optimization.py --strategy infrastructure --scenario ssp1

# All 5 strategies in sequence
python scripts/run_05_optimization.py --strategy all --scenario ssp2

# Compare policy start years
python scripts/run_05_optimization.py --strategy cooperative --scenario ssp2 --ramp_start 2026
python scripts/run_05_optimization.py --strategy cooperative --scenario ssp2 --ramp_start 2031
```

Available strategies: `infrastructure`, `social`, `economic`, `environment`, `cooperative`.

---


## Data

All input data must be downloaded before running. See [`data/Data source.txt`](data/Data source.txt) for complete source URLs, temporal coverage, spatial resolution, and directory structure for each dataset.

Key datasets: CMAB building footprints, NPP-VIIRS nighttime light, OpenStreetMap POI/roads, LandScan population, and 42 socioeconomic variables from multiple sources.

---

## Citation

If you use this code, please cite the associated paper (citation to be added upon publication).

