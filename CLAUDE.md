# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python/R scientific computing project focused on **Functional Simultaneous Prediction Bands (FSPB)**. The project implements and compares different algorithms for generating confidence and prediction bands for functional data, with particular focus on fair, minimum width, and conformal inference approaches.

## Development Environment

The project uses **Pixi** as the primary package manager and task runner. The environment includes:
- Python 3.13 with scientific computing libraries (NumPy, SciPy, pandas, scikit-learn, matplotlib, JAX)
- R 4.4.3 with renv for R package management
- Optimization libraries (optimagic, cyipopt, nlopt)
- Development tools (pytest, mypy, ruff, pre-commit)

### Initial Setup

**IMPORTANT**: After cloning the repository, install the R package first:
```bash
pixi run R
```
Then in the R console:
```R
install.packages("conformalInference.fd", repos="https://cloud.r-project.org")
```

**Platform Support**: Currently only Linux and Intel-based MacOS are supported.

## Key Commands

### Testing
- `pixi run pytest` - Run all Python tests
- `pixi run pytest tests/` - Run tests in tests directory
- `pixi run pytest tests/bands/test_bands.py` - Run specific test file
- `pixi run pytest -k "test_name"` - Run specific test by name pattern
- `pixi run pytest --pdb` - Drop into debugger on test failure (uses pdbp)

### Development Tasks (pytask)
- `pixi run pytask` - Run the full computational pipeline (reproduces all results)
- `pixi run pytask clean` - Clean all build artifacts in `bld/`
- `pixi run pytask collect` - List all available tasks without running them
- `pixi run pytask dag` - Visualize task dependency graph
- Tasks are defined in `task_*.py` files in `src/fspb/`

### Code Quality
- `pixi run mypy src` - Type check source code (strict settings enabled)
- `pixi run ruff check src` - Lint and auto-fix code style issues
- `pixi run pre-commit run --all-files` - Run all pre-commit hooks manually

## Architecture

### Core Components

#### 1. Band Generation (`src/fspb/bands/`)
- **`band.py`**: Main `Band` class with `fit()` method, coverage checking, and width statistics
- **`critical_values.py`**: Core algorithm to solve for critical values (fair and minimum width methods)
- **`linear_model.py`**: `ConcurrentLinearModel` - concurrent linear regression for functional data
- **`covariance.py`**: Covariance estimation (stationary vs non-stationary, confidence vs prediction)
- **`dof.py`**: Degrees of freedom estimation for distributional assumptions
- **`roughness.py`**: Roughness penalty calculations for regularization

#### 2. Simulation Framework (`src/fspb/simulation/`)
- **`simulation_study.py`**: Main simulation orchestration
- **`model_simulation.py`**: Data generation and model simulation
- **`processing.py`**: Post-processing of simulation results
- **`results_tables.py`**: LaTeX table generation

#### 3. Configuration (`src/fspb/config.py`)
- Path definitions for build directories
- Scenario configurations for different simulation setups
- Global constants (N_SIMULATIONS, N_JOBS, etc.)

#### 4. Type System (`src/fspb/types.py`)
- Enums for band types, estimation methods, covariance types
- Utility functions for enum parsing

### Task Pipeline

The project uses **pytask** for orchestrating computational tasks. Tasks are defined in `task_*.py` files in `src/fspb/`:

- **`task_simulation_study.py`**: Main simulation execution. Generates tasks dynamically for each scenario and estimation method (FAIR, MIN_WIDTH, CI). Exports data to JSON for R and runs conformal inference via pytask-r integration.
- **`task_process_simulation_results.py`**: Aggregates simulation results across methods
- **`task_produce_results_tables.py`**: Generates LaTeX tables from processed results
- **`task_visualize_band_simulation.py`** / **`task_visualize_band_application.py`**: Band visualizations
- **`task_visualize_outcome.py`** / **`task_visualize_application_outcomes.py`**: Outcome visualizations
- **`task_clean_application_data.py`**: Application data cleaning
- **`task_move_to_paper_dir.py`**: Copies results to paper directory (hostname-specific, only runs on "thinky")

### Build Structure

```
bld/
├── simulation/          # Simulation outputs
│   ├── data/           # Raw simulation data (JSON format for R)
│   ├── ci/             # Conformal inference results from R
│   ├── fair/           # Fair algorithm results (Python)
│   ├── min_width/      # Minimum width results (Python)
│   └── processed/      # Aggregated results across all methods
├── application/        # Application-specific data and results
├── figures/            # Generated plots and visualizations
└── tables/             # LaTeX tables for papers
```

Results are organized by scenario strings like `n=30-d=5-c=stationary-b=prediction` where:
- `n` = number of samples
- `d` = degrees of freedom
- `c` = covariance type (stationary/non_stationary)
- `b` = band type (confidence/prediction)

## Key Concepts

### Scenario Configuration
Scenarios are defined by:
- **n_samples**: Number of samples (30, 100)
- **dof**: Degrees of freedom (5, 15)
- **covariance_type**: STATIONARY or NON_STATIONARY
- **band_type**: CONFIDENCE or PREDICTION

### Estimation Methods
- **FAIR**: Fair critical value selection
- **MIN_WIDTH**: Minimum width optimization
- **CI**: Conformal inference

### Dual Language Setup
- **Python**: Main computational work, band algorithms, simulation orchestration
- **R**: Conformal inference implementation (`src/fspb/R/conformal_prediction.R` and `functions.R`)
- **pytask-r**: Integration layer that passes parameters from Python tasks to R scripts
- **Data Exchange**: Python exports simulation data to JSON, R processes it, returns results as JSON

The R conformal inference is skipped for CONFIDENCE bands (only runs for PREDICTION bands) and can be globally disabled with `SKIP_R = True` in `config.py`.

## Important Notes

### Configuration and Constants (`config.py`)
- **Scenario definitions**: `PREDICTION_SCENARIOS` and `CONFIDENCE_SCENARIOS` define parameter combinations
- **Simulation parameters**: `N_SIMULATIONS = 1_000`, `N_JOBS = 12`, `LENGTH_SCALE = 1.0`
- **Hostname-specific logic**: Results are automatically copied to paper directory only when running on hostname "thinky" (see `config.py:29-36`)
- **Path structure**: All build paths are defined here (`BLD_SIMULATION`, `BLD_FIGURES`, etc.)

### Type System (`types.py`)
Key enums that control behavior throughout the codebase:
- `BandType`: CONFIDENCE vs PREDICTION
- `EstimationMethod`: FAIR, MIN_WIDTH, CI (conformal inference)
- `CovarianceType`: STATIONARY vs NON_STATIONARY
- `DistributionType`: GAUSSIAN vs STUDENT_T
Use `parse_enum_type()` to convert strings to enum values safely.

### Testing and Type Checking
- Tests are in `tests/` with structure mirroring `src/fspb/`
- Type checking is strict: `disallow_untyped_defs = true`, `disallow_any_generics = true`
- Tests are exempt from strict typing (see `pyproject.toml`)
- `CHECKS.md` tracks which components have been validated

### Performance Considerations
- Simulations run with `N_SIMULATIONS = 1_000` repetitions using `joblib` parallelization
- Some pytask tasks take significant time due to optimization and Monte Carlo simulation
- R tasks use parallel processing via `future.apply` package

## Code Style and Import Guidelines

**CRITICAL**: Do not use relative imports. Always use absolute imports starting from the package name.

Good:
```python
from fspb.bands.dof import estimate_dof
from fspb.types import BandType
```

Bad:
```python
from .dof import estimate_dof  # Never do this
from ..types import BandType   # Never do this
```
