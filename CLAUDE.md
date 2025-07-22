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

## Key Commands

### Testing
- `pixi run pytest` - Run Python tests
- `pixi run pytest tests/` - Run tests in tests directory
- `pixi run pytest -k "test_name"` - Run specific test

### Development Tasks
- `pixi run pytask` - Run the task pipeline (equivalent to `pixi run pytask build`)
- `pixi run pytask clean` - Clean build artifacts
- `pixi run pytask collect` - Show available tasks
- `pixi run pytask dag` - Visualize task dependencies

### Code Quality
- `pixi run mypy src` - Type checking
- `pixi run ruff check src` - Linting (auto-fixes enabled)
- `pixi run pre-commit run --all-files` - Run pre-commit hooks

## Architecture

### Core Components

#### 1. Band Generation (`src/fspb/bands/`)
- **`band.py`**: Main `Band` class with coverage and width statistics
- **`fair_algorithm.py`**: Fair critical value selection algorithm
- **`min_width_algorithm.py`**: Minimum width critical value selection
- **`linear_model.py`**: Linear modeling for functional data
- **`covariance.py`**: Covariance estimation for functional data
- **`roughness.py`**: Roughness penalty calculations

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

The project uses **pytask** for orchestrating computational tasks defined in `task_*.py` files:

- **`task_simulation_study.py`**: Main simulation execution
- **`task_process_simulation_results.py`**: Results aggregation
- **`task_produce_results_tables.py`**: LaTeX table generation
- **`task_visualize_band.py`**: Band visualization
- **`task_visualize_outcome.py`**: Outcome visualization
- **`task_move_to_paper_dir.py`**: Results transfer to paper directory

### Build Structure

```
bld/
├── simulation/          # Simulation outputs
│   ├── data/           # Raw simulation data
│   ├── ci/             # Conformal inference results
│   ├── fair/           # Fair algorithm results
│   ├── min_width/      # Minimum width results
│   └── processed/      # Aggregated results
├── figures/            # Generated plots and visualizations
└── tables/             # LaTeX tables for papers
```

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
- **Python**: Main computational work, band algorithms, simulation
- **R**: Conformal inference implementation (`src/fspb/R/`)
- **pytask-r**: Integration between Python tasks and R scripts

## Development Workflow

1. **Environment Setup**: Use `pixi install` to set up the environment
2. **Task Execution**: Use `pixi run pytask` to run the computational pipeline
3. **Testing**: Use `pixi run pytest` to run tests
4. **Type Checking**: Use `pixi run mypy src` before committing
5. **Code Quality**: Pre-commit hooks handle formatting and linting

## Important Notes

- The project includes hostname-specific logic for result file management (see `config.py:29-36`)
- R environment is managed separately with renv
- Some tasks may take significant time due to simulation complexity
- Build artifacts are organized by scenario parameters for systematic comparison

## Code Style and Import Guidelines

- Do not use relative imports, always start with the package name and then traverse the package structure. I.e., do not do from .dof import ..., but do from fspb.bands.dof import ...!
