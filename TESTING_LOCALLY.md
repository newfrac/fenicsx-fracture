# Local Testing Guide

This directory contains scripts to test the Jupyter Book build locally, replicating the GitHub Actions environment.

## Quick Start

### Option 1: Conda (Recommended - Most Compatible)

Uses conda to install all dependencies including FEniCS/DOLFINx:

```bash
./test_build_local.sh
```

This script will:
1. Check for conda installation
2. Create/update the `fenicsx-fracture` conda environment from `fenicsx-fracture.yml`
3. Activate the environment
4. Build the book
5. Create the `.nojekyll` file for GitHub Pages

**Requirements:** Anaconda or Miniconda installed

**Manual conda setup** (if you prefer):
```bash
conda env create -f fenicsx-fracture.yml
conda activate fenicsx-fracture
jupyter-book build .
```

### Option 2: Docker (For GitHub Actions Replication)

Uses the same FEniCS/DOLFINx container as GitHub Actions:

```bash
./test_build_docker.sh
```

This script will:
1. Pull the FEniCS/DOLFINx stable container
2. Install Node.js and dependencies via conda in the container
3. Build the book in an isolated, reproducible environment
4. Create the `.nojekyll` file for GitHub Pages

**Requirements:** Docker installed and running

## Viewing the Built Website

After a successful build, view the website locally:

```bash
python3 -m http.server --directory _build/html 8000
```

Then open: **http://localhost:8000**

## Troubleshooting

### Conda not found
If you don't have conda installed:
- **macOS:** `brew install miniconda`
- **Or download:** https://docs.conda.io/projects/conda/en/latest/user-guide/install/

After installing, restart your terminal and try again.

### Build fails with missing dependencies
Check that all conda dependencies are installed:
```bash
conda env create -f fenicsx-fracture.yml --force
```

For FEniCS/DOLFINx specific issues, the Docker option is more reliable.

### `.nojekyll` file not created
The scripts create this automatically after a successful build. This file is **essential** for GitHub Pages to serve static files in `_static/` and `_images/` directories.

### Docker build fails
```bash
# Try pulling manually first
docker pull ghcr.io/fenics/dolfinx/lab:stable
```

## Continuous Testing

To quickly rebuild while developing:

```bash
rm -rf _build && ./test_build_local.sh
```

Or with Docker:
```bash
rm -rf _build && ./test_build_docker.sh
```

## Cleaning Up

To remove the virtual environment and build artifacts:

```bash
rm -rf .venv-test _build
```

## Matching GitHub Actions

These scripts replicate the exact GitHub Actions workflow:
- **Docker option** uses `ghcr.io/fenics/dolfinx/lab:stable` container (exact match)
- **Local option** approximates the environment with your system Python
- Both create the `.nojekyll` file required for GitHub Pages deployment
- Both install dependencies from `pyproject.toml` using the same pip flags
