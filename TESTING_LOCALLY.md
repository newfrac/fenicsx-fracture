# Local Testing Guide

This directory contains scripts to test the Jupyter Book build locally, replicating the GitHub Actions environment.

## Quick Start

### Option 1: Docker (Recommended - Most Accurate)

Uses the same FEniCS/DOLFINx container as GitHub Actions:

```bash
./test_build_docker.sh
```

This script will:
1. Pull the FEniCS/DOLFINx stable container
2. Install dependencies from `pyproject.toml`
3. Build the book in an isolated, reproducible environment
4. Create the `.nojekyll` file for GitHub Pages

**Requirements:** Docker installed and running

### Option 2: Local Python Virtual Environment (Faster)

For quick iteration without Docker:

```bash
./test_build_local.sh
```

This script will:
1. Check for Python 3.11+ and Node.js (required for Jupyter Book)
2. Install Node.js via Homebrew if needed
3. Create a Python virtual environment (`.venv-test`)
4. Install all dependencies from `pyproject.toml`
5. Build the book
6. Create the `.nojekyll` file for GitHub Pages

**Requirements:** 
- Python 3.11+ installed locally
- Node.js (script will try to install via Homebrew if missing)

## Viewing the Built Website

After a successful build, view the website locally:

```bash
python3 -m http.server --directory _build/html 8000
```

Then open: **http://localhost:8000**

## Troubleshooting

### Node.js not found
Jupyter Book 2.x requires Node.js. The local script will try to install it automatically if missing.

If you see "Node.js is required" error:
```bash
# On macOS with Homebrew
brew install node

# Or install from https://nodejs.org/
```

### Build fails with missing dependencies
- **Docker option:** Make sure Docker is running: `docker ps`
- **Local option:** Ensure you have Python 3.11+ installed: `python3 --version`
- Check that `pyproject.toml` has the correct dependencies
- For FEniCS/DOLFINx related issues, Docker is highly recommended

### `.nojekyll` file not created
The scripts create this automatically after a successful build. This file is **essential** for GitHub Pages to serve static files in `_static/` and `_images/` directories.

### Docker image pull fails
```bash
# Try pulling manually first
docker pull ghcr.io/fenics/dolfinx/lab:stable
```

### Local build fails with Python version issues
Use Docker instead - it has all the right dependencies pre-configured:
```bash
./test_build_docker.sh
```

## Continuous Testing

To quickly rebuild while developing:

```bash
rm -rf _build && ./test_build_docker.sh
```

Or with local Python:
```bash
rm -rf _build && ./test_build_local.sh
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
