# Local Testing Guide

This directory contains scripts to test the Jupyter Book build locally, replicating the GitHub Actions environment.

## Quick Start

### Option 1: Local Python Virtual Environment (Faster)

```bash
./test_build_local.sh
```

This script will:
1. Create a Python virtual environment (`.venv-test`)
2. Install all dependencies from `docker/requirements.txt`
3. Install jupyter-book and sphinx
4. Build the book
5. Create the `.nojekyll` file for GitHub Pages

**Requirements:** Python 3.12 or Python 3

### Option 2: Docker (Most Accurate - Matches GitHub Actions exactly)

```bash
./test_build_docker.sh
```

This script will:
1. Pull the Ubuntu 22.04 Docker image (same as GitHub Actions)
2. Install Python 3.12 and all dependencies inside the container
3. Build the book in an isolated environment
4. Create the `.nojekyll` file

**Requirements:** Docker installed

## Viewing the Built Website

After a successful build, view the website locally:

```bash
python3 -m http.server --directory _build/html 8000
```

Then open: **http://localhost:8000**

## Troubleshooting

### Build fails with missing dependencies
- Check `docker/requirements.txt` for required packages
- Make sure you have Python 3 installed
- For Docker option: ensure Docker daemon is running

### `.nojekyll` file not created
The scripts create this automatically after a successful build. This file is essential for GitHub Pages to serve static files in `_static/` and `_images/` directories.

### Slow first-time build
The first build caches notebooks. Subsequent builds should be faster. You can force a full rebuild by manually deleting the `_build/` directory.

## Cleaning Up

To remove the virtual environment and build artifacts:

```bash
rm -rf .venv-test _build
```

## Continuous Testing

To quickly test while developing, use:

```bash
rm -rf _build && ./test_build_local.sh
```
