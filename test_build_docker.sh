#!/bin/bash
# Docker-based test script to replicate GitHub Actions build exactly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REPO_ROOT=$(pwd)
DOCKER_IMAGE="ghcr.io/fenics/dolfinx/lab:stable"
CONTAINER_NAME="fenicsx-fracture-build-test"

echo -e "${YELLOW}=== Building Jupyter Book in Docker (FEniCS/DOLFINx) ===${NC}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Pulling FEniCS/DOLFINx container: ${DOCKER_IMAGE}${NC}"
docker pull ${DOCKER_IMAGE}

echo -e "${YELLOW}Starting Docker container...${NC}"

# Run the build in Docker using the FEniCS/DOLFINx container
docker run --rm \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    --name ${CONTAINER_NAME} \
    -e HDF5_MPI="ON" \
    -e HDF5_DIR="/usr/local/" \
    -e H5PY_SETUP_REQUIRES=0 \
    -e DEB_PYTHON_INSTALL_LAYOUT=deb_system \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e PYVISTA_OFF_SCREEN=false \
    -e PYVISTA_JUPYTER_BACKEND=html \
    --entrypoint bash \
    ${DOCKER_IMAGE} \
    -c '
    set -e
    
    echo "=== Installing Node.js ==="
    apt-get update && apt-get install -y nodejs npm
    
    echo "=== Installing dependencies with conda ==="
    conda install -c conda-forge -y fenics-dolfinx jupyter-book meshio h5py seaborn pandas tqdm pyvista sympy
    conda install -c conda-forge -y jupytext nbmake ipyparallel
    
    echo ""
    echo "=== Building Jupyter Book ==="
    rm -rf _build
    jupyter-book build .
    
    echo "=== Creating .nojekyll file ==="
    touch ./_build/html/.nojekyll
    
    echo ""
    echo "=== Build completed successfully ==="
    '

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Docker build successful! ===${NC}"
    echo ""
    echo "Build output is in: _build/html/"
    echo ""
    echo -e "${YELLOW}To view the built website locally, run:${NC}"
    echo "python3 -m http.server --directory _build/html 8000"
    echo "Then open: http://localhost:8000"
else
    echo ""
    echo -e "${RED}=== Docker build failed ===${NC}"
    exit 1
fi
