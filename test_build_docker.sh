#!/bin/bash
# Build the book inside the same container the GitHub Actions workflows use
# (ghcr.io/fenics/dolfinx/lab:v0.11.0, DOLFINx 0.11.0).
#
#   ./test_build_docker.sh           build the book
#   ./test_build_docker.sh --test    run the notebook test suite instead

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_ROOT=$(pwd)
DOCKER_IMAGE="ghcr.io/fenics/dolfinx/lab:v0.11.0"
CONTAINER_NAME="fenicsx-fracture-build-test"

RUN_TESTS=0
[ "$1" = "--test" ] && RUN_TESTS=1

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Pulling ${DOCKER_IMAGE}${NC}"
docker pull ${DOCKER_IMAGE}

if [ "$RUN_TESTS" -eq 1 ]; then
    INNER_CMD='
    set -e
    python3 -m pip install --quiet nbmake jupytext ipyparallel sympy
    cd notebooks
    pytest --nbmake --nbmake-timeout=1800 \
        --ignore=elastodynamics/02-Elastodynamics-Implementation.ipynb \
        --ignore=elastodynamics/04-Elastodynamics_Explicit-Implementation.ipynb \
        --ignore=phase-field-dynamics/02-Elastodynamics-damage-Implementation.ipynb
    '
else
    INNER_CMD='
    set -e
    apt-get update && apt-get install -y libxrender1 xvfb
    python3 -m pip install --quiet "jupyter-book<2" meshio seaborn pandas tqdm pyvista sympy jupytext nbmake ipyparallel
    rm -rf _build
    jupyter-book build .
    touch ./_build/html/.nojekyll
    '
fi

echo -e "${YELLOW}Starting container...${NC}"
docker run --rm \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    --name ${CONTAINER_NAME} \
    -e HDF5_MPI="ON" \
    -e HDF5_DIR="/usr/local/" \
    -e H5PY_SETUP_REQUIRES=0 \
    -e DEB_PYTHON_INSTALL_LAYOUT=deb_system \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e PYVISTA_OFF_SCREEN=true \
    -e PYVISTA_JUPYTER_BACKEND=html \
    --entrypoint bash \
    ${DOCKER_IMAGE} \
    -c "${INNER_CMD}"

echo ""
echo -e "${GREEN}=== Done ===${NC}"
if [ "$RUN_TESTS" -eq 0 ]; then
    echo "Output in _build/html/. To view it:"
    echo "  python3 -m http.server --directory _build/html 8000"
fi
