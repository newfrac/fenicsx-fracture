#!/bin/bash
# Docker-based test script to replicate GitHub Actions build exactly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REPO_ROOT=$(pwd)
DOCKER_IMAGE="ubuntu:22.04"
CONTAINER_NAME="fenicsx-fracture-build-test"

echo -e "${YELLOW}=== Building Jupyter Book in Docker (Ubuntu 22.04) ===${NC}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Pulling base image: ${DOCKER_IMAGE}${NC}"
docker pull ${DOCKER_IMAGE}

echo -e "${YELLOW}Starting Docker container...${NC}"

# Run the build in Docker
docker run --rm \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    --name ${CONTAINER_NAME} \
    ${DOCKER_IMAGE} \
    bash -c '
    set -e
    
    echo "=== Setting up environment ==="
    apt-get update && apt-get install -y \
        python3.12 \
        python3-pip \
        libxrender1 \
        xvfb \
        git
    
    echo "=== Upgrading pip ==="
    python3 -m pip install --upgrade pip
    
    echo "=== Installing dependencies ==="
    pip3 install -r docker/requirements.txt
    
    echo "=== Installing jupyter-book and sphinx ==="
    pip install "jupyter-book<2.0.0" "sphinx>=8.0.0"
    
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
