#!/bin/bash
# Local test script to replicate GitHub Actions build

set -e

echo "=== Starting Jupyter Book Build Test ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3 is available
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

PYTHON_CMD="python3"
echo "Using Python: $($PYTHON_CMD --version)"
echo ""

# Check for Node.js (required for Jupyter Book)
echo -e "${YELLOW}Checking Node.js installation...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Node.js not found. Attempting to install via Homebrew...${NC}"
    if command -v brew &> /dev/null; then
        echo "Installing Node.js with Homebrew..."
        brew install node 2>&1 | tail -5
    else
        echo -e "${RED}Error: Node.js is required for Jupyter Book${NC}"
        echo "Please install Node.js from https://nodejs.org/ or via:"
        echo "  brew install node  (if you have Homebrew)"
        exit 1
    fi
fi

echo "Node.js version: $(node --version)"
echo ""
echo ""

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv .venv-test
source .venv-test/bin/activate

# Upgrade pip and install build tools
echo -e "${YELLOW}Upgrading pip and installing build tools...${NC}"
pip install --upgrade pip setuptools pkgconfig poetry-core Cython wheel

# Install jupyter-book first (simpler dependencies)
echo -e "${YELLOW}Installing jupyter-book...${NC}"
pip install jupyter-book

# Install project dependencies
echo -e "${YELLOW}Installing project dependencies from pyproject.toml...${NC}"
pip install --no-build-isolation --no-binary=h5py . 2>/dev/null || pip install . 2>/dev/null || echo "Warning: some dependencies may not have installed"

echo ""
echo -e "${YELLOW}Dependencies installed. Building Jupyter Book...${NC}"
echo ""

# Clean and build
rm -rf _build
if jupyter-book build .; then
    echo ""
    echo -e "${GREEN}=== Build successful! ===${NC}"
    
    # Check for .nojekyll file
    if [ -f "_build/html/.nojekyll" ]; then
        echo -e "${GREEN}✓ .nojekyll file exists${NC}"
    else
        echo -e "${YELLOW}Creating .nojekyll file...${NC}"
        touch _build/html/.nojekyll
        echo -e "${GREEN}✓ .nojekyll file created${NC}"
    fi
    
    echo ""
    echo "Build output located in: _build/html/"
    echo ""
    echo -e "${YELLOW}To view the built website locally, run:${NC}"
    echo "python3 -m http.server --directory _build/html 8000"
    echo "Then open: http://localhost:8000"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}=== Build failed ===${NC}"
    echo "Check the error messages above for details"
    echo ""
    echo -e "${YELLOW}Note: For a fully compatible build environment, use Docker:${NC}"
    echo "./test_build_docker.sh"
    exit 1
fi
