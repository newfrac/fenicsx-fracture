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

# Check if Python 3.12 is available
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v python3.12 &> /dev/null; then
    echo -e "${YELLOW}Python 3.12 not found, checking for python3...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 not found${NC}"
        exit 1
    fi
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python3.12"
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo ""

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv .venv-test
source .venv-test/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies from docker/requirements.txt
echo -e "${YELLOW}Installing dependencies from docker/requirements.txt...${NC}"
if [ -f "docker/requirements.txt" ]; then
    pip install -r docker/requirements.txt
else
    echo -e "${RED}Error: docker/requirements.txt not found${NC}"
    exit 1
fi

# Install jupyter-book and sphinx
echo -e "${YELLOW}Installing jupyter-book and sphinx...${NC}"
pip install "jupyter-book<2.0.0" "sphinx>=8.0.0"

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
    exit 1
fi
