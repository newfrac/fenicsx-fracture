#!/bin/bash
# Local test script using conda to install dependencies

set -e

echo "=== Starting Jupyter Book Build Test with Conda ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for conda
echo -e "${YELLOW}Checking conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda from https://conda.io/projects/conda/en/latest/user-guide/install/"
    exit 1
fi

echo "Conda version: $(conda --version)"
echo ""

# Check for Node.js
echo -e "${YELLOW}Checking Node.js installation...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Node.js not found. Will install via conda.${NC}"
else
    echo "Node.js version: $(node --version)"
fi
echo ""

# Create conda environment
echo -e "${YELLOW}Creating/updating conda environment for fenicsx-fracture...${NC}"
conda env create -f fenicsx-fracture.yml --force 2>&1 | grep -E "(Creating|Updating|Solving|Preparing|Downloading|Extracting|Package|completed|ERROR)" || true

# Activate environment
echo ""
echo -e "${YELLOW}Activating environment...${NC}"
source activate fenicsx-fracture

echo ""
echo -e "${YELLOW}Environment activated. Building Jupyter Book...${NC}"
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
