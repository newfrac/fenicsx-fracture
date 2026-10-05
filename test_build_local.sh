#!/bin/bash
# Build the book locally in the conda environment described by fenicsx-fracture.yml
# (DOLFINx 0.11.0).
#
#   ./test_build_local.sh            build the book
#   ./test_build_local.sh --test     run the notebook test suite instead
#   ./test_build_local.sh --recreate rebuild the conda environment from scratch

set -e

ENV_NAME=fenicsx-fracture
ENV_FILE=fenicsx-fracture.yml

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

RUN_TESTS=0
RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --test) RUN_TESTS=1 ;;
        --recreate) RECREATE=1 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "See https://conda.io/projects/conda/en/latest/user-guide/install/"
    exit 1
fi
echo "Conda version: $(conda --version)"

if [ "$RECREATE" -eq 1 ]; then
    echo -e "${YELLOW}Removing the existing '${ENV_NAME}' environment...${NC}"
    conda env remove -n "${ENV_NAME}" -y || true
fi

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo -e "${YELLOW}Updating the '${ENV_NAME}' environment from ${ENV_FILE}...${NC}"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo -e "${YELLOW}Creating the '${ENV_NAME}' environment from ${ENV_FILE}...${NC}"
    conda env create -f "${ENV_FILE}"
fi
echo -e "${GREEN}✓ Environment ready${NC}"

conda run --no-capture-output -n "${ENV_NAME}" python -c \
    "import dolfinx; print(f'DOLFINx {dolfinx.__version__}')"

export PYVISTA_OFF_SCREEN=true
export PYVISTA_JUPYTER_BACKEND=static
export LIBGL_ALWAYS_SOFTWARE=1

if [ "$RUN_TESTS" -eq 1 ]; then
    echo -e "${YELLOW}Running the notebook test suite...${NC}"
    cd notebooks
    conda run --no-capture-output -n "${ENV_NAME}" python -m pytest --nbmake \
        --nbmake-timeout=1800 \
        --ignore=elastodynamics/02-Elastodynamics-Implementation.ipynb \
        --ignore=elastodynamics/04-Elastodynamics_Explicit-Implementation.ipynb \
        --ignore=phase-field-dynamics/02-Elastodynamics-damage-Implementation.ipynb
    echo -e "${GREEN}=== Notebook tests passed ===${NC}"
    exit 0
fi

echo -e "${YELLOW}Building the Jupyter Book...${NC}"
rm -rf _build
conda run --no-capture-output -n "${ENV_NAME}" jupyter-book build .

touch _build/html/.nojekyll
echo -e "${GREEN}=== Build successful ===${NC}"
echo ""
echo "Output in _build/html/. To view it:"
echo "  python3 -m http.server --directory _build/html 8000"
