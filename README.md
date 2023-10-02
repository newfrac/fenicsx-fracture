[![Notebook CI](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/test_stable.yml/badge.svg)](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/test_stable.yml)
[![Book CI](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/book_stable.yml/badge.svg)](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/book_stable.yml)

# FEniCSx Fracture mechanics examples

This repository automatically generates a jupyter-book here: `https://newfrac.github.io/fenicsx-fracture/`

## Installation

### Conda

To create the conda environement and activate it

```bash
conda env create -f fenicsx-fracture.yml
conda activate fenicsx-fracture
```

To generate the book the book locally:

```bash
jupyter-book build notebooks
```

To visualize the results, open in your browser the generated file `_build/html/index.html`.

### Docker

To build a docker image for this documentation, you can run

```
docker build -t fenicsx-fracture -f docker/Dockerfile .
```

from the root of this repository. To create a one-time usage container you can call:

```
docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 fenicsx-fracture
```

You can then access the jupyter lab notebook at `http://localhost:8888` in your browser.

## Binder

Although we recommend to execute the notebook locally, you can also use the cloud-based binder service to execute the notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/newfrac/fenicsx-fracture/HEAD)

## Acknowledgements

The funding received from the European Union’s Horizon 2020 research and
innovation programme under Marie Skłodowska-Curie grant agreement No.
861061-NEWFRAC is gratefully acknowledged.

## License

MIT License, see `LICENSE` file.
