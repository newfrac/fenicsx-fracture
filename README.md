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
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm fenicsx-fracture
```

```
docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 fenicsx-fracture
```

# Docker images

Docker images for this tutorial can be found in the [packages tab](https://github.com/jorgensd/dolfinx-tutorial/pkgs/container/dolfinx-tutorial) of the dolfinx-tutorial

Additional requirements on top of the `dolfinx/lab` images can be found at [Dockerfile](docker/Dockerfile) and [requirements.txt](docker/requirements.txt)

##

An image building DOLFINx, Basix, UFL and FFCx from source can be built using:

```bash
cd docker
docker build -f LocalDockerfile -t local_lab_env .
```

and run

```bash
 docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 local_lab_env
 ```

from the main directory.

## Binder

Although we recommend to execute the notebook locally, you can also use the cloud-based binder service to execute the notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/newfrac/fenicsx-fracture/HEAD)

## Acknowledgements

The funding received from the European Union’s Horizon 2020 research and
innovation programme under Marie Skłodowska-Curie grant agreement No.
861061-NEWFRAC is gratefully acknowledged.

## License

MIT License, see `LICENSE` file.
