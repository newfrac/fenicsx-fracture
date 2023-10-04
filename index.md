# Computational fracture mechanics examples with FEniCSx

This webpage provides basic examples on computational methods to solve fracture mechanics problems using [DOLFINx](https://github.com/FEniCS/dolfinx/).

This work includes the contribution of the Early Stage Researchers (ESR) of the ITN project [Newfrac](https://www.newfrac.eu) funded by the European Commission under a Marie Skłodowska-Curie Actions Grant Agreement n. 861061.

The webpage is build using Jupyter-book, reusing the configuration of the [Dolfinx Tutorial](https://jsdokken.com/dolfinx-tutorial/).

Comments and corrections to this webpage should be submitted to the issue tracker by going to the relevant page, then click the ![git](git.png)-symbol in the top right corner and "open issue".

### Interactive tutorials

As this book has been published as a Jupyter Book, we provide interactive notebooks that can be run in the browser. To start such a notebook click the ![Binder symbol](binder.png)-symbol in the top right corner of the relevant tutorial.

[![Notebook CI](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/test_stable.yml/badge.svg)](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/test_stable.yml)
[![Book CI](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/book_stable.yml/badge.svg)](https://github.com/newfrac/fenicsx-fracture/blob/main/.github/workflows/book_stable.yml)

### Installation

#### Conda

To create the conda environment and activate it

```bash
conda env create -f fenicsx-fracture.yml
conda activate fenicsx-fracture
```

To generate the book locally:

```bash
jupyter-book build .
```

To visualize the results, open in your browser the generated file `_build/html/index.html`.

#### Docker

To build a docker image for this documentation, you can run

```
docker build -t fenicsx-fracture -f docker/Dockerfile .
```

from the root of this repository. To create a one-time usage container you can call:

```
docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 fenicsx-fracture
```

You can then access the jupyter lab notebook at `http://localhost:8888` in your browser. 

If you do not have docker installed, you can install it from [here](https://docs.docker.com/get-docker/).

#### Binder

Although we recommend to execute the notebook locally, you can also use the cloud-based binder service to execute the notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/newfrac/fenicsx-fracture/HEAD)

## Acknowledgements

The funding received from the European Union’s Horizon 2020 research and
innovation programme under Marie Skłodowska-Curie grant agreement No.
861061-NEWFRAC is gratefully acknowledged.


This project is created using the open source [Jupyter Book project](https://jupyterbook.org/) and the book of [dolfinx-tutorial](https://github.com/jorgensd/dolfinx-tutorial/blob/dokken/jupyterbook/Dockerfile) as a template.

## License

MIT License, see `LICENSE` file.
