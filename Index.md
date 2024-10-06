[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/newfrac/fenicsx-fracture/HEAD)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/msolides-2024/fenicsx-fracture?quickstart=1)

# Computational fracture mechanics examples with FEniCSx

This webpage provides basic examples on computational methods to solve fracture mechanics problems using [DOLFINx](https://github.com/FEniCS/dolfinx/).

This work includes the contribution of the Early Stage Researchers (ESR) of the ITN project [Newfrac](https://www.newfrac.eu) funded by the European Commission under a Marie Skłodowska-Curie Actions Grant Agreement n. 861061.

The webpage is build using Jupyter-book, reusing the configuration of the [Dolfinx Tutorial](https://jsdokken.com/dolfinx-tutorial/).

Comments and corrections to this webpage should be submitted to the issue tracker by going to the relevant page, then click the ![git](git.png)-symbol in the top right corner and "open issue".

## Citation
If you find the material of this repository useful, please cite it in your publications using the following reference:

Chao Correas, A., Jack S. Hale, Jiménez Alfaro, S., Andrey Latyshev, & Maurini, C. (2024). newfrac/fenicsx-fracture: v1.0 (v1.0). Zenodo. https://doi.org/10.5281/zenodo.11518791

[![DOI](https://zenodo.org/badge/642212191.svg)](https://zenodo.org/doi/10.5281/zenodo.11518790)


## Installation

To run this notebooks on your computer, we suggest using Docker or Conda, as exaplained below.

### Docker

1. First, install docker for your operating system. You can find instructions here: https://docs.docker.com/get-docker/

2. Download and unzip the present repository. If you have git installed, you can clone the repository with `git clone 
https://github.com/newfrac/fenicsx-fracture.git`.  
Otherwise download and unzip the file  `https://github.com/newfrac/fenicsx-fracture/archive/refs/heads/main.zip`. 

3. To build a docker image for this documentation, you can run from the root of the downloaded repository (use the `PowerShell` if you are on Windows)

```
docker build -t fenicsx-fracture -f docker/Dockerfile .
```

4. To create a one-time usage container you can call:

```
docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 fenicsx-fracture
```

You can then access the jupyter lab notebook opening in your browser one of the links starting with `http://...` indicated in the terminal.

Steps 1-3 need to be done only the first time. After, you can then start the container with the command in step 4 directly.

### Conda

To run the notebooks locally, we recommend to use the conda environment provided in this repository. To install conda, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

To create the conda environment and activate it

```bash
conda env create -f fenicsx-fracture.yml
conda activate fenicsx-fracture
```

### Binder

Although we recommend executing the notebook locally, you can also use the cloud-based binder service to execute the notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/newfrac/fenicsx-fracture/HEAD)

### Google Colab

Go to [Google Colab](https://colab.research.google.com) and create a new notebook. We will use the FEM on Colab project to install FEniCSx. Copy and paste into a new notebook cell:

```python
try:
    import dolfinx
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/fenicsx-install-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
import dolfinx
```

and press Shift+Enter. You should see output from the install process and no errors.

## Acknowledgements

The funding received from the European Union’s Horizon 2020 research and
innovation programme under Marie Skłodowska-Curie grant agreement No.
861061-NEWFRAC is gratefully acknowledged.


This project is created using the open source [Jupyter Book project](https://jupyterbook.org/) and the book of [dolfinx-tutorial](https://github.com/jorgensd/dolfinx-tutorial/blob/dokken/jupyterbook/Dockerfile) as a template.


## Building the book locally (for developers)

To generate the book locally:

```bash
jupyter-book build .
```

To visualize the results, open in your browser the generated file `_build/html/index.html`.


## License

MIT License, see `LICENSE` file.
