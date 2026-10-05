# Local testing

The book targets **DOLFINx 0.11.0**. Two scripts reproduce what the GitHub Actions
workflows do, so a failure can be diagnosed without pushing.

| | `test_build_local.sh` | `test_build_docker.sh` |
|---|---|---|
| Environment | conda, from `fenicsx-fracture.yml` | `ghcr.io/fenics/dolfinx/lab:v0.11.0` |
| Matches CI | approximately (conda-forge builds) | exactly (same image) |
| Requires | conda | Docker |

## Conda

```bash
./test_build_local.sh              # build the book into _build/html
./test_build_local.sh --test       # run the notebook test suite
./test_build_local.sh --recreate   # rebuild the environment from scratch first
```

The script creates the `fenicsx-fracture` environment if it is missing and updates
it (`conda env update --prune`) otherwise.

Manual setup:

```bash
conda env create -f fenicsx-fracture.yml
conda activate fenicsx-fracture
jupyter-book build .
```

## Docker

```bash
./test_build_docker.sh             # build the book
./test_build_docker.sh --test      # run the notebook test suite
```

This uses the same container image as the workflows, pinned to `v0.11.0`.

## Viewing the result

```bash
python3 -m http.server --directory _build/html 8000
```

Then open <http://localhost:8000>.

## Notebooks excluded from the test suite

Three notebooks are skipped, both here and in CI, because they are long-running
time-stepping simulations:

- `elastodynamics/02-Elastodynamics-Implementation.ipynb`
- `elastodynamics/04-Elastodynamics_Explicit-Implementation.ipynb`
- `phase-field-dynamics/02-Elastodynamics-damage-Implementation.ipynb`

They are still executed when the book is built.

## Troubleshooting

**Environment fails to solve.** Use strict channel priority:

```bash
conda config --set channel_priority strict
```

**`.nojekyll` missing.** Both scripts create it after a successful build. Without
it GitHub Pages does not serve `_static/` and `_images/`.

**Stale build.** `rm -rf _build` before rebuilding; both scripts do this already.

**`EngineError: Engine set stopped: {'exit_code': 16}`** when a notebook starts an
`ipyparallel` MPI cluster. Another MPI implementation is shadowing the one in the
conda environment — typically Homebrew's Open MPI in `/opt/homebrew/bin`, while
the conda packages are built against MPICH. Check with

```bash
conda activate fenicsx-fracture
which -a mpiexec        # the environment's bin must come first
mpiexec --version       # must report MPICH/Hydra, not Open MPI
```

and put the environment ahead of Homebrew in `PATH` for the session, or
`brew unlink open-mpi`.

## Cleaning up

```bash
rm -rf _build
conda env remove -n fenicsx-fracture
```
