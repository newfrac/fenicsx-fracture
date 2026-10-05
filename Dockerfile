# Image used by Binder (mybinder.org), built from the repository root.
# The image for local use is docker/Dockerfile.

FROM ghcr.io/fenics/dolfinx/lab:v0.11.0

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME=/home/${NB_USER}

# for binder: base image upgrades lab to require jupyter-server 2,
# but binder explicitly launches jupyter-notebook
# force binder to launch jupyter-server instead
RUN nb=$(which jupyter-notebook) \
    && rm $nb \
    && ln -s $(which jupyter-lab) $nb

# Notebook dependencies not shipped by the base image
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python3 -m pip cache purge

ENV PYVISTA_JUPYTER_BACKEND="static"
ENV LIBGL_ALWAYS_SOFTWARE=1

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []
