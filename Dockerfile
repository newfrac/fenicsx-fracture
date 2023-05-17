FROM ghcr.io/fenics/dolfinx/lab:v0.6.0-r1

RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb nodejs
ENV PYVISTA_JUPYTER_BACKEND="static"

ADD docker/requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME /home/${NB_USER}

# for binder: base image upgrades lab to require jupyter-server 2,
# but binder explicitly launches jupyter-notebook
# force binder to launch jupyter-server instead
RUN nb=$(which jupyter-notebook) \
    && rm $nb \
    && ln -s $(which jupyter-lab) $nb

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []