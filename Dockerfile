FROM dolfinx/dolfinx as lab
LABEL description="DOLFIN-X Jupyter Lab for Binder"

USER root
WORKDIR /tmp/

# Dependencies for pyvista and related packages
RUN wget -qO - https://deb.nodesource.com/setup_15.x | bash && \
    apt-get -qq update && \
    apt-get install -y libgl1-mesa-dev xvfb nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pyvista ITKWidgets dependencies
RUN pip3 install --no-cache-dir itkwidgets ipywidgets matplotlib
RUN jupyter labextension install jupyter-matplotlib jupyterlab-datawidgets itkwidgets 

# Install meshio
RUN pip3 install --no-cache-dir --no-binary=h5py h5py meshio 

# Install progress-bar
RUN pip3 install tqdm

# Additional python modules
RUN pip3 install --no-cache-dir sympy

ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}
ENV PETSC_ARCH "linux-gnu-real-32"
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
COPY . ${HOME}
USER ${USER}
