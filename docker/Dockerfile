# Execute from root of repo as: docker buildx build --platform=linux/arm64,linux/amd64 -f docker/Dockerfile ./docker/ --progress=plain

FROM ghcr.io/fenics/dolfinx/lab:v0.9.0
ARG TARGETPLATFORM


ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"
ENV HDF5_DIR="/usr/local"
ENV PYVISTA_JUPYTER_BACKEND="static"

WORKDIR /tmp/RUN 
# Requirements for pyvista (gl1 and render1) and jupyterlab (nodejs and curl)
RUN apt-get update && apt-get install -y  libgl1-mesa-dev mesa-utils  libxrender1 xvfb curl 

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Upgrade setuptools and pip
RUN python3 -m pip install -U setuptools pip pkgconfig

# BUILD VTK
# ENV VTK_VERSION="v9.2.6"
# RUN git clone --recursive --branch ${VTK_VERSION} --single-branch https://gitlab.kitware.com/vtk/vtk.git vtk && \
#     cmake -G Ninja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON vtk/ && \
#     ninja && \
#     python3 setup.py bdist_wheela


RUN python3 -m pip install vtk
RUN python3 -m pip install --no-cache-dir --no-binary=h5py h5py 
RUN python3 -m pip cache purge

ENV PYVISTA_OFF_SCREEN="false"
ENV PYVISTA_JUPYTER_BACKEND="trame"
ENV LIBGL_ALWAYS_SOFTWARE=1

# RUN jupyter lab build
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]