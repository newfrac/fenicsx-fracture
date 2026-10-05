# Image for the fenicsx-fracture book, built on the official DOLFINx JupyterLab image.
#
# Build from the root of the repository:
#   docker build -t fenicsx-fracture -f docker/Dockerfile .
#
# Multi-platform build:
#   docker buildx build --platform=linux/arm64,linux/amd64 -f docker/Dockerfile . --progress=plain

FROM ghcr.io/fenics/dolfinx/lab:v0.11.0
ARG TARGETPLATFORM

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"
ENV HDF5_DIR="/usr/local"

WORKDIR /tmp/

# Requirements for pyvista off-screen rendering
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev mesa-utils libxrender1 xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U setuptools pip pkgconfig

# Book and notebook dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir vtk && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    HDF5_MPI=${HDF5_MPI} CC=mpicc HDF5_DIR=${HDF5_DIR} \
        python3 -m pip install --no-cache-dir --no-binary=h5py h5py && \
    python3 -m pip cache purge

ENV PYVISTA_OFF_SCREEN="false"
ENV PYVISTA_JUPYTER_BACKEND="trame"
ENV LIBGL_ALWAYS_SOFTWARE=1

EXPOSE 8888/tcp
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
