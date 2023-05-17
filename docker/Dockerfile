FROM ghcr.io/fenics/dolfinx/lab:v0.6.0-r1

WORKDIR /tmp/

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV PYVISTA_JUPYTER_BACKEND="static"

# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb nodejs

# Upgrade setuptools and pip
RUN python3 -m pip install -U "setuptools" pip pkgconfig

RUN python3 -m pip install cython

ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python3 -m pip cache purge
# RUN jupyter lab build
# ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
ENTRYPOINT [ "/bin/bash" ]