FROM dolfinx/dolfinx as lab
LABEL description="DOLFIN-X Jupyter Lab for Binder"

#USER root#

#ARG NB_USER
#ARG NB_UID
#ENV USER ${NB_USER}
#ENV HOME /home/${NB_USER}
#ENV PETSC_ARCH "linux-gnu-real-32"
#RUN adduser --disabled-password \
#    --gecos "Default user" \
#    --uid ${NB_UID} \
#    ${NB_USER}


#WORKDIR ${HOME}
#COPY . ${HOME}
#USER ${USER}
##

# create user with a home directory
ARG NB_USER
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY . ${HOME}
ENV PETSC_ARCH "linux-gnu-real-32"
USER root
RUN chown -R ${NB_UID} ${HOME}
RUN pip3 install jupyterhub nbconvert itkwidgets jupyter-book sympy

USER ${NB_USER}
ENTRYPOINT []