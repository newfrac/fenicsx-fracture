#!/bin/bash
# docker pull dolfinx/lab
# docker stop dolfinx-newfrac
# docker rm dolfinx-newfrac
docker run --init -p 8888:8888 --name dolfinx-newfrac -v "$(pwd)":/root/shared -w /root/shared dolfinx/lab

