#!/bin/bash
docker pull jhale/newfrac-fenicsx-training
docker stop dolfinx-newfrac
docker rm dolfinx-newfrac
docker run --init -p 8888:8888 --name dolfinx-newfrac -v "$(pwd)":/root/shared jhale/newfrac-fenicsx-training
