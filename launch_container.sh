#!/bin/bash
docker pull cmaurini/fenicsx-fracture
docker stop fenicsx-fracture
docker rm fenicsx-fracture
docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 fenicsx-fracture
