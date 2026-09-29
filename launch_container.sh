#!/bin/bash
# Run the image built by build_image.sh, mounting the current directory.
docker run --rm -ti -v "$(pwd)":/root/shared -w /root/shared --init -p 8888:8888 fenicsx-fracture
