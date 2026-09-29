#!/bin/bash
# Build the book image (DOLFINx 0.11.0) from the root of the repository.
docker build -t fenicsx-fracture -f docker/Dockerfile .
