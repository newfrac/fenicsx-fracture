Files to build Docker images locally on your computer.

Can be run using

    docker build --tag=local_jupyter .
    cd ..
    docker run --rm -v $PWD:/root/shared  --init -p 8888:8888 local_jupyter
