from mpi4py import MPI

comm = MPI.COMM_WORLD
print(f"Hello world from rank {comm.rank} of {comm.size}")