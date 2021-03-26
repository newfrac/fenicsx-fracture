from mpi4py import MPI

comm = MPI.COMM_WORLD
assert(comm.size == 2)

if comm.rank == 0:
    b = 3
    c = 5
    a = b + c
    comm.send(a, dest=1, tag=20)
    print(f"Rank {comm.rank} a: {a}")
elif comm.rank == 1:
    a = comm.recv(source=0, tag=20)
    print(f"Rank {comm.rank} a: {a}")