from mpi4py import MPI
import dolfinx
import dolfinx.io

comm = MPI.COMM_WORLD

def mpi_print(s):
    print(f"Rank {comm.rank}: {s}")

mesh = dolfinx.UnitSquareMesh(comm, 1, 1, diagonal="right", ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

V = dolfinx.FunctionSpace(mesh, ("CG", 1))

mpi_print(f"Global size: {V.dofmap.index_map.size_global}")
mpi_print(f"Local size: {V.dofmap.index_map.size_local}")
mpi_print(f"Ghosts (global numbering): {V.dofmap.index_map.ghosts}")
