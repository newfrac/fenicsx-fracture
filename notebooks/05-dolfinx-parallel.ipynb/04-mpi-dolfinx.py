from mpi4py import MPI
import dolfinx
import dolfinx.io

# DOLFINx uses mpi4py communicators.
comm = MPI.COMM_WORLD

def mpi_print(s):
    print(f"Rank {comm.rank}: {s}")

# When you construct a mesh you must pass an MPI communicator.
# The mesh will automatically be *distributed* over the ranks of the MPI communicator.
# Important: In this script we use dolfinx.cpp.mesh.GhostMode.none.
# This is *not* the default (dolfinx.cpp.mesh.GhostMode.shared_facet).
# We will discuss the effects of the ghost_mode parameter in the next section.
mesh = dolfinx.UnitSquareMesh(comm, 1, 1, diagonal="right", ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
mesh.topology.create_connectivity_all()

mpi_print(f"Number of local cells: {mesh.topology.index_map(2).size_local}")
mpi_print(f"Number of global cells: {mesh.topology.index_map(2).size_global}")
mpi_print(f"Number of local vertices: {mesh.topology.index_map(0).size_local}")
mpi_print("Cell (dim = 2) to vertex (dim = 0) connectivity")
mpi_print(mesh.topology.connectivity(2, 0))