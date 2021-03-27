from mpi4py import MPI
import dolfinx
import dolfinx.io

comm = MPI.COMM_WORLD

def mpi_print(s):
    print(f"Rank {comm.rank}: {s}")

mesh = dolfinx.UnitSquareMesh(comm, 1, 1, diagonal="right", ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

V = dolfinx.FunctionSpace(mesh, ("CG", 1))

u = dolfinx.Function(V)
vector = u.vector

mpi_print(f"Local size of vector: {vector.getLocalSize()}")

# .localForm() allows us to access the local array with space for both owned and local degrees of freedom.
with vector.localForm() as v_local:
    mpi_print(f"Local + Ghost size of vector: {v_local.getLocalSize()}")
    
vector.ghostUpdate()
