from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.io

comm = MPI.COMM_WORLD

def mpi_print(s):
    print(f"Rank {comm.rank}: {s}")

mesh = dolfinx.UnitSquareMesh(comm, 1, 1, diagonal="right", ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

V = dolfinx.FunctionSpace(mesh, ("CG", 1))

u = dolfinx.Function(V)
vector = u.vector

# Set the value locally. No communication is performed.
u.vector.setValueLocal(0, comm.rank + 1)

# Print the local and ghosted memory to screen. Notice that the memory on each process is inconsistent.
mpi_print("Before communication")
with vector.localForm() as v_local:
    mpi_print(v_local.array)
    
vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

mpi_print("After communication")
with vector.localForm() as v_local:
    mpi_print(v_local.array)