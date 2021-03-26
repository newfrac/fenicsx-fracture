from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.io
import ufl
import os

comm = MPI.COMM_WORLD

def mpi_print(s):
    print(f"Rank {comm.rank}: {s}")

mesh = dolfinx.UnitSquareMesh(comm, 1, 1, diagonal="right", ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

V = dolfinx.FunctionSpace(mesh, ("CG", 1))

u = dolfinx.Function(V)
v = ufl.TestFunction(V)

L = ufl.inner(1.0, v)*ufl.dx

b = dolfinx.fem.assemble_vector(L)

mpi_print("Before communication")
with b.localForm() as b_local:
    mpi_print(b_local.array)
    
print("\n")

# This call takes the values from the ghost regions and accumulates (adds) them to the owning process.
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

mpi_print("After ADD/REVERSE update")
with b.localForm() as b_local:
    mpi_print(b_local.array)
    
print("\n")

# Important point: The ghosts are still inconsistent!
# This call takes the values from the owning processes and updates the ghosts.
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

mpi_print("After INSERT/FORWARD update")
with b.localForm() as b_local:
    mpi_print(b_local.array)