from dolfinx.fem import assemble_vector, apply_lifting, set_bc, assemble_matrix
import dolfinx.la
import ufl
from petsc4py import PETSc

def project(v, target_func, bcs=[]):
    
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = ufl.inner(Pv, w) * dx
    L = ufl.inner(v, w) * dx

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
    
if __name__ == "__main__":
    
    import dolfinx
    from mpi4py import MPI
    from petsc4py import PETSc

    # Create a mesh and a first function
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.quadrilateral)
    V1 = dolfinx.FunctionSpace(mesh, ("CG", 1))
    u1 = dolfinx.Function(V1)
    u1.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    u1.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Project the derivative of u1 on a new function u2
    V2 = dolfinx.FunctionSpace(mesh, ("DG", 0))
    u2 = dolfinx.Function(V2)
    project(u1.dx(0), u2)