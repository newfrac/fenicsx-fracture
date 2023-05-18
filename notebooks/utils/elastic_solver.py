import sys
sys.path.append("../python")

# Import required libraries
import matplotlib.pyplot as plt
import numpy as np

import dolfinx.fem as fem
import dolfinx.mesh as mesh
import dolfinx.io as io
import dolfinx.plot as plot
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from meshes import generate_mesh_with_crack

def solve_elasticity(
    nu=0.3,
    E=1,
    load=1,
    Lx=1,
    Ly=0.5,
    Lcrack=0.3,
    lc=0.1,
    refinement_ratio=10,
    dist_min=0.2,
    dist_max=0.3,
    verbosity=10
):
    msh, mt, ft = generate_mesh_with_crack(
        Lcrack=Lcrack,
        Lx=Lx,
        Ly=Ly,
        lc=lc,  # caracteristic length of the mesh
        refinement_ratio=refinement_ratio,  # how much it is refined at the tip zone
        dist_min=dist_min,  # radius of tip zone
        dist_max=dist_max,  # radius of the transition zone
        verbosity=verbosity
    )
    element = ufl.VectorElement("Lagrange", msh.ufl_cell(), degree=1, dim=2)
    V = fem.FunctionSpace(msh, element)

    def bottom_no_crack(x):
        return np.logical_and(np.isclose(x[1], 0.0), x[0] > Lcrack)

    def right(x):
        return np.isclose(x[0], Lx)

    bottom_no_crack_facets = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, bottom_no_crack
    )
    bottom_no_crack_dofs_y = fem.locate_dofs_topological(
        V.sub(1), msh.topology.dim - 1, bottom_no_crack_facets
    )

    right_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, right)
    right_dofs_x = fem.locate_dofs_topological(
        V.sub(0), msh.topology.dim - 1, right_facets
    )
    bc_bottom = fem.dirichletbc(ScalarType(0), bottom_no_crack_dofs_y, V.sub(1))
    bc_right = fem.dirichletbc(ScalarType(0), right_dofs_x, V.sub(0))
    bcs = [bc_bottom, bc_right]

    dx = ufl.Measure("dx", domain=msh)
    top_facets = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], Ly))
    mt = mesh.meshtags(msh, 1, top_facets, 1)
    ds = ufl.Measure("ds", subdomain_data=mt)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    # this is for plane-stress
    lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)

    def eps(u):
        """Strain"""
        return ufl.sym(ufl.grad(u))

    def sigma(eps):
        """Stress"""
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)

    def a(u, v):
        """The bilinear form of the weak formulation"""
        return ufl.inner(sigma(eps(u)), eps(v)) * dx

    def L(v):
        """The linear form of the weak formulation"""
        # Volume force
        b = fem.Constant(msh, ScalarType((0, 0)))

        # Surface force on the top
        f = fem.Constant(msh, ScalarType((0, load)))
        return ufl.dot(b, v) * dx + ufl.dot(f, v) * ds(1)

    problem = fem.petsc.LinearProblem(
        a(u, v), L(v), bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()
    uh.name = "displacement"

    energy = fem.assemble_scalar(fem.form(0.5 * a(uh, uh) - L(uh)))
    print(f"The potential energy for Lcrack={Lcrack:2.3e} is {energy:2.3e}")
    sigma_ufl = sigma(eps(uh))
    return uh, energy, sigma_ufl


if __name__ == "__main__":
    from mpi4py import MPI

    uh, energy = solve_elasticity(
        Lx=1,
        Ly=0.5,
        Lcrack=0.3,
        lc=0.05,
        refinement_ratio=10,
        dist_min=0.2,
        dist_max=0.3,
    )

    with io.XDMFFile(
        MPI.COMM_WORLD, "output2/elasticity-demo.xdmf", "w"
    ) as file:
        file.write_mesh(uh.function_space.mesh)
        file.write_function(uh)
