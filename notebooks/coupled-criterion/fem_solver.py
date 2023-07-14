import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
import sys

sys.path.append("./pycodes_coupledcriterion")
from SNES_solver import *


def fem_solver(mesh, facets, Mechanical_data, Geometrical_data, dl):
    # ............ Parameters in the problem
    E, nu = Mechanical_data.get("E"), Mechanical_data.get("nu")

    # .............. Parallel computation: in this case, we work with just one processor
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    # ............ Elastic constant in the lame constitutive law
    lmbda = dolfinx.fem.Constant(mesh, E * nu / (1 + nu) / (1 - 2 * nu))
    mu = dolfinx.fem.Constant(mesh, E / 2 / (1 + nu))

    # ............ Elastic variable: stress and strains
    def eps(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return lmbda * ufl.tr(eps(v)) * ufl.Identity(2) + 2 * mu * eps(v)

    # ............. Function spaces
    V_u = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))

    # ............. Boundary condition - degrees of freedom: vertical line symmetry + imposed displacement
    hor_load_dofs = dolfinx.fem.locate_dofs_topological(
        V_u, facets.dim, facets.indices[facets.values == 1]
    )

    # ............ Boundary condition - degrees of freedom: horizontal symmetry ( x = [dl, H/2 - a/2] and y = 0 )
    def hor_sym_function(x):
        return np.logical_and(np.greater_equal(x[0], dl), np.isclose(x[1], 0))

    hor_sym_points = dolfinx.mesh.locate_entities_boundary(mesh, 0, hor_sym_function)
    hor_sym_dofs = dolfinx.fem.locate_dofs_topological(V_u.sub(1), 0, hor_sym_points)

    # ............. Boundary conditions - loading set
    bcs = [
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(mesh, 0.0), hor_sym_dofs, V_u.sub(1)
        ),
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 1))),
            hor_load_dofs,
            V_u,
        ),
    ]

    # ............. Boundary conditions - Neumann boundary conditions and body forces
    q = dolfinx.fem.Constant(mesh, np.zeros((2,)))
    f = dolfinx.fem.Constant(mesh, np.zeros((2,)))

    # ............. Functions
    u = dolfinx.fem.Function(V_u, name="Displacement")

    # ............. Definition of the elastic energy and the derivative
    functional = (
        ufl.inner(sigma(u), eps(u)) * ufl.dx
        - ufl.dot(f, u) * ufl.dx
        - ufl.dot(q, u) * ufl.ds
    )
    dfunctional = ufl.derivative(functional, u, ufl.TestFunction(V_u))
    ddfunctional = ufl.derivative(dfunctional, u, ufl.TrialFunction(V_u))

    # ............. Definition of the problem
    problem = SNES_problem(dfunctional, ddfunctional, u, bcs)

    # ............. b = xk (degrees of freedom)
    dofs_domain, dofs_borders = V_u.dofmap.index_map, V_u.dofmap.index_map_bs
    b = dolfinx.la.create_petsc_vector(dofs_domain, dofs_borders)
    J = dolfinx.fem.petsc.create_matrix(problem.a)

    # ............ Definition of the solver
    snes = PETSc.SNES().create()
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)
    snes.setTolerances(rtol=1.0e-10, max_it=10)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().setTolerances(1.0e-9)

    # ............ Execution of the solver // clear memory after
    snes.solve(None, u.vector)
    snes.destroy, b.destroy(), J.destroy()

    # ........... Representation of displacements function
    Path("output").mkdir(parents=True, exist_ok=True)
    if dl == 0:
        with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, "output/u_undamaged.xdmf", "w"
        ) as file:
            file.write_mesh(mesh)
            file.write_function(u)
    else:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/u_damaged.xdmf", "w") as file:
            file.write_mesh(mesh)
            file.write_function(u)

    # ............ Calculation of the elastic energy
    energy = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(functional)
    )  # calculation in 1/2 of the whole domain
    force = energy * 2 / Geometrical_data.get("H")

    # ............ Undamaged configuration (dl = 0)
    if dl == 0:
        vector_dl, stress, cells, points_on_proc = [], [], [], []

        # .................. Tensile stress in the undamaged configuration
        V_s = dolfinx.fem.TensorFunctionSpace(mesh, ("CG", 1))

        sig_tensor = dolfinx.fem.Function(V_s, name="Stress tensor")

        sig_expres = dolfinx.fem.Expression(
            sigma(u), V_s.element.interpolation_points()
        )
        sig_tensor.interpolate(sig_expres)

        # ................. Nodes in the expected crack path (vector_dl)
        for i in range(0, len(mesh.geometry.x)):
            if mesh.geometry.x[i, 1] == 0 and mesh.geometry.x[i, 0] > 0:
                vector_dl.append(mesh.geometry.x[i, 0])

        vector_dl = np.sort(vector_dl)

        # ................. Evaluation of stresses along the expected crack path
        points_eval = np.zeros((3, len(vector_dl)))
        points_eval[0] = vector_dl

        bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)

        cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points_eval.T)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(
            mesh, cell_candidates, points_eval.T
        )

        for i, point in enumerate(points_eval.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)

        tensile_sig = sig_tensor.eval(points_on_proc, cells)[:, 3]
        # ............Comment: nly in this particular case we omit the singularity at the other v-notch, and we reduce in one component the vectors         of the tensile stress
        return (
            energy,
            force,
            tensile_sig[0 : len(tensile_sig) - 2],
            vector_dl[0 : len(tensile_sig) - 2],
        )

    else:
        return energy, force
