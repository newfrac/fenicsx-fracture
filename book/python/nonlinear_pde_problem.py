from typing import List

from petsc4py import PETSc

import dolfinx
import ufl

class NonlinearPDEProblem:
    """Nonlinear problem class compatible with dolfinx.NewtonSolver.
    """

    def __init__(self, F: ufl.form.Form, J: ufl.form.Form, u: dolfinx.Function, bcs: List[dolfinx.DirichletBC]):
        """This class set up structures for solving a non-linear problem using Newton's method.

        Parameters
        ==========
        F: Residual.
        J: Jacobian.
        u: Solution.
        bcs: Dirichlet boundary conditions.
        """
        self.L = F
        self.a = J
        self.bcs = bcs

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.A = dolfinx.fem.create_matrix(self.a)
        self.b = dolfinx.fem.create_vector(self.L)

    def form(self, x: PETSc.Vec):
        """This function is called before the residual or Jacobian is computed inside the Newton step. 
        It is usually used to update ghost values.

        Parameters
        ==========
        x: Vector containing the latest solution.
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b. 

        Parameters
        ==========
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        # Zero the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.assemble_vector(b, self.L)
        # Apply boundary conditions
        dolfinx.fem.apply_lifting(b, [self.a], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        dolfinx.fem.assemble_matrix(A, self.a, self.bcs)
        A.assemble()



