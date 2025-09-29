import dolfinx
from petsc4py import PETSc


class SNES_problem:
    def __init__(self, F, J, u, bcs):
        """Definition of the class atributes"""

        self.bcs = bcs  # bcs = boundary conditions
        self.u = u  # u = solution
        self.L = dolfinx.fem.form(F)  # F = residual
        self.a = dolfinx.fem.form(J)  # J = jacobian

    def F(self, snes, x, b):
        """Assemble F into the vector b"""
        x.copy(self.u.x.petsc_vec)
        with b.localForm() as b_local:  # zero residual vector
            b_local.set(0.0)
        dolfinx.fem.petsc.apply_lifting(
            b, [self.a], [self.bcs], x0=[x], alpha=-1.0
        )
        dolfinx.fem.petsc.assemble_vector(b, self.L)
        dolfinx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x, A, P):
        """Assemble the Jacobian matrix"""

        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.a, self.bcs)
        A.assemble()
