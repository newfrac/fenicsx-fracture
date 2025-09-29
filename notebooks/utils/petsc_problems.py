from petsc4py import PETSc

import ufl
from dolfinx import fem


class SNESProblem:
    def __init__(self, F, u, bcs, J=None):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        if J is None:
            self.a = fem.form(ufl.derivative(F, u, du))
        else:
            self.a = fem.form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.petsc.assemble_vector(F, self.L)
        fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()
