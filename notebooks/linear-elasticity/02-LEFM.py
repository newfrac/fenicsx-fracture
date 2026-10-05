# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: fenicsx-fracture
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear Elasticity Fracture Mechanics
#
# *Authors:* 
# - *Laura De Lorenzis (ETH Zürich)*
# - *Veronique Lazarus (ENSTA, IPP)*
# - *Corrado Maurini (Sorbonne Université, corrado.maurini@sorbonne-universite.fr)*
#
# This notebook serves as a tutorial for linear elastic fracture mechanics
#

# %%
import sys
sys.path.append("../utils")

# Import required libraries
import matplotlib.pyplot as plt
import numpy as np

import dolfinx.fem as fem
import dolfinx.plot as plot
import dolfinx.fem.petsc
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType


plt.rcParams["figure.figsize"] = (6,3)

outdir = "output"
from pathlib import Path
Path(outdir).mkdir(parents=True, exist_ok=True)


# %% [markdown]
# # Asymptotic field and SIF ($K_I$)
#
# Let us first get the elastic solution for a given crack length 

# %%
from elastic_solver import solve_elasticity

Lx = 1.
Ly = 0.5
Lcrack = 0.3
lc =.05
refinement_ratio = 20
dist_min = .1
dist_max = .3
h_tip = lc / refinement_ratio  # the mesh size at the crack tip
uh, energy, sigma_ufl = solve_elasticity(Lx=Lx,
                                         Ly=Ly,
                                         Lcrack=Lcrack,
                                         lc=lc,
                                         refinement_ratio=refinement_ratio,
                                         dist_min=dist_min,
                                         dist_max=dist_max,
                                         verbosity=1)

from plots import warp_plot_2d
import pyvista
pyvista.set_jupyter_backend("static")

sigma_iso = 1./3*ufl.tr(sigma_ufl)*ufl.Identity(len(uh))
sigma_dev =  sigma_ufl - sigma_iso
von_Mises = ufl.sqrt(3./2*ufl.inner(sigma_dev, sigma_dev))
V_dg = fem.functionspace(uh.function_space.mesh, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_dg.element.interpolation_points)
vm_stress = fem.Function(V_dg)
vm_stress.interpolate(stress_expr)

plotter = warp_plot_2d(uh,cell_field=vm_stress,field_name="Von Mises stress", factor=.1,show_edges=True,clim=[0.0, 1.0],show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot(f"{outdir}/VonMises.png")

# %% [markdown]
# ## Crack opening displacement (COD)
#
# Let us get the vertical displacement at the crack lip

# %%
from evaluate_at_points import evaluate_at_points
xs = np.linspace(0,Lcrack * 1.2 ,100)
ys = 0.0 * np.ones_like(xs)
zs = 0.0 * np.ones_like(xs)
points = np.array([xs,ys,zs])
u_values = evaluate_at_points(points,uh)
us = u_values[:,1]
plt.plot(xs,us,".")
plt.xlim([0.,Lcrack *1.2])
plt.xlabel("x - coordinate")
plt.ylabel(r"$u_y$")
plt.title("Crack opening displacement")
plt.savefig(f"{outdir}/COD.png")

# %% [markdown]
# As detailed in the lecture notes, we can estimate the value of the stress intensity factor $K_I$ by extrapolating $u \sqrt{2\pi/ r}$

# %%
# the distance to the tip, on the crack faces only: ahead of the tip the
# opening is zero and the extrapolation is meaningless
behind = xs < Lcrack
r = Lcrack - xs[behind]

nu = 0.3
E = 1.0
mu = E / (2.0 * (1.0 + nu))
kappa = (3 - nu) / (1 + nu)
factor = 2 * mu / (kappa + 1)

KI_cod = us[behind] * np.sqrt(2*np.pi/r) * factor

plt.semilogx(r,KI_cod,".")
plt.xlabel("r")
plt.ylabel(r"${u_y} \,\frac{2\mu}{k+1} \,\sqrt{2\pi/r}$")
plt.title("Crack opening displacement")
plt.savefig(f"{outdir}/KI-COD.png")

# the plateau of the curve above: far enough from the tip for the mesh to
# resolve the field, close enough for the asymptotic term to dominate
plateau = np.logical_and(r > 2 * h_tip, r < Lcrack / 4)
KI_from_cod = KI_cod[plateau].mean()
print(f"K_I from the COD is {KI_from_cod:2.4f}")

# %% [markdown]
# The curve flattens between the mesh size at the tip and a quarter of the crack length: that plateau is the stress intensity factor. We estimate $K_I\simeq 1.7$.

# %% [markdown]
# ## Stress at the crack tip
#
# Let us get the stress around the crack tip

# %%
xs = np.linspace(Lcrack,2*Lcrack,1000)
ys = 0.0 * np.ones_like(xs)
zs = 0.0 * np.ones_like(xs)
points = np.array([xs,ys,zs])
r_ahead = (xs-Lcrack)
sigma_yy_expr = fem.Expression(sigma_ufl[1,1], V_dg.element.interpolation_points)
sigma_yy = fem.Function(V_dg)
sigma_yy.interpolate(sigma_yy_expr)
sigma_yy_values = evaluate_at_points(points,sigma_yy)[:,0]

# %%
plt.plot(r_ahead,sigma_yy_values,"o")
plt.xlabel("r")
plt.ylabel(r"$\sigma_{yy}$")
plt.title("Stress at the crack tip")
plt.savefig(f"{outdir}/stress.png")

# %% [markdown]
# As detailed in the lecture notes, we can estimate the value of the stress intensity factor $K_I$ by extrapolating $\sigma_{yy} \sqrt{2\pi r}$.
#
# Ahead of the tip, on $\theta=0$, the asymptotic field gives $\sigma_{xx}=\sigma_{yy}=K_I/\sqrt{2\pi r}$, so either component carries the same leading term. The next term of the expansion, the T-stress, adds to $\sigma_{xx}$ alone, which is why $\sigma_{yy}$ is extrapolated here.

# %%
KI_stress = sigma_yy_values * np.sqrt(2*np.pi*r_ahead)

plt.semilogx(r_ahead,KI_stress,"o")
plt.xlabel("r")
plt.ylabel(r"$\sigma_{yy}\,\sqrt{2\pi\,r}$")
plt.title("Stress at the crack tip")
plt.savefig(f"{outdir}/KI-stress.png")

plateau_ahead = np.logical_and(r_ahead > 2 * h_tip, r_ahead < Lcrack / 4)
print(f"K_I from the stress is {KI_stress[plateau_ahead].mean():2.4f}")


# %% [markdown]
# This second estimate agrees with the one read on the crack opening, but it is less precise: the stress is the derivative of the field the computation solves for, it is constant over each cell, and it is the quantity that the singularity makes unbounded.
#
# From Irwin's formula in plane-stress, we get the energy release rate (ERR)
#

# %%
KI_estimate = KI_from_cod
G_estimate = KI_estimate ** 2 / E # Irwin's formula in plane stress
print(f"ERR estimate is {G_estimate:2.4f}")

# %% [markdown]
# # The elastic energy release rate 

# %% [markdown]
# ## Naïf method: finite difference of the potential energy

# %% [markdown]
# Let us first calculate the potential energy for several crack lengths. We multiply the result by `2` to account for the symmetry when comparing with the $K_I$ estimate above.

# %%
Ls = np.linspace(Lcrack*.7,Lcrack*1.3,10)
energies = np.zeros_like(Ls)
for (i, L) in enumerate(Ls):
    uh, energies[i], _ = solve_elasticity(Lx=Lx,
                                          Ly=Ly,
                                          Lcrack=L,
                                          lc=.05,
                                          refinement_ratio=10,
                                          dist_min=.1,
                                          dist_max=1.,
                                          verbosity=1)
    
energies = energies * 2

# %% [markdown]
# We can estimate the ERR by taking the finite-difference approximation of the derivative

# %%
ERR_naif = -np.diff(energies)/np.diff(Ls)

plt.figure()
plt.plot(Ls, energies,"*")
plt.xlabel("L_crack")
plt.ylabel("Potential energy")
plt.figure()
plt.plot(Ls[0:-1], ERR_naif,"-")
plt.ylabel("ERR")
plt.xlabel("L_crack")
plt.axhline(G_estimate,linestyle='--',color="gray")
plt.axvline(Lcrack,linestyle='--',color="gray") 


# %% [markdown]
# # G-theta method: domain derivative
# This function implements the G-theta method to compute the ERR as described in the lecture notes (see https://gitlab.com/newfrac/CORE-school/newfrac-core-numerics/-/blob/master/Core_School_numerical_NOTES.pdf?ref_type=heads).
#
# We first create by an auxiliary computation a suitable theta-field.
#
# To this end, we solve an auxiliary problem for finding a $\theta$-field which is equal to $1$ in a disk around the crack tip and vanishing on the boundary.
# This field defines the "direction" for the domain derivative, which should change the crack length, but not the outer boundary.  
#
# Here we determine the $\theta$ field by solving the following problem
#
# $$
# \Delta \theta = 0\quad \text{for}\quad x\in\Omega,
# \quad \theta=1\quad \text{for} \quad x\in \mathrm{D}\equiv\{\Vert x-x_\mathrm{tip}\Vert<R_{\mathrm{int}}\},
# \quad \theta=0\quad \text{for} \quad x\in\partial\Omega, \;\Vert x-x_\mathrm{tip}\Vert>R_{\mathrm{ext}}
# $$
#
# This is implemented in the function below

# %%
def create_theta_field(domain,crack_tip,R_int,R_ext):
    
    def tip_distance(x):
          return np.sqrt((x[0]-crack_tip[0])**2 + (x[1]-crack_tip[1])**2) 
    
    V_theta = fem.functionspace(domain,("Lagrange",1))
    
   
    # Define variational problem to define the theta-field. 
    # We solve a simple laplacian
    theta, theta_ = ufl.TrialFunction(V_theta), ufl.TestFunction(V_theta)
    a = ufl.dot(ufl.grad(theta), ufl.grad(theta_)) * ufl.dx
    L = fem.Constant(domain,ScalarType(0.)) * theta_ * ufl.dx(domain=domain) 

    # Set the BCs
    # Imposing 1 in the inner circle and zero in the outer circle
    dofs_inner = fem.locate_dofs_geometrical(V_theta,lambda x : tip_distance(x) < R_int)
    dofs_out = fem.locate_dofs_geometrical(V_theta,lambda x : tip_distance(x) > R_ext)
    bc_inner = fem.dirichletbc(ScalarType(1.),dofs_inner,V_theta)
    bc_out = fem.dirichletbc(ScalarType(0.),dofs_out,V_theta)
    bcs = [bc_out, bc_inner]

    # solve the problem
    problem = fem.petsc.LinearProblem(a, L, petsc_options_prefix="theta_", bcs=bcs, petsc_options={"ksp_type": "gmres", "pc_type": "gamg"})
    thetah = problem.solve()
    return thetah


# the solution this section works on: the theta-field is built on its mesh, and
# the energy release rate is the integral of its stress. Solving again here is
# what keeps the two together: `uh` currently holds the last crack length of
# the finite-difference loop above, not `Lcrack`.
uh, energy, sigma_ufl = solve_elasticity(Lx=Lx,Ly=Ly,Lcrack=Lcrack,lc=.05,refinement_ratio=30,dist_min=.1,dist_max=1.0)

# the outer circle must stay inside the domain: centred on the tip, a radius of
# Lcrack would reach the left edge, and theta would not vanish there
crack_tip = np.array([Lcrack,0])
R_int = Lcrack/8
R_ext = Lcrack/2
thetah = create_theta_field(uh.function_space.mesh,crack_tip,R_int,R_ext)


# Plot theta
topology, cell_types, geometry = plot.vtk_mesh(thetah.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["theta"] = thetah.x.array.real
grid.set_active_scalars("theta")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.add_title("theta-field")
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()

# %% [markdown]
# From the scalar field, we define a vector field by multiplying by the tangent vector to the crack: t=[1,0]

# %% [markdown]
# Hence, we can compute the ERR with the formula
# $$
# G  = \int_\Omega \left(\sigma(\varepsilon(u))\cdot(\nabla u\nabla\theta)-\dfrac{1}{2}\sigma(\varepsilon(u))\cdot \varepsilon(u) \mathrm{div}(\theta)\,\right)\mathrm{dx}$$

# %%
eps_ufl = ufl.sym(ufl.grad(uh))
theta_vector = ufl.as_vector([1.,0.]) * thetah
dx = ufl.dx(domain=uh.function_space.mesh)
first_term = ufl.inner(sigma_ufl,ufl.grad(uh) * ufl.grad(theta_vector)) * dx
second_term = - 0.5 * ufl.inner(sigma_ufl,eps_ufl) * ufl.div(theta_vector) * dx

# the factor 2 accounts for the symmetry: only half of the slab is meshed.
# each process assembles its own part of the integral
G_theta = 2 * uh.function_space.mesh.comm.allreduce(
    fem.assemble_scalar(fem.form(first_term + second_term)), op=MPI.SUM
)
print(f'The ERR computed with the G-theta method is {G_theta:2.4f}' )

# %% [markdown]
# ## The result does not depend on the $\theta$ field
#
# Any $\theta$ field with the two properties above gives the same $G$. The radii below span a factor of eight and the integration annulus moves across the mesh, yet the four values agree to the fourth digit. This is the check to run on a G-theta implementation: an error in the formula shows up here as a spread.

# %%
for R_int_i, R_ext_i in [(Lcrack/8, Lcrack/2), (Lcrack/16, Lcrack/4),
                         (Lcrack/4, 0.9*Lcrack), (Lcrack/2, 0.95*Lcrack)]:
    theta_i = create_theta_field(uh.function_space.mesh,crack_tip,R_int_i,R_ext_i)
    tv = ufl.as_vector([1.,0.]) * theta_i
    G_i = 2 * uh.function_space.mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            ufl.inner(sigma_ufl, ufl.grad(uh) * ufl.grad(tv)) * dx
            - 0.5 * ufl.inner(sigma_ufl, eps_ufl) * ufl.div(tv) * dx
        )), op=MPI.SUM
    )
    print(f"R_int = {R_int_i:.4f}, R_ext = {R_ext_i:.4f}:  G = {G_i:2.4f}")

# %% [markdown]
# **Note:**
# The $\theta$ field represents how the domain is "varied" to take the domain derivative. It must be 1 on the tip and zero on the boundary of the domain. The choice of the field is otherwise arbitrary. With the generation method above, given the crack length `Lcrack`, the disk radius `R_ext` should be chosen such that the disk does not intersect the boundary of the domain, which here means `R_ext < Lcrack`: the tip is at a distance `Lcrack` from the left edge. The internal radius `R_int` can be chosen as a few times the mesh size at the tip, for example.

# %% [markdown]
#
