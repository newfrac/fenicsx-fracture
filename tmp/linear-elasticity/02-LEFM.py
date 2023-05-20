# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: fenicsx-0.6.0
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
import dolfinx.mesh as mesh
import dolfinx.io as io
import dolfinx.plot as plot
import ufl

from mpi4py import MPI
from petsc4py import PETSc
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

Lcrack = 0.1
Lx = 1.0
Ly = 1.25
uh, energy, sigma_ufl = solve_elasticity(Lx=Lx,Ly=Ly,Lcrack=Lcrack,lc=0.05,refinement_ratio=20,dist_min=.1,dist_max=1)

from plots import warp_plot_2d
import pyvista

import ufl
sigma_iso = 1./3*ufl.tr(sigma_ufl)*ufl.Identity(len(uh))
sigma_dev =  sigma_ufl - sigma_iso
von_Mises = ufl.sqrt(3./2*ufl.inner(sigma_dev, sigma_dev))
V_dg = fem.FunctionSpace(uh.function_space.mesh, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_dg.element.interpolation_points())
vm_stress = fem.Function(V_dg)
vm_stress.interpolate(stress_expr)
plotter = warp_plot_2d(uh,cell_field=vm_stress,field_name="Von Mises stress", factor=.1,show_edges=True,clim=[0, 0.5],show_scalar_bar=False)
pyvista.OFF_SCREEN = False
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    pyvista.start_xvfb(wait=0.1)
    figure = plotter.screenshot(f"{outdir}/VonMises.png")

# %% [markdown]
# ## Crack opening displacement (COD)
#
# Let us get the vertical displacement at the crack lip

# %%
from evaluate_on_points import evaluate_on_points
xs = np.linspace(0,Lcrack,100)
ys = 0.0 * np.ones_like(xs)
zs = 0.0 * np.ones_like(xs)
points = np.array([xs,ys,zs])
u_values = evaluate_on_points(uh,points)
us = u_values[1][:,1]


# %%
plt.plot(xs,us,".")
plt.xlim([0.,Lcrack])
plt.xlabel("x - coordinate")
plt.ylabel(r"$u_y$")
plt.title("Crack opening displacement")
plt.savefig(f"{outdir}/COD.png")

# %% [markdown]
# As detailed in the lectures notes, we can estimate the value of the stress intensity factor $K_I$ by extrapolating $u \sqrt{2\pi/ r}$

# %%
r = (Lcrack-xs)
#r_ = r[np.where(xs<Lcrack)]
#us_ = us[np.where(xs<Lcrack)]

nu = 0.3
E = 1.0
mu = E / (2.0 * (1.0 + nu))
kappa = (3 - nu) / (1 + nu)
factor = 2 * mu / (kappa + 1)

plt.semilogx(r,us * np.sqrt(2*np.pi/r)*factor,".")
plt.xlabel("r")
plt.ylabel(r"${u_y} \,\frac{2\mu}{k+1} \,\sqrt{2\pi/r}$")
plt.title("Crack opening displacement")
plt.savefig(f"{outdir}/KI-COD.png")

# %% [markdown]
# We estimate $K_I\simeq 1.8\pm0.1$

# %% [markdown]
# ## Stress at the crack tip
#
# Let us get the stress around the crack tip

# %%
xs = np.linspace(Lcrack,2*Lcrack,1000)
ys = 0.0 * np.ones_like(xs)
zs = 0.0 * np.ones_like(xs)
points = np.array([xs,ys,zs])
r = (xs-Lcrack)
sigma_xx_expr = fem.Expression(sigma_ufl[0,0], V_dg.element.interpolation_points())
sigma_xx = fem.Function(V_dg)
sigma_xx.interpolate(sigma_xx_expr)
sigma_xx_values = evaluate_on_points(sigma_xx,points)[1]

# %%
plt.plot(r,sigma_xx_values[:,0],"o")
plt.xlabel("r")
plt.ylabel(r"$\sigma_{rr}$")
plt.title("Stress at the crack tip")
plt.savefig(f"{outdir}/stress.png")

# %% [markdown]
# As detailed in the lectures notes, we can estimate the value of the stress intensity factor $K_I$ by extrapolating $\sigma_{rr} \sqrt{2\pi r}$

# %%
plt.semilogx(r,sigma_xx_values[:,0]*np.sqrt(2*np.pi*r),"o")
plt.xlabel("r")
plt.ylabel(r"$\sigma_{rr}*\sqrt{2\pi\,r}$")
plt.title("Stress at the crack tip")
plt.savefig(f"{outdir}/KI-stress.png")


# %% [markdown]
# We can say that $K_I\simeq 1.5\pm .5$ as from the COD, but this estimate is not precise and reliable. 
#
# From Irwing formula in plane-stress, we get the energy release rate (ERR)
#

# %%
KI_estimate = 1.8 
G_estimate = KI_estimate ** 2 / E # Irwin's formula in plane stress
print(f"ERR estimate is {G_estimate}")

# %% [markdown]
# # The elastic energy release rate 

# %% [markdown]
# ## Naïf method: finite difference of the potential energy

# %% [markdown]
# Let us first calculate the potential energy for several crack lengths. We multiply the result by 2 to account for the symmetry when comparing with the $K_I$ estimate above.

# %%
Ls = np.linspace(Lcrack*.7,Lcrack*1.3,10)
energies = np.zeros_like(Ls)
Gs = np.zeros_like(Ls)
for (i, L) in enumerate(Ls):
    uh, energies[i], _ = solve_elasticity(Lx=Lx,Ly=Ly,Lcrack=L,lc=.05,refinement_ratio=10,dist_min=.1,dist_max=1.,verbosity=1)
    
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
plt.plot(Ls[0:-1], ERR_naif,"o")
plt.ylabel("ERR")
plt.xlabel("L_crack")
#plt.axhline(G_estimate,linestyle='--',color="gray")
#plt.axvline(Lcrack,linestyle='--',color="gray") 

# %% [markdown]
# # G-theta method: domain derivative
# This function implement the G-theta method to compte the ERR as described in the lecture notes.
#
# We first create by an auxiliary computation a suitable theta-field.
#
# To this end we solve an auxiliary problem solving for finding a $\theta$-field which is equal to $1$ in a disk around the crack tip and vanishing on the boundary.
# This field is used to define the "direction" for the domain derivative, which should change the crack length, but not the outer boundary.  
#
# Here we determine the $\theta$ field by solving the floowing problem
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
    
    V_theta = fem.FunctionSpace(domain,("Lagrange",1))
    
   
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
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "gmres", "pc_type": "gamg"})
    thetah = problem.solve()
    return thetah

crack_tip = np.array([Lcrack,0])
crack_tangent = np.array([1,0])
crack_tip = np.array([Lcrack,0])
R_int = Lcrack/4.
R_ext = Lcrack
domain = uh.function_space.mesh
thetah = create_theta_field(domain,crack_tip,R_int,R_ext)


# Plot theta
topology, cell_types, geometry = plot.create_vtk_mesh(thetah.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["theta"] = thetah.x.array.real
grid.set_active_scalars("theta")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.add_title("Theta-field")
plotter.view_xy()
plotter.show()

# %% [markdown]
# From the scalar field, we define a vector field by multiplying by the tangent vector to the crack: t=[1,0]

# %% [markdown]
# Hence, we can compute the ERR with the formula
# $$
# G  = \int_\Omega \left(\sigma(\varepsilon(u))\cdot(\nabla u\nabla\theta)-\dfrac{1}{2}\sigma(\varepsilon(u))\cdot \varepsilon(u) \mathrm{div}(\theta)\,\right)\mathrm{dx}$$

# %%
Lcrack = 0.1
Lx = 1.
Ly = 1.25
crack_tip = np.array([Lcrack,0])
crack_tangent = np.array([1,0])
crack_tip = np.array([Lcrack,0])
R_int = Lcrack/4
R_ext = Lcrack
uh, energy, sigma_ufl = solve_elasticity(Lx=Lx,Ly=Ly,Lcrack=Lcrack,lc=.02,refinement_ratio=30,dist_min=.1,dist_max=1.0)

thetah = create_theta_field(uh.function_space.mesh,crack_tip,R_int,R_ext)
eps_ufl = ufl.sym(ufl.grad(uh))
theta_vector = ufl.as_vector([1.,0.]) * thetah
dx = ufl.dx(domain=uh.function_space.mesh)
first_term = ufl.inner(sigma_ufl,ufl.grad(uh) * ufl.grad(theta_vector)) * dx
second_term = - 0.5 * ufl.inner(sigma_ufl,eps_ufl) * ufl.div(theta_vector) * dx

G_theta = 2 * fem.assemble_scalar(fem.form(first_term + second_term))
print(f'The ERR computed with the G-theta method is {G_theta:2.4f}' )

# %%

# %%
