# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: fenicsx-fracture
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Numerical implementation of the elastodynamics problem
#
# __Authors__: A. Chao Correas (arturo.chaocorreas@polito.it) and C. Maurini (corrado.maurini@sorbonne-universite.fr)

# %% [markdown]
# In this notebook we will present the dolfinx implementation of the elastodynamics problem defined in the _Elastodynamics Theory_ notebook. The coding contained in this notebook is executable in parallel. To run the notebook in parallel, we are going to set up an `ipyparallel.Cluster` using `mpi` and `n` processors. In case the  `ipyparallel` library is not available in your system, you can use `!pip install ipyparallel`

# %%
import logging
import ipyparallel as ipp

# create a cluster
rc = ipp.Cluster(engines="mpi", n=2, log_level=logging.WARNING).start_and_connect_sync(activate=True)

# %% [markdown]
# Once the cluster is set up, let us import the libraries required for running the script and set the stage. `sympy` is the only one that does not come installed in the dolfinx environment by default, so if it was not previously installed in the executing environment, it will raise an error. To solve this, you can for instance run `pip install sympy` beforehand. The `%%px` magic command in the first row of the cells indicates the cluster to run those.

# %%
# %%px

import dolfinx, ufl
import numpy as np
import sympy as sp
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
import sys
import dolfinx.fem.petsc
#from mesh_plotter import mesh_plotter

sys.path.append("../utils/")
from petsc_problems import SNESProblem

comm = MPI.COMM_WORLD

# %% [markdown]
# _Insert schematic image_

# %% [markdown]
# Then, we determine the input parameters of our problem which control the geometry, material, mesh, time stepping and output preferences. By defining them within dictionaries the subsequent code is more readable and the input information can be stored in a `.json` file for the record. In particular, this code will make use of a rectangular domain of size $L_{x} \times L_{y}$ centered at $(0,0)$, filled with a linear elastic material of density $\rho$ whose quasistatic behavior is defined by its Young's modulus $E$ and Poisson's ratio $\nu$. This domain will be clamped on its left edge and have an imposed horizontal motion on the right side, so that: 
#
# $$
# \underline{u} (\underline{x},t) 
# = 
# \dot{\underline{u}} (\underline{x},t)  
# = 
# \ddot{\underline{u}} (\underline{x},t) 
# = 
# \underline{0} 
# \quad \forall \left\{ \underline{x}, t \mid x = -\frac{L_{x}}{2}\right\}
# $$
#
#
# $$
# u_{x} (\underline{x},t) 
# = 
# f(t)
# =
# \begin{cases} 
# 0 & \mathrm{if} \quad t < T_{0} \\ 
# \frac{U}{2}\left[ 1-\cos \left( \frac{t-T_{0}}{T_{1}-T_{0}}\pi\right)  \right] & T_{0} \le t \le T_{1} \\
# U_{t} & t > T_{1}
# \end{cases},
# $$
# $$
# \dot{u}_{\underline{x},t} 
# = 
# \dot{f}(t)
# , \quad
# \ddot{u}_{\underline{x},t} 
# = 
# \ddot{f}(t)
# \quad \forall \left\{ \underline{x}, t \mid x = \frac{L_{x}}{2}\right\}
# $$
#
# Likewise, no external force is to be considered:
#
# $$
# \underline{f} 
# = 
# \underline{0}
# \quad \forall \left\{\underline{x},t \mid \underline{x} \in \partial_{f}\Omega\right\}
# $$
# $$
# \underline{b} 
# = 
# \underline{0}
# \quad \forall \left\{\underline{x},t \mid \underline{x} \in \partial_{f}\Omega\right\}
# $$
#

# %%

# %%
# %%px

# Characteristic dimensions of the rectangular domain
geometry_parameters = {'Lx': 1., 
                       'Ly': 0.1}

# Regularization length               
material_properties = {'E':   3e3, 
                       'nu':  0.3, 
                       'rho': 1.8e-9,
                       'c1':  1e-5,
                       'c2':  5e-5}

# Mesh control
mesh_parameters = {'nx': 50,
                   'ny': 4}

# Time stepping control
timestepping_parameters = {'initial_time': 0., 
                           'total_time':   20e-6, 
                           'delta_t':      1.e-8}

# Time integration control (Newmark's \beta-method)
timeintegration_parameters = {'beta': 0.25,
                              'gamma': 0.5}

# Output parameters
OTP_settings = {'xdmf_filename': f"implicit_elastodynamics.xdmf"}

# Top facet displacement
T_0 = 0.
T_1 = 2.5e-6
U_t = 1e-2

t_sp = sp.Symbol('t', real = True)
U_imp = sp.Piecewise((0, t_sp<T_0),
                     (0.5*U_t*(1-sp.cos(((t_sp-T_0)/(T_1-T_0))*sp.pi)), t_sp<=T_1),
                     (U_t, True)) 
V_imp = sp.diff(U_imp, t_sp)
A_imp = sp.diff(V_imp, t_sp)

# Body forces
b_ = np.asarray([0.,0.])

# %% [markdown]
# As seen, using the symbolic math library `sympy` to define the imposed motion in the right facet results convenient for the differentiation w.r.t. time is automatically handled. Plotting these against time shows interesting features, for the imposed displacement function is continuous and smooth while the imposed velocity is continuous but non-smooth and the imposed acceleration is not continuous. 

# %%
# %%px

if comm.rank == 0:
    t_sampling = np.linspace(timestepping_parameters['initial_time'], 
                             timestepping_parameters['total_time'], 
                             1000) 
    U_imp_sampling = np.zeros_like(t_sampling)
    V_imp_sampling = np.zeros_like(t_sampling)
    A_imp_sampling = np.zeros_like(t_sampling)
    
    for i in enumerate (t_sampling):
        U_imp_sampling[i[0]] = U_imp.subs({t_sp:t_sampling[i[0]]})
        V_imp_sampling[i[0]] = V_imp.subs({t_sp:t_sampling[i[0]]})
        A_imp_sampling[i[0]] = A_imp.subs({t_sp:t_sampling[i[0]]})
        
    fig, ax = plt.subplots(1, 3, figsize=(18,6))
    ax[0].plot(t_sampling, U_imp_sampling, ls='none', marker='.')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Imposed X Displacement [mm]')
    ax[1].plot(t_sampling, V_imp_sampling, ls='none', marker='.')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Imposed X Velocity [mm/s]')
    ax[2].plot(t_sampling, A_imp_sampling, ls='none', marker='.')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Imposed X Acceleration [mm/s2]');

# %% [markdown]
# Since the geometry is rather simple, the `create_rectangle` built-in function of `dolfinx.mesh` is used, creating a bidimensional mesh with $2 n_{x} \times n_{y}$ triangular elements. Over this mesh, the different geometrical regions are identified thorugh the use of meshtags, so that the nodes in the top, bottom, right and left facets have a integer tag equal to $1$, $2$, $3$ and $4$, respectively.

# %%
# %%px

mesh = dolfinx.mesh.create_rectangle(comm, 
                                     [np.array([-geometry_parameters['Lx']/2, -geometry_parameters['Ly']/2]), 
                                      np.array([+geometry_parameters['Lx']/2, +geometry_parameters['Ly']/2])], 
                                     [mesh_parameters['nx'], mesh_parameters['ny']], 
                                     dolfinx.mesh.CellType.triangle)
gdim = mesh.topology.dim
fdim = gdim - 1 
#mesh_plotter(mesh)

# %%
# %%px

# Geometrical regions  
def top(x):
    return np.isclose(x[1], +geometry_parameters["Ly"]/2)
def bottom(x):
    return np.isclose(x[1], -geometry_parameters["Ly"]/2)
def right (x):
    return np.isclose(x[0], +geometry_parameters["Lx"]/2)
def left (x):
    return np.isclose(x[0], -geometry_parameters["Lx"]/2)

# Geometrical sets
top_facets    = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)
bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom)
right_facets  = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right)
left_facets   = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)

tagged_facets = np.hstack([top_facets, 
                           bottom_facets, 
                           right_facets, 
                           left_facets])

tag_values = np.hstack([np.full_like(top_facets,    1), 
                        np.full_like(bottom_facets, 2),
                        np.full_like(right_facets,  3),
                        np.full_like(left_facets,   4)])

tagged_facets_sorted = np.argsort(tagged_facets)

mt = dolfinx.mesh.meshtags(mesh, fdim, 
                           tagged_facets[tagged_facets_sorted], 
                           tag_values[tagged_facets_sorted])

# Domain and subdomain measures
dx  = ufl.Measure("dx", domain=mesh)                   
ds  = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
n   = ufl.FacetNormal(mesh)                            

# %% [markdown]
# On top of the geometrical discretization already presented, a first order Continuous Galerkin finite element of dimension 2 will be used for interpolating the displacement, velocity and acceleration fields. Please note that the three vector fields share the same generalized affine space $\mathrm{V}(t)$ that contains any bidimensional vector field defined in $\Omega$, below represended by the variable `V_t`. It is also a good practise to create a `state` dictionary to simplify passing the references of the different fields in the problem to potential auxiliary functions.  

# %%
# %%px

# --------- Main functions and function spaces
V_t = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
u = dolfinx.fem.Function(V_t, name="Displacement")
u_new = dolfinx.fem.Function(V_t)
v = dolfinx.fem.Function(V_t, name="Velocity")
v_new = dolfinx.fem.Function(V_t)
a = dolfinx.fem.Function(V_t, name="Acceleration")
a_new = dolfinx.fem.Function(V_t)

# --------- State of each field 
state     = {"u": u, 
             "v": v, 
             "a": a}
state_new = {"u_new": u_new, 
             "v_new": v_new, 
             "a_new": a_new}

# %% [markdown]
# We now define the dirichlet boundary conditions for the displacement, velocity and acceleration fields, and store it in `bc_u`, `bc_v` and `bc_a`, respectively. Please notice that the values defined within the `dolfinx.fem.Constant` instances are just used as references to create the correct data structure in each case, and their proper values will be assigned afterwards.

# %%
# %%px

# Clamped left (ux=uy=0, vx=vy=0, ax=ay=0) 
left_u = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0,0)))
left_v = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0,0)))
left_a = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0,0)))
blocked_dofs_left_Vt = dolfinx.fem.locate_dofs_topological(V_t, fdim, left_facets)
bc_u_left = dolfinx.fem.dirichletbc(left_u, blocked_dofs_left_Vt, V_t)
bc_v_left = dolfinx.fem.dirichletbc(left_v, blocked_dofs_left_Vt, V_t)
bc_a_left = dolfinx.fem.dirichletbc(left_a, blocked_dofs_left_Vt, V_t)

# Imposed displacement right (ux=U_imp(t), vx=U_imp'(t), ax=U_imp''(t))
right_ux = dolfinx.fem.Constant(mesh,PETSc.ScalarType(0))
right_vx = dolfinx.fem.Constant(mesh,PETSc.ScalarType(0))
right_ax = dolfinx.fem.Constant(mesh,PETSc.ScalarType(0))
right_boundary_dofs_Vtx = dolfinx.fem.locate_dofs_topological(V_t.sub(0), fdim, right_facets)
bc_ux_right = dolfinx.fem.dirichletbc(right_ux, right_boundary_dofs_Vtx, V_t.sub(0))
bc_vx_right = dolfinx.fem.dirichletbc(right_vx, right_boundary_dofs_Vtx, V_t.sub(0))
bc_ax_right = dolfinx.fem.dirichletbc(right_ax, right_boundary_dofs_Vtx, V_t.sub(0))

# Collect the BCs
bcs_u = [bc_u_left, bc_ux_right]
bcs_v = [bc_v_left, bc_ux_right]
bcs_a = [bc_a_left, bc_ax_right]

# %% [markdown]
# At this point the stage is ready to define the variational formulation of the problem. For the sake of simplicity, a linear elastic behavior is here implemented, although the code here provided supports more advanced material models by just modifying the definition in the `elastic_energy_density` instance. Likewise, different definitions of the dissipative power can be introduced without further code changes by modifying `dissipated_power_density`.

# %%
# %%px

t       = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
delta_t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))

# Material properties
E   = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1))
nu  = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
rho = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1))
c1  = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
c2  = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))

# Body forces
b = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0,0)))

## Kinetic energy density
kinetic_energy_density = 0.5 * rho * ufl.inner(v,v)

### Dissipated energy density
eps_v = ufl.variable(ufl.sym(ufl.grad(v)))
dissipated_power_density = 0.5 * (c1 * ufl.inner(v,v) + c2 * ufl.inner(eps_v,eps_v))

# Lame constants (Plane strain)
mu    = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# Infinitesimal strain tensor
eps = ufl.variable(ufl.sym(ufl.grad(u)))

## Strain energy density (Linear elastic)
elastic_energy_density = lmbda / 2 * ufl.tr(eps) ** 2 + mu * ufl.inner(eps,eps)

# Stress tensor
sigma = ufl.diff(elastic_energy_density, eps) + c2*eps_v

## External workd density
external_work_density = ufl.dot(b,u)

# System's energy components
kinetic_energy   = kinetic_energy_density * dx
dissipated_power = dissipated_power_density * dx
elastic_energy   = elastic_energy_density * dx 
external_work    = external_work_density * dx 

potential_energy = elastic_energy - external_work
total_energy     = kinetic_energy + potential_energy

# Energy derivatives
u_test = ufl.TestFunction(V_t)
P_du = ufl.derivative(potential_energy, u, u_test) 
K_dv = ufl.derivative(kinetic_energy,   v, u_test)
Q_dv = ufl.derivative(dissipated_power, v, u_test)

# Residual 
Res = ufl.replace(K_dv, {v: a}) + Q_dv + P_du

# %% [markdown]
# Now, the residual of the elastodynamic problem is stored in `Res`. Nonetheless, as above mentioned, for the elastodynamics residual depends on the displacement, velocity and acceleration fields, it cannot be directly solved and the time integrator is needed. In this case, Newmark's $\beta$-method is used, so that the displacement and acceleration are propagated using the functions `u_newmark` and `v_newmark`. Then, using the function `ufl.replace`, we can substitute `a` by `a_new`, `v` by `v_newmark` and `u` by `u_newmark` in `Res` to obtain the Newmark's approximation of the residual in the _new_ instant, `Res_newmark`, which in turn is a function of `a_new` and admits `beta`, `gamma`, `delta_t`, `u`, `v` and `a` as parameters. 

# %%
# %%px

beta  = timeintegration_parameters['beta']
gamma = timeintegration_parameters['gamma']

def u_newmark(u,v,a,a_new,delta_t=delta_t,beta=beta):
    return u + delta_t * v + 0.5 * delta_t ** 2 * ((1 - 2 * beta) * a + 2 * beta * a_new)

def v_newmark(v,a,a_new,delta_t=delta_t,gamma=gamma):
    return (v + delta_t * ((1 - gamma) * a + gamma * a_new))

Res_acc_newmark = ufl.replace(Res, {a: a_new, 
                                    v: v_newmark(v, a, a_new, delta_t=delta_t, gamma=gamma),                                  
                                    u: u_newmark(u, v, a, a_new, delta_t=delta_t,beta=beta)})

# %% [markdown]
# For solving the acceleration problem, a `PETSc.SNES` solver is setup. Since the acceleration problem here considered is linear, the `ksponly` solver type is used. For non-linear cases, other more advanced specifications should be used, such as `newtonls` (for more information on PETSc SNES solvers for python, visit https://petsc4py.readthedocs.io/en/stable/manual/snes/).

# %%
# %%px
## --------- Acceleration explicit problem ----------------------------------------------------
Res_acc_newmark_da = ufl.derivative(Res_acc_newmark, a_new, ufl.TrialFunction(V_t)) 

acc_problem = SNESProblem(Res_acc_newmark, a_new, bcs_a, J=Res_acc_newmark_da)
b_vector = dolfinx.fem.petsc.create_vector(V_t)
J_matrix  = dolfinx.fem.petsc.create_matrix(dolfinx.fem.form(acc_problem.a))

# Acceleration Solver (with consistent mass matrix)
solver_acc_snes = PETSc.SNES().create()
solver_acc_snes.setType("ksponly")
solver_acc_snes.setFunction(acc_problem.F, b_vector)
solver_acc_snes.setJacobian(acc_problem.J, J_matrix)
solver_acc_snes.setTolerances(rtol=1.0e-9, max_it=15)
solver_acc_snes.getKSP().setType("preonly")
solver_acc_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_acc_snes.getKSP().getPC().setType("lu")

# %% [markdown]
# Once everything is defined, we can define the forms of the different dynamic magnitudes that we will use as output of our problem, in this case all the relevant energetic components plus the horizontal component of the reaction force. The latter is here calculated out of the stresses for the sake of simplicity, despite the most accurate way is through the proper integration of the residual at the Dirichlet boundary.

# %%
# %%px

kinetic_energy_form       = dolfinx.fem.form(kinetic_energy)
dissipated_power_form     = dolfinx.fem.form(dissipated_power)
elastic_energy_form       = dolfinx.fem.form(elastic_energy)
potential_energy_form     = dolfinx.fem.form(potential_energy)
total_energy_form         = dolfinx.fem.form(potential_energy + kinetic_energy)
reaction_force_right_form = dolfinx.fem.form(ufl.inner(n,sigma*n)*ds(3))

# %% [markdown]
# So far, all we have defined is the mathematical "skeleton" of our problem, and no numerical values have been assigned to any of the variables involved, passing them just as references. Now, we can assign the respective numerical values and perform the iterative resolution of the problem.

# %%
# %%px

# Initialization
t0         = timestepping_parameters['initial_time']
delta_t0   = timestepping_parameters['delta_t']
total_time = timestepping_parameters["total_time"]

E.value   = material_properties["E"]
nu.value  = material_properties["nu"]
rho.value = material_properties["rho"]
c1.value  = material_properties["c1"]
c2.value  = material_properties["c2"]

b.value = b_

t.value = t0

for const in [left_u,left_v,left_a]:
    const.value = (0.,0.)

for func in [u,v,a] :
    func.x.array[:] = 0.

stp_cont = 0    
ts = []
kinetic_energies       = []
dissipated_energies    = []
dissipated_energy      = 0
elastic_energies       = []
external_works         = []
potential_energies     = []
reaction_force_right_s = []
    
with dolfinx.io.XDMFFile(comm,'tmp.xdmf','w') as xdmf_file:
    xdmf_file = dolfinx.io.XDMFFile(mesh.comm, OTP_settings['xdmf_filename'], "w")
    xdmf_file.write_mesh(mesh)
    xdmf_file.write_function(u, t.value)

if comm.rank == 0:
    print ('RESOLUTION STATUS')
    sys.stdout.flush()

while t.value < total_time :
    
    #if comm.rank == 0:
    #    print(f"Solving for t = {t.value}")
    #    sys.stdout.flush()
    
    right_ux.value = float(U_imp.subs({t_sp:t.value}))
    right_vx.value = float(V_imp.subs({t_sp:t.value}))
    right_ax.value = float(A_imp.subs({t_sp:t.value}))

    solver_acc_snes.solve(None, a_new.x.petsc_vec)
    a_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    u_new.x.array[:] = u_newmark(u.x.array, v.x.array, a.x.array, a_new.x.array,delta_t=float(delta_t))
    dolfinx.fem.set_bc(u_new.x.petsc_vec,bcs_u)
    u_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    v_new.x.array[:] = v_newmark(v.x.array, a.x.array, a_new.x.array, delta_t=float(delta_t))
    dolfinx.fem.set_bc(v_new.x.petsc_vec,bcs_v)
    v_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
     # Copy i+1 into i
    u.x.array[:] = u_new.x.array
    v.x.array[:] = v_new.x.array
    a.x.array[:] = a_new.x.array
    
    dissipated_energy += float(comm.allreduce(float(delta_t)*dolfinx.fem.assemble_scalar(dissipated_power_form), op=MPI.SUM)) 
    
    t.value += float(delta_t.value)
    delta_t.value = float(ufl.conditional(ufl.lt(delta_t0,total_time-t), 
                                                 delta_t0,
                                                 total_time - t))
    
    kinetic_energies = np.concatenate((kinetic_energies,
                             [comm.allreduce(dolfinx.fem.assemble_scalar(kinetic_energy_form), op=MPI.SUM)]))
    dissipated_energies = np.concatenate((dissipated_energies,[dissipated_energy]))
    elastic_energies = np.concatenate((elastic_energies,
                            [comm.allreduce(dolfinx.fem.assemble_scalar(elastic_energy_form), op=MPI.SUM)]))
    potential_energies = np.concatenate((potential_energies,
                            [comm.allreduce(dolfinx.fem.assemble_scalar(potential_energy_form), op=MPI.SUM)]))
    reaction_force_right_s = np.concatenate((reaction_force_right_s,
                            [comm.allreduce(dolfinx.fem.assemble_scalar(reaction_force_right_form), op=MPI.SUM)]))
    ts = np.concatenate((ts,[t.value]))
        
    if comm.rank == 0:
        if stp_cont != round((100*t.value/total_time)):
            stp_cont = round((100*t.value/total_time))
            print (f"="*stp_cont+"> "+str(stp_cont)+"%", end="\r") 
            sys.stdout.flush()
            
           # with dolfinx.io.XDMFFile(comm,OTP_settings['xdmf_filename'],'a') as xdmf_file:
           #     xdmf_file.write_function(u, t.value)            

# %%
# %%px

if comm.rank == 0:
    fig2, ax2 = plt.subplots(1, 2, figsize=(18,6))
    
    ax2[0].plot(ts, kinetic_energies, label='Kinetic energy')
    ax2[0].plot(ts, dissipated_energies, label='Dissipated energy')
    ax2[0].plot(ts, elastic_energies, label='Strain energy')
    ax2[0].plot(ts, kinetic_energies + elastic_energies, label='Total energy')
    ax2[0].set_xlabel('Time [s]')
    ax2[0].set_ylabel('Energy [mJ]')
    ax2[0].legend()
    
    ax2[1].plot (ts, reaction_force_right_s)
    ax2[1].set_xlabel('Time [s]')
    ax2[1].set_ylabel('Reaction force [N]')
    plt.show()

# %% [markdown]
# The engines launched at the beginning keep running once the notebook reaches its end. Stopping the cluster releases them.

# %%
rc.cluster.stop_cluster_sync()
