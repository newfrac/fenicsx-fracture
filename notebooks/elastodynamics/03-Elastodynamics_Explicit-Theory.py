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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Optimized explicit solver for the elastodynamics problem
#
# __Authors__: A. Chao Correas (arturo.chaocorreas@polito.it) and C. Maurini (corrado.maurini@sorbonne-universite.fr)

# %% [markdown]
# ## Explicit time integration scheme

# %% [markdown]
# In an explicit time integrator, the independent unknown of the system at an instant $t_{i+1}$ (typically $\ddot{\underline{u}}$) is entirely determined through the propagation of the known system's state at a previous instant $t_i$. According to this, Newmark's $\beta$-Method becomes an explicit time integrator for the elastodynamics problem whenever the dissipative power $\mathcal{Q}$ and $\beta$ are both null. The latter choice is often accompanied by setting $\gamma = 1/2$, leading to the so-called "Central difference method". Subsequently, Newmark's approximation of the displacement and velocity propagations to the instant $t_{i+1}$ particularizes as follows:
#
# $$ 
# \underline{u}_{t_{i+1}} (\underline{x})
# \approx
# \underline{u}_{i+1} \left(\underline{x};\, \underline{u}_{i}, \dot{\underline{u}}_{i}, \ddot{\underline{u}}_{i} \right)
# =
# \underline{u}_{i} (\underline{x})
# +
# \Delta t_{i}\,\dot{\underline{u}}_i (\underline{x})
# +
# \frac{{\Delta t_{i}}^2}{2}\,\ddot{\underline{u}}_{i} (\underline{x})
# $$
#
# $$ 
# \dot{\underline{u}}_{t_{i+1}} (\underline{x})
# \approx
# \dot{\underline{u}}_{i+1} \left(\underline{x};\, \dot{\underline{u}}_{i}, \ddot{\underline{u}}_{i} \right)
# = 
# \dot{\underline{u}}_{i} (\underline{x})
# +
# \frac{\Delta t_{i}}{2} \left(\ddot{\underline{u}}_{i} (\underline{x})+\ddot{\underline{u}}_{i+1} (\underline{x}) \right) 
# $$
#
# Clearly, these relations imply that :
#
# * $\ddot{\underline{u}}_{i+1}$ will not play any role in the determination of $\underline{u}_{i+1}$
#
# * The velocity $\dot{\underline{u}}_{i+1}$ is propagated from $\dot{\underline{u}}_{i}$ using the mean value of $\ddot{\underline{u}}_{i}$ and $\ddot{\underline{u}}_{i+1}$
#
# As a consequence, the displacement field $\underline{u}_{i+1}$ can be entirely determined on the basis of the last known system's state, i.e. $\underline{u}_{i}$, $\dot{\underline{u}}_i$ and $\ddot{\underline{u}}_{i}$. Therefore, considering the conservative case $\left( \mathcal{Q}=0 \right)$, the particularization of the weak form to the instant $t_{i+1}$ after substituting $\ddot{\underline{u}}_{t}$ and $\underline{u}_{t}$ by their Newmark approximations gives the following expression:
#
# $$
# \int_{\Omega}\rho\,\ddot{\underline{u}}_{i+1}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x})\,\mathrm{d}x
# = 
# -\int_{\Omega} \underline{\underline{\sigma}}\left(\underline{\underline{\varepsilon}}(\underline{u}_{i+1}, \underline{x}),\underline{x}\right) : \underline{\underline{\varepsilon}}(\hat {\underline{u}}, \underline{x}) \,\mathrm{d}x 
# + 
# \int_{\Omega} \underline{b}_{i+1}(\underline{x}) \cdot \underline{\hat u}(\underline{x}) \,\mathrm{d}x + 
# \int_{\partial_f\Omega} {\underline{f}}_{i+1}(\underline{x}) \cdot \underline{\hat u}(\underline{x}) \,\mathrm{d}x
# \quad \forall \underline{\hat u}(\underline{x}) \in \mathrm{V}_{0}
# $$
#
# in which the sole unknown is $\ddot{\underline{u}}_{i+1}$, and $\underline{u}_{i}$, $\dot{\underline{u}}_{i}$ and $\ddot{\underline{u}}_{i}$ are known parameters. Consequently, the acceleration $\ddot{\underline{u}}_{i+1}$ is obtained from the resolution of the equation above, which once determined, allows for also propagating the velocity and determine $\dot{\underline{u}}_{i+1}$, thus rendering the state at $t_{i+1}$ fully determined.

# %% [markdown]
# ### Stability of the solution

# %% [markdown]
# The explicitness of the expression used to determine the independent unknown of the elastodynamics problem, i.e. $\ddot{\underline{u}}_{i+1}$, leads to a much simpler and computationally cheaper approach. Nonetheless, this comes at the cost of a resolution scheme that is not unconditionally stable with respect to the magnitude of $\Delta t$. In fact, for it to be stable, the Courant–Friedrichs–Lewy (CFL) condition must be satisfied, which mathematically writes in the absence of damping as:
#
# $$
# \Delta t 
# <
# \frac{2}{\omega_{\mathrm{max}}}
# $$
#
# where $\omega_{\mathrm{max}}$ represents the highest natural frequency in the model. Nonetheless, since the latter is not known _a priori_ and its computation might result expensive, it is common to substitute the CFL condition above by the estimate:
#
# $$
# \Delta t 
# <
# \frac{h_{min}}{\eta c}
# $$
#
#
# where $h_{min}$ represents the minimum distance between nodes within the discretized domain, $\eta$ is a safety factor larger than 1 and $c$ stands for the speed of sound in the respective continuum, which in turn is proportional to $\sqrt{E/\rho}$. Conceptually, this condition can be undertood as the requirement for the resolution of the temporal discretization to be small enough so it is ensured to be able to capture the minimum time required for the information to travel between any two nodes within the domain. 

# %% [markdown]
# ### Viscous dissipation in explicit solvers

# %% [markdown]
# Strictly speaking, the conventional explicit version of the Newmark $\beta$-method does not allow for incoporating a dissipative term that depends on $\dot{\underline{u}}$. This arises from the dependence of the latter's approximation at the instant $t_{i+1}$, i.e. $\dot{\underline{u}}_{i+1}$, with $\ddot{\underline{u}}_{i+1}$, which would imply that the time integrator is no longer explicit.
#
# Nonethless, the smallness required for $\Delta t$ in order to achieve stability in the solution also makes it possible to use a workaround and implement velocity-based viscous dissipation in implicit solvers. Indeed, given the small $\Delta t$, and considering that the velocity in the domain is a continuous function in time, one can assume that:
#
# $$
# \dot{\underline{u}}_{i+1}
# \approx
# \dot{\underline{u}}_{i}
# $$
#
# Obviously, the validity of this assumption is reduced upon scenarios in which abrupt changes in velocity occur.
#
# Then, using the approximation above, the viscous dissipative term at the instant $t_{i+1}$ can be determined by using the known velocity $\dot{\underline{u}}_{i}$ instead. This leaves the governing variational equation as:
#
# $$
# \int_{\Omega}\rho\,\ddot{\underline{u}}_{i+1} (\underline{x}) \cdot \hat{\underline{u}} (\underline{x})\,\mathrm{d}x
# =
# -
# \int_{\Omega} c_{1} \,\dot{\underline{u}}_{i} (\underline{x}) \cdot \hat{\underline{u}} (\underline{x}) \,\mathrm{d}x
# - 
# \int_{\Omega} \underline{\underline{\sigma}} \left( \underline{\underline{\varepsilon}}(\underline{u}_{i+1}, \underline{x}), \underline{\underline{\varepsilon}}(\dot{\underline{u}}_{i}, \underline{x}) \right) : \underline{\underline{\varepsilon}}(\underline{\hat {u}}, \underline{x}) \,\mathrm{d}x
# + 
# \int_{\Omega} \underline{b}_{i+1}(\underline{x}) \cdot \underline{\hat u}(\underline{x}) \,\mathrm{d}x 
# + 
# \int_{\partial_f \Omega} {\underline{f}}_{i+1}(\underline{x}) \cdot \underline{\hat u}(\underline{x}) \,\mathrm{d}x 
# \quad 
# \forall \underline{\hat u} \in \mathrm{V}_{0}
# $$

# %% [markdown]
# ### Lumping of the mass matrix

# %% [markdown]
# Upon discretization of the continuous equation above using a Finite Element approach, a system of finite number of equations is obtained. This can generally be expressed in matricial form as follows:
#
# $$
# \underline{\underline{M}}_{i+1} \underline{a}_{i+1} 
# = 
# -
# \underline{\underline{C}}_{i+1} \underline{v}_{i} 
# - 
# \underline{\underline{K}}_{i+1} \underline{u}_{i+1}
# +
# \underline{P}_{i+1}
# $$
#
# where $\underline{\underline{M}}$, $\underline{\underline{C}}$ and $\underline{\underline{K}}$ are the mass, damping and stiffness matrices, respectively, $\underline{P}$ is the vector of nodal external forces, and $\underline{a}$, $\underline{v}$ and $\underline{u}$ correspondingly stand for the nodal acceleration, velocity and displacement vectors. Note that generally the matrices $\underline{\underline{M}}$, $\underline{\underline{C}}$ and $\underline{\underline{K}}$ and vector $\underline{P}$ can evolve with time.
#
# Consequently, after propagating $\underline{u}$ to the instant $t_{i+1}$, the nodal acceleration at the instant $\underline{a}_{i+1}$ can be obtained through the resolution of the following linear system of equations:
#
# $$
# \underline{a}_{i+1} 
# = 
# \underline{\underline{M}}_{i+1}^{-1} \left(
# -
# \underline{\underline{C}}_{i+1} \underline{v}_{i} 
# - 
# \underline{\underline{K}}_{i+1} \underline{u}_{i+1}
# +
# \underline{P}_{i+1} \right)
# =
# \underline{\underline{M}}_{i+1}^{-1} \underline{F}_{i+1}
# $$
#
# where $\underline{F}$ is a vector that aggregates all the (known) nodal forces, both internal and external. 
#
# Now, should conventional Finite Element procedures be applied to determine $\underline{\underline{M}}$ out of the interpolation functions and the density $\rho$, the so-called __Consistent Mass matrix__ is obtained. Particularly, this method generates a full matrix in which crossed terms are generally not zero, thus noticeably increasing the computational cost upon fine meshes and/or large simulation times.
#
# Instead, most implementations of explicit time integrators for elastodynamic problems make use of the so called __Lumped Mass matrix__, a diagonalized counterpart of the latter, which allows to keep the computational cost reasonable. Different strategies to diagonalize this exist, though the one here implemented is the most straightforward and widely spread. 
#
# In particular, this approach consists in determining the Lumped Mass matrix $\underline{\underline{M_{l}}}$ as that one diagonal generating the same inertial force than $\underline{\underline{M}}$ upon unit nodal accelerations (represented these by $\underline{1}$). Mathematically, this reads as:
#
# $$
# \underline{\underline{M_{l}}}
# =
# \left(\underline{\underline{M}} \underline{1}\right) \underline{\underline{I}}
# $$
#
# where $\underline{\underline{I}}$ is the identity matrix. Indeed, this operation is equivalent to determining each diagonal component of $\underline{\underline{M_{l}}}$ as the sum of all the components in the respective row.
