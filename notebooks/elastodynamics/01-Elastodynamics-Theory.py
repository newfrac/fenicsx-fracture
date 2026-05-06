# ---
# jupyter:
#   jupytext:
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
# # The elastodynamics problem
#
# __Authors__: A. Chao Correas (arturo.chaocorreas@polito.it) and C. Maurini (corrado.maurini@sorbonne-universite.fr)

# %% [markdown]
# ## Weak form of the elastodynamics problem

# %% [markdown]
# _Insert image_

# %% [markdown]
# Let us assume an elastodynamics problem as the one above in which a domain $\Omega$ is subjected to:
#
# 1. Time-dependent displacement boundary conditions $\underline{U}_{t}$ on the Dirichlet boundary $\partial_{u}{\Omega}$.
# 2. Time-dependent surface loads  $\underline{f}_{t}$ (Neumann BCs) along the Neumann boundary $\partial_{f}{\Omega}= \partial{\Omega} \setminus \partial_{u}{\Omega}$.
# 3. Time-dependent distributed body forces $\underline{b}_{t}$ in $\Omega$. 
#
# And where the hypotheses of small deformations and small displacements hold. In such a problem, $\underline{u}_{t}$, $\dot{\underline{u}}_{t}$ and $\ddot{\underline{u}}_{t}$ denote the displacement, velocity and acceleration vector fields at the instant $t$, respectively.
#
# To fulfill the Dirichlet boundary conditions, the displacement field $\underline{u}_{t}$ must be admissible and belong to the affine space $\mathrm{V}(t)$, which is defined as:
#
# $$ 
# \mathrm{V}(t) 
# = 
# \left\{ \underline{u}_{t}(\underline{x}) = \underline{U}_{t}(\underline{x}) \quad \forall \underline{x} \in \partial_{u}\Omega \right\} 
# $$
#
# At the same time, the imposition of $\underline{u}_{t}$ along $\partial_{u}\Omega$ also constrains the velocity and acceleration admissible fields to belong to the vector spaces $\dot{\mathrm{V}}(t)$ and $\ddot{\mathrm{V}}(t) $, respectively. In turn, these get defined mathematically as:
#
# $$ 
# \dot{\mathrm{V}}(t) 
# = 
# \left\{ \underline{v}_{t}(\underline{x}) = \dot{\underline{U}_{t}}(\underline{x}) \quad \forall \underline{x} \in \partial_{u}\Omega \right\} 
# $$
#
# $$ 
# \ddot{\mathrm{V}}(t) 
# = 
# \left\{ \underline{a}_{t}(\underline{x}) = \ddot{\underline{U}_{t}}(\underline{x}) \quad \forall \underline{x} \in \partial_{u}\Omega \right\} 
# $$
#
# All three affine spaces share the same homogenized vector space counterparts:
#
# $$ 
# \mathrm{V}_{0} = \dot{\mathrm{V}}_{0} = \ddot{\mathrm{V}}_{0}
# $$

# %% [markdown]
# From these kinematic considerations, we can now state the elastodynamic problem for an elastic body submitted to dynamic external loads. In this scenario, the total potential energy of the system $\mathcal{P}$ writes as:
#
# $$ 
# \mathcal{P}\left(\underline{u}_{t}, t\right)  
# =
# \mathcal{E}\left(\underline{u}_{t}, t\right) 
# - 
# \mathcal{W}_{\mathrm{ext}}\left(\underline{u}_{t}, t\right)
# $$
#
# where 
#
# $$
# \mathcal{E}\left(\underline{u}_{t},t\right) 
# = 
# \int_{\Omega} \psi \left(\underline{\underline{\varepsilon}} (\underline{u}_{t}), \underline{x}\right) \,\mathrm{d}{x},
# \qquad
# \mathcal{W}_{\mathrm{ext}}\left(\underline{u}_{t},t\right) 
# = 
# \int_{\Omega} \underline{b}_{t}(\underline{x}) \cdot \underline{u}_{t}(\underline{x}) \,\mathrm{d}{x} 
# + 
# \int_{\partial_{f}\Omega} \underline{f}_{t}(\underline{x}) \cdot \underline{u}_{t}(\underline{x}) \,\mathrm{d}{s}
# $$
#
# are the strain energy and the work of the conservative external loads, respectively, with $\psi$ being the strain energy density. 
#
# On the other hand, the kinetic energy of the system is defined as:
#
# $$
# \mathcal{K}\left(\dot{\underline{u}}_{t},t\right) 
# = 
# \frac{1}{2} \int_{\Omega} \rho \dot{\underline{u}}_{t}(\underline{x}) \cdot \dot{\underline{u}}_{t}(\underline{x}) \,\mathrm{d}{x}
# $$
#
# where $\rho$ represents the mass density of the material filling the domain.

# %% [markdown]
# So far, only conservative energies have been introduced, which would mean that the structure's motion would keep on forever. Nonetheless, in reality the systems dissipate energy when deformed through several mechanisms (heat, noise...). There exists different strategies to account for such energy losses, the one here implemented being to include a non-conservative Rayleigh-like dissipative term, whose power is defined as follows:
#
# $$
# \mathcal{Q} \left( \dot{\underline{u}}_{t}, t \right) 
# = 
# \frac{1}{2} \int_{\Omega} c_{1} \dot{\underline{u}}_{t}(\underline{x}) \cdot \dot{\underline{u}}_{t}(\underline{x})\,\mathrm{d}{x} 
# +
# \frac{1}{2} \int_{\Omega} c_{2} \left( \underline{\underline{\varepsilon}} (\dot{\underline{u}}_{t}, \underline{x}) : \underline{\underline{\varepsilon}} (\dot{\underline{u}}_{t},\underline{x})\right)\,\mathrm{d}{x}
# $$
#
# where $c_{1}$ and $c_{2}$ are two "viscosity" moduli, whose units are $\mathrm{N} \mathrm{s} \mathrm{m}^{-4}$ and $\mathrm{N} \mathrm{s} \mathrm{m}^{-2}$, respectively, and $\underline{\underline{\varepsilon}} (\underline{\bullet}, \underline{x})$ represents the symetric part of the gradient of $(\underline{\bullet})$ at $\underline{x}$.  With this definition, the energy losses in the continuum are considered to arise from:
#
# 1. The velocity field, which can lead to dissipation through external mechanisms such as the friction with external fluids.
# 2. The gradient of the velocity field, which accounts for dissipation due to the internal material friction when deforming. 

# %% [markdown]
# Then, using the D'Alembert principle, one can determine the following weak formulation of the elastodynamics: 
#
# $$ 
# \frac{\mathrm{d}}{\mathrm{d}t}\left(\mathrm{D}_{\underline{\dot{u}}}\mathcal{K}\left( \dot{\underline{u}}_{t},t\right)\left[\hat{\underline{u}} \right]\right) 
# +
# \mathrm{D}_{\underline{u}} \mathcal{P}\left(\underline{u}_{t}, t\right)\left[\hat{\underline{u}} \right] 
# = 
# -\mathrm{D}_{\underline{\dot{u}}}\mathcal{Q} \left( \dot{\underline{u}}_{t}, t \right) \left[\hat{\underline{u}} \right]
# \quad \forall \hat{\underline{u}}(\underline{x}) \in \mathrm{V}_{0}
# $$ 
#
# where  $\mathrm{D}_{\underline{u}} \mathcal{P}(\underline u)[\hat{\underline{u}}]$ denotes the directional derivative of the functional $\mathcal{P}(\underline u)$ with respect to $\underline{u}$ in the direction of $\hat{\underline{u}}$. In turn, the latter represents an admissible variation of $\underline{u}_{t}$ so that it belongs to the homogeneous vector space $\mathrm{V}_{0}$ above introduced.
#
# Expanding the different terms of the weak form, one obtains:
#
# $$
# \int_{\Omega}\rho\,\ddot{\underline{u}}_{t}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# +
# \int_{\Omega} c_{1} \,\dot{\underline{u}}_{t}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# + 
# \int_{\Omega} \underline{\underline{\sigma}} \left( \underline{\underline{\varepsilon}} (\underline{u}_{t}, \underline{x}), \underline{\underline{\varepsilon}} (\dot{\underline{u}}_{t}, \underline{x}), \underline{x} \right) : \underline{\underline{\varepsilon}} (\hat{\underline{u}}_{t}, \underline{x}) \,\mathrm{d}x
# =
# \int_{\Omega} \underline{b}_{t}(\underline{x})\cdot \hat{\underline{u}}(\underline{x})\,\mathrm{d}x
# + 
# \int_{\partial_\Sigma\Omega} {\underline{f}}_{t}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# \quad \forall \hat{\underline{u}} (\underline{x}) \in \mathrm{V}_{0}
# $$
#
# where $\underline{\underline{\sigma}}$ stands for the stress tensor field, defined as follows:
#
# $$
# \underline{\underline{\sigma}}\left(\underline{\underline{\varepsilon}} (\underline{u}_{t}, \underline{x}), \underline{\underline{\varepsilon}} (\dot{\underline{u}}_{t}, \underline{x}), \underline{x} \right)
# = 
# c_{2} \, \underline{\underline{\varepsilon}} (\dot{\underline{u}}_{t}, \underline{x}) 
# + 
# \frac{\partial \psi\left(\underline{\underline{\varepsilon}} (\underline{u}_{t}, \underline{x}), \underline{x}\right)}{\partial \underline{\underline{\varepsilon}}} 
# $$
#
# The presence of the acceleration $\ddot{\underline{u}}_{t}$ and velocity $\dot{\underline{u}}_{t}$ in the inertial and dissipative terms, respectively, makes the system history-dependent. As a consequence, and differently from the quasi-static case, the problem cannot be independently solved at each instant. To circumvent this difficulty, a time integration scheme is required towards relating the velocity and displacement fields to that of the acceleration, so that the governing equation can be entirely written in terms of the latter. In what follows, the well-established Newmark $\beta$-method for time integration will be presented and used.

# %% [markdown]
# # Newmark $\beta$-method for time integration of dynamic problems
# Reference: Newmark N. M., 1959. A Method of Computation for Structural Dynamics. _J Eng Mech Div_. 85:67–94. https://doi.org/10.1061/JMCEA3.0000098

# %% [markdown]
# Let us assume a time discretization of the temporal domain of the problem so that $t_i$ represents the latest instant at which the system's state is known, and $t_{i+1}$ is the earliest instant at which the solution is not known. On this basis, the one can define the time increment $\Delta t_{i}$ as follows:
#
# $$ \Delta t_{i} = t_{i+1}-t_{i}$$
#
# Then, using the extended mean value theorem, the displacement $\underline{u}_{i+1}$ and velocity $\dot{\underline{u}}_{i+1}$ vector fields at $t_{i+1}$ can be approximated as:
#
# $$ 
# \underline{u}_{t_{i+1}}(\underline{x})
# \approx
# \underline{u}_{i+1}\left(\ddot{\underline{u}}_{i+1}, \underline{x};\, \beta, \underline{u}_{i}, \dot{\underline{u}}_{i}, \ddot{\underline{u}}_{i} \right) 
# = 
# \underline{u}_{i}(\underline{x})
# +
# \Delta t_{i}\dot{\underline{u}}_{i}(\underline{x})
# +
# \frac{{\Delta t_i}^2}{2}\left((1-2\beta)\,\ddot{\underline{u}}_{i}(\underline{x})
# +
# 2\beta\, \ddot{\underline{u}}_{i+1}(\underline{x}) \right) 
# \quad 0\le 2\beta\le1 
# $$
#
# $$ 
# \dot{\underline{u}}_{t_{i+1}}(\underline{x})
# \approx
# \dot{\underline{u}}_{i+1} \left(\ddot{\underline{u}}_{i+1}, \underline{x};\, \gamma, \dot{\underline{u}}_{i}, \ddot{\underline{u}}_{i} \right)
# =
# \dot{\underline{u}}_{i}(\underline{x})
# +
# \Delta t_{i} \left((1-\gamma)\,\ddot{\underline{u}}_{i}(\underline{x})
# +
# \gamma\,\ddot{\underline{u}}_{i+1}(\underline{x}) \right) 
# \quad 0\le\gamma\le1 
# $$
#
# Where the notation $(\bullet) \left(\ddot{\underline{u}}_{i+1};\, \{\beta, \gamma\}, \underline{u}_{i}, \dot{\underline{u}}_{i}, \ddot{\underline{u}}_{i}\right)$ indicates that the corresponding magnitudes are a function of the unknown variable $\ddot{\underline{u}}_{i+1}$ and admits the known variables $\{\beta, \gamma\}, \underline{u}_{i}, \dot{\underline{u}}_{i}, \ddot{\underline{u}}_{i}$ as parameters.
# Furthermore, $\beta$ and $\gamma$ are scalar parameters that allow to modify the characteristics of the time stepping scheme by weighting how much the acceleration at the instant $t_{i+1}$, i.e. $\ddot{\underline{u}}_{i+1}$, affects $\underline{u}_{i+1}$ and $\dot{\underline{u}}_{i+1}$. Clearly, in this manner $\ddot{\underline{u}}_{i+1}$ becomes the only independent unkown of the elastodynamic problem. Now, particularizing the weak form of the elastodynamic problem to the instant $t_{i+1}$ and substituting $\underline{u}_{t_{i+1}}$ and $\dot{\underline{u}}_{t_{i+1}}$ by their approximations above, one gets:
#
# $$
# \int_{\Omega}\rho\,\ddot{\underline{u}}_{i+1}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x})\,\mathrm{d}x
# +
# \int_{\Omega} c_{1} \,\dot{\underline{u}}_{i+1}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# + 
# \int_{\Omega} \underline{\underline{\sigma}} \left( \underline{\underline{\varepsilon}}(\underline{u}_{i+1}, \underline{x}), \underline{\underline{\varepsilon}}(\dot{\underline{u}}_{i+1}, \underline{x}), \underline{x} \right) \cdot \underline{\underline{\varepsilon}}(\hat{\underline{u}}, \underline{x}) \,\mathrm{d}x
# = 
# \int_{\Omega} \underline{b}_{i+1}(\underline{x})\cdot \underline{\hat u}(\underline{x}) \,\mathrm{d}x + 
# \int_{\partial_\Sigma\Omega} {\underline{f}}_{i+1}(\underline{x})\cdot \underline{\hat u}(\underline{x}) \,\mathrm{d}x 
# \quad \forall \underline{\hat u}(\underline{x}) \in \mathrm{V}_{0}
# $$
#
# Where the dependences of $\underline{u}_{i+1}$ and $\dot{\underline{u}}_{i+1}$ have been ommitted for the sake of clarity. Consequently, the Newmark $\beta$-method generally constitutes an implicit time integration scheme for it determines the independent unknown of the problem $\ddot{\underline{u}}_{i+1}$ as a function of the system's state at both instants $t_{i}$ and $t_{i+1}$. Nonetheless, in the absence of disspitation $\mathcal {Q} = 0$ (although this limitation can be circumvented) and upon specific choice of the parameters $\beta$ and $\gamma$, this time integrator becomes explicit and it implementation can be highly optimized (See notebook `Elastodynamics: Explicit solver`).  
#
