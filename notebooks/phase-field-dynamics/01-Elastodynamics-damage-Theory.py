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
# # The Elastodynamics and Damage coupled problem
#
# __Author__: A. Chao Correas (arturo.chaocorreas@polito.it) and C. Maurini (corrado.maurini@sorbonne-universite.fr)

# %% [markdown]
# ## Derivation of the weak form of the elastodynamics and damage problem

# %% [markdown]
# Let us assume an elastodynamics problem as the one considered in `01-Elastodynamics-Theory`, but in which there also exists a continuous damage field $\alpha$ defined all along $\Omega$ that regularizes the discontinuous nature of fracture. In this sense, $\alpha = 0$ represents pristine material conditions while $\alpha = 1$ implies fully broken states. Furthermore, this field is subjected to its own Dirichlet boundary conditions along $\partial_{\alpha}\Omega$ as well as to an irreversibility condition, so that the admissible space for $\alpha$ is:
#
# $$
# \mathrm{B}(t) 
# =
# \left\{\alpha_{t}(\underline{x}) = A_{t}(\underline{x})  \,\, \forall \,\, \underline {x} \in \partial_{\alpha}\Omega \quad \& \quad \alpha_{t} (\underline{x}) \ge \alpha_{t'}(\underline{x}) \,\, \forall \,\, \underline {x} \times t' \in \Omega\times [0,t) \right\}
# $$
#
# Under those conditions, we can write the total energy of the system $\mathcal{E}_{TOT}$ as:
#
# $$ 
# \mathcal{E}_{TOT}\left(\underline{u}_{t}, \dot{\underline{u}}_{t}, \alpha_{t}, t\right)  
# = 
# \mathcal{K}\left(\dot{\underline{u}}_{t}, t\right) 
# + 
# \mathcal{E}\left(\underline{u}_{t}, \alpha_{t}, t\right) 
# + 
# \mathcal{F}\left(\alpha_{t}, t\right) 
# -
# \mathcal{W}_\mathrm{ext} \left(\underline{u}_{t}, t\right)
# $$ 
#
# where $\mathcal{K}$, $\mathcal{E}$ and $\mathcal{F}$ represent the kinetic, strain and fracture energy of the system, respectively, and $\mathcal{W}_\mathrm{ext}$ is the work done by the external forces. The mathematical definitions of each of these components is:
#
# $$
# \mathcal{K}\left(\dot{\underline{u}}_{t}, t\right)
# =
# \frac{1}{2}\int_{\Omega} \rho\dot{\underline{u}}_{t}(\underline{x}) \cdot \dot{\underline{u}}_{t}(\underline{x}) \, \mathrm{d}x
# $$
#
# $$
# \mathcal{E} \left( \underline{u}_{t}, \alpha_{t}, t\right)
# =
# \int_{\Omega} \psi \left( \underline{\underline{\varepsilon}} (\underline{u}, \underline{x}), \alpha_{t}(\underline{x})\right) \, \mathrm{d}x
# $$
#
# $$
# \mathcal{F} \left( \alpha_{t}, t \right)
# =
# \frac{G_c}{c_w} \int_{\Omega} \frac{w\left(\alpha_{t}(\underline{x})\right)}{\ell}
# +
# \ell \left(\nabla\alpha_{t}(\underline{x}) \cdot \nabla\alpha_{t}(\underline{x})\right) \, \mathrm{d}x
# $$
#
# $$
# \mathcal{W}_{\mathrm{ext}}\left(\underline{u}_{t}, t\right) 
# = 
# \int_{\Omega} \underline{b}_{t}(\underline{x}) \cdot \underline{u}_{t} \,\mathrm{d}{x} 
# + 
# \int_{\partial_{f}\Omega} \underline{f}_{t}(\underline{x}) \cdot \underline{u}_{t} \,\mathrm{d}{s}
# $$
#
# with $\psi \left( \underline{\underline{\varepsilon}} (\underline{u}, \underline{x}), \alpha_{t}(\underline{x})\right)$ representing the strain energy density modulated by the damage field $\alpha$, $G_{c}$ is the fracture energy, and $c_{w}$ is a scaling factor defined as:
#
# $$
# c_{w} 
# = 
# 4\int_{0}^{1} w \left(\beta\right) \, \mathrm{d}\beta
# $$
#
# Furthermore, the potential energy of the system $\mathcal{P}$ gets defined as:
#
# $$
# \mathcal{P} \left( \underline{u}_{t}, \alpha_{t}, t \right)
# =
# \mathcal{E} \left( \underline{u}_{t}, \alpha_{t}, t \right)
# +
# \mathcal{F} \left( \alpha_{t}, t \right)
# -
# \mathcal{W}_{\mathrm{ext}} \left(\underline{u}_{t}, t \right) 
# $$
#
# The viscous dissipation is accounted for using a non-conservative Rayleigh-like dissipative term, whose power is defined as follows:
#
# $$
# \mathcal{Q} \left( \dot{\underline{u}}_{t}, t \right) 
# = 
# \frac{1}{2} \int_{\Omega} c_{1} \dot{\underline{u}}_{t}(\underline{x}) \cdot \dot{\underline{u}}_{t}(\underline{x})\,\mathrm{d}{x} 
# +
# \frac{1}{2} \int_{\Omega} c_{2} \left( \underline{\underline{\varepsilon}} (\dot{\underline{u}}_{t}, \underline{x}) : \underline{\underline{\varepsilon}} (\dot{\underline{u}}_{t},\underline{x})\right)\,\mathrm{d}{x}
# $$
#
# Based on the D'Alambert principle and considering the irreversibility condition in $\alpha$, one can determine the governing variational form as: 
#
# $$ 
# \frac{\mathrm{d}}{\mathrm{d}t}\left(\mathrm{D}_{\underline{\dot{u}}} \mathcal{K}\left(\dot{\underline{u}}_{t}, t\right) 
# \left[\hat{\underline{u}} \right] \right)
# + 
# \mathrm{D}_{\underline{u}} \mathcal{P}\left(\underline{u}_{t}, \alpha_{t}, t\right)\left[\hat{\underline{u}} \right] 
# +
# \mathrm{D}_{\alpha} \mathcal{P}\left(\underline{u}_{t}, \alpha_{t}, t\right)\left[\hat{\alpha} \right]
# \ge 
# -\mathrm{D}_{\underline{\dot{u}}}\mathcal{Q} \left( \dot{\underline{u}}_{t}, t \right) \left[\hat{\underline{u}} \right] 
# \quad \forall \left\{\hat{\underline{u}}, \hat{\alpha} \right\} \in \left\{\mathrm{V}_0, \mathrm{B}_0\right\} 
# $$ 
#
# Due to the difficulties of solving the variational form above in a monolithic manner, a staggered approach is commonly utilized. This consists in separating the coupled problem into two related problems: 
#
# 1. The deformation problem with $\underline{u}$, $\dot{\underline{u}}$ and $\ddot{\underline{u}}$ as the unknowns and $\alpha$ as a parameter.
# 2. The damage problem with $\alpha$ as the unknown and $\underline{u}$, $\dot{\underline{u}}$ and $\ddot{\underline{u}}$ as parameters.
#
# In turn, this materializes in the expressions:
#
# $$ 
# \frac{\mathrm{d}}{\mathrm{d}t}\left(\mathrm{D}_{\underline{\dot{u}}} \mathcal{K}\left(\dot{\underline{u}}_{t}, t\right) 
# \left[\hat{\underline{u}} \right] \right)
# + 
# \mathrm{D}_{\underline{u}} \mathcal{P}\left(\underline{u}_{t}, t; \, \alpha\right)\left[\hat{\underline{u}} \right] 
# = 
# -\mathrm{D}_{\underline{\dot{u}}}\mathcal{Q} \left( \dot{\underline{u}}_{t}, t \right) \left[\hat{\underline{u}} \right] 
# \quad \forall \hat{\underline{u}} \in \mathrm{V}_0
# $$ 
#
# $$ 
# \mathrm{D}_{\alpha} \mathcal{P}\left(\alpha_{t}, t;\,\underline{u}\right)\left[\hat{\alpha} \right]
# \ge 
# 0 
# \quad \forall \hat{\alpha} \in \mathrm{B}_0
# $$ 
#
#
# Which upon substitution lead to:
#
# $$
# \int_{\Omega}\rho\,\ddot{\underline{u}}_{t}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# +
# \int_{\Omega} c_{1} \,\dot{\underline{u}}_{t}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# + 
# \int_{\Omega} \underline{\underline{\sigma}}\left(\underline{\underline{\varepsilon}} (\underline{u}_{t}, \underline{x}); \, \alpha(\underline{x}) \right) : \underline{\underline{\varepsilon}} \left( \hat{\underline{u}}, \underline{x} \right) \,\mathrm{d}x
# = 
# \int_{\Omega} \underline{b}_{t}(\underline{x})\cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# + 
# \int_{\partial_{f}\Omega} \underline{f}_{t}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x 
# \quad 
# \forall \hat{\underline{u}}(\underline{x}) \in \mathrm{V}_{0}
# $$
#
# $$
# \int_{\Omega} \frac{\partial \psi\left(\alpha_{t}(\underline{x});\, \underline{\underline{\varepsilon}} (\underline{u}, \underline{x})\right)}{\partial \alpha} \cdot \hat{\alpha}(\underline{x}) \,\mathrm{d}x 
# +
# \frac{G_c}{c_w}\left[ \int_{\Omega} \frac{\mathrm{d}}{\mathrm{d}\alpha} \left(\frac{w(\alpha_{t}(\underline{x}))}{\ell} \right) \cdot \hat{\alpha}(\underline{x}) \,\mathrm{d}x + \int_{\Omega} 2 \ell \left( \nabla \alpha_{t}(\underline{x}) \cdot \nabla \hat{\alpha}(\underline{x}) \right) \,\mathrm{d}x \right]  
# \ge 
# 0 
# \quad \forall \hat{\alpha}(\underline{x}) \in \mathrm{B}_{0} 
# $$
#
# Clearly, for the sake of correctness the parameters in each of the equations above should correspond to the best guess available for each fields in question. In this sense, the staggered approach is often coupled with an alternate minimization procedure at each time instant, which consists in iterating back and forth the resolution of the deformation and damage problems until certain convergence criteria is met.
#

# %% [markdown]
# # Newmark $\beta$-method for time integration of elastodynamic problems with damage
# Reference: Newmark N.M., 1959. A Method of Computation for Structural Dynamics. J Eng Mech Div. 85:67–94. https://doi.org/10.1061/JMCEA3.0000098

# %% [markdown]
# Introducing Newmark's $\beta$-Method (see `01-Elastodynamics-Theory` notebook) and particularizing the equations for the instant $t_{i+1}$, we can generally set up the alternate minimization scheme so that, for an interator index $j$ starting at $1$ and increasing by $1$ at each iteration until the convergence criterion is met, we determine $\ddot{\underline{u}}_j$, $\underline{u}_j$ and $\alpha_j$ through the following equations:
#
# $$
# \int_{\Omega}\rho\,\ddot{\underline{u}}_{j}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# +
# \int_{\Omega} c_{1} \,\dot{\underline{u}}_{j}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# + 
# \int_{\Omega} \underline{\underline{\sigma}}\left(\underline{\underline{\varepsilon}} (\underline{u}_{j}, \underline{x}); \, \alpha_{j-1}(\underline{x}) \right) : \underline{\underline{\varepsilon}} \left( \hat{\underline{u}}, \underline{x} \right) \,\mathrm{d}x 
# = 
# \int_{\Omega} \underline{b}_{i+1}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# + 
# \int_{\partial_{f}\Omega} {\underline{f}}_{i+1}(\underline{x}) \cdot \hat{\underline{u}}(\underline{x}) \,\mathrm{d}x
# \quad 
# \forall \hat{\underline{u}}(\underline{x}) \in \mathrm{V}_{0}
# $$
#
# $$
# \underline{u}_{j}(\underline{x})
# =
# \underline{u}_{i}(\underline{x}) 
# +
# \Delta t_{i} \dot{\underline{u}}_{i}(\underline{x})
# +
# \frac{{\Delta t_i}^2}{2}\left((1-2\beta)\,\ddot{\underline{u}}_{i}(\underline{x})+ 2\beta\, \ddot{\underline{u}}_{j}(\underline{x}) \right) \quad 0\le 2\beta\le1 
# $$
#
# $$
# \int_{\Omega} \frac{\partial \psi\left(\alpha_{j}(\underline{x});\, \underline{\underline{\varepsilon}} (\underline{u}_{j}, \underline{x})\right)}{\partial \alpha} \cdot \hat{\alpha}(\underline{x}) \,\mathrm{d}x 
# +
# \frac{G_c}{c_w} \left[ \int_{\Omega} \frac{\mathrm{d}}{\mathrm{d}\alpha} \left(\frac{w\left(\alpha_{j}(\underline{x})\right)}{\ell} \right) \cdot \hat{\alpha}(\underline{x}) \,\mathrm{d}x + \int_{\Omega} 2 \ell \left( \nabla \alpha_{j}(\underline{x}) \cdot \nabla \hat{\alpha}(\underline{x}) \right) \,\mathrm{d}x \right]
# \ge
# 0 
# \quad \forall \hat{\alpha}(\underline{x}) \in \mathrm{B}_{0}
# $$
#
# where $\alpha_{j=0}(\underline{x}) = \alpha_{i}(\underline{x})$.
#
# Once the convergence criterion for the alternate minimization is met, we assign the latest guesses for $\ddot{\underline{u}}_j$, $\underline{u}_j$ and $\alpha_j$ as the corresponding values at $t_{i+1}$.

# %%
