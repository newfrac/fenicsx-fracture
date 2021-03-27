
import numpy as np
import dolfinx.plot
import pyvista

def plot_damage_state(state,load=None):
    """
    Plot the displacement and damage field with pyvista
    """
    u = state["u"]    
    alpha = state["alpha"]
    
    mesh = u.function_space.mesh
    
    plotter = pyvista.Plotter(title="Daamge state",window_size=[800, 300],shape=(1, 2))
    
    topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    
    plotter.subplot(0, 0)
    if load is not None:
        plotter.add_text(f"Displacement - load {load:3.3f}", font_size=11)
    else:
        plotter.add_text("Displacement", font_size=11)
    vals_2D = u.compute_point_values().real 
    vals = np.zeros((vals_2D.shape[0], 3))
    vals[:,:2] = vals_2D
    grid["u"] = vals
    warped = grid.warp_by_vector("u", factor=.1)
    actor_1 = plotter.add_mesh(warped, show_edges=False)
    plotter.view_xy()
    
    plotter.subplot(0, 1)
    if load is not None:
        plotter.add_text(f"Damage - load {load:3.3f}", font_size=11)
    else:
        plotter.add_text("Damage", font_size=11)
    grid.point_arrays["alpha"] = alpha.compute_point_values().real
    grid.set_active_scalars("alpha")
    plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, clim=[0, 1])
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
       plotter.show()