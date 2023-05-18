import numpy as np
import dolfinx.plot as plot
import pyvista


def plot_damage_state(state, load=None):
    """
    Plot the displacement and damage field with pyvista
    """
    u = state["u"]
    alpha = state["alpha"]

    mesh = u.function_space.mesh

    plotter = pyvista.Plotter(
        title="Damage state", window_size=[800, 300], shape=(1, 2)
    )

    topology, cell_types, geometry = plot.create_vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    plotter.subplot(0, 0)
    if load is not None:
        plotter.add_text(f"Displacement - load {load:3.3f}", font_size=11)
    else:
        plotter.add_text("Displacement", font_size=11)
    vals_2D = u.compute_point_values().real
    vals = np.zeros((vals_2D.shape[0], 3))
    vals[:, :2] = vals_2D
    grid["u"] = vals
    warped = grid.warp_by_vector("u", factor=0.1)
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


def warp_plot_2d(u,cell_field=None,field_name="Field",factor=1.,backend="none",**kwargs):
    #"ipyvtklink", "panel", "ipygany", "static", "pythreejs", "none"
    msh = u.function_space.mesh
    
    # Create plotter and pyvista grid
    plotter = pyvista.Plotter()

    topology, cell_types, geometry = plot.create_vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
    values[:, :len(u)] = u.x.array.real.reshape((geometry.shape[0], len(u)))
    grid["u"] = values
    warped_grid = grid.warp_by_vector("u", factor=factor)
    if cell_field is not None:
        warped_grid.cell_data[field_name] = cell_field.vector.array
        warped_grid.set_active_scalars(field_name)
    plotter.add_mesh(warped_grid,**kwargs)
    #plotter.show_axes()
    plotter.camera_position = 'xy'
    
    return plotter

