import pyvista
import dolfinx.plot

def mesh_plotter(mesh):
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    # Set some global options for all plots
    transparent = False
    figsize = 100
    pyvista.rcParams["background"] = [0.5, 0.5, 0.5]
    topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()