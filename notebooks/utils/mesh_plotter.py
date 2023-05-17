import pyvista
import dolfinx.plot

def mesh_plotter(mesh):
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    if pyvista.OFF_SCREEN:
        from pyvista.utilities.xvfb import start_xvfb
        start_xvfb(wait=0.5)
    # Set some global options for all plots
    transparent = False
    figsize = 100
    pyvista.rcParams["background"] = [0.5, 0.5, 0.5]
    topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    from pyvista.utilities.xvfb import start_xvfb
    start_xvfb(wait=0.5)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()