import dolfinx
import matplotlib.pyplot as plt


# ................... Initialization of pyvista
def init_pyvista():
    import pyvista

    pyvista.start_xvfb(wait=0.5)


# ................. Representation of the mesh
def plot_mesh(mesh):
    import pyvista

    # Create plotter
    plotter = pyvista.Plotter(window_size=(400, 400))
    # we first add the grid
    grid = create_grid(mesh)
    plotter.add_mesh(grid, show_edges=True, style="wireframe", color="k")
    # Then we display the scene
    plotter.view_xy()
    plotter.set_background("lightgray")
    plotter.show()


def create_grid(mesh):
    import pyvista

    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    # Create pyvista grid
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    return pyvista.UnstructuredGrid(topology, cell_types, geometry)


# ................... Representation of the tensile stress along the expected crack path
def plot_stresses(L, S):
    plt.figure()
    plt.plot(L, S, "tab:blue", label="tensile stress")
    plt.xlabel("$s$ [nm]")
    plt.title("Tensile stress along the expected crack path")
    plt.show()
    plt.savefig("./output/stress_patil.png")


# ................... Representation of the energy release rate
def plot_ginc(L, Ginc):
    plt.figure()
    plt.plot(L, Ginc, "tab:orange", label="Energy release rate")
    plt.xlabel("$dl$ [nm]")
    plt.ylabel("$G_\mathrm{inc}$ [GPa nm]")
    plt.title("Energy release rate")
    plt.show()
    plt.savefig("./output/ginc_patil.png")


# .................. European way of representing the Coupled Criterion
def plot_cc_europe(L, P_ginc, P_strs):
    fig, ax = plt.subplots()
    ax.plot(L, P_strs, "tab:blue", label="Stress condition")
    ax.plot(L, P_ginc, "tab:orange", label="Energy condition")
    ax.set_xlabel("$dl$ [nm]")
    ax.set_ylabel("$P/P_\mathrm{min}$")
    ax.set_title("European representation of the Coupled Criterion")
    plt.show()
    plt.savefig("./output/cceur_patil.png")


def plot_cc_france(L, Ginc, S):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(L, S, "tab:blue", label="Stress condition")
    ax.plot(L, Ginc, "tab:orange", label="Energy condition")
    ax.set_xlabel("$dl$ [nm]")
    ax.set_title("French representation of the Coupled Criterion")
    plt.show()
    plt.savefig("./output/ccfr_patil.png")
