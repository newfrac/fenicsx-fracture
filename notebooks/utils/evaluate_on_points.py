import numpy as np
from dolfinx import geometry, plot

def evaluate_on_points(function,points):
    domain = function.function_space.mesh
    comm = domain.comm
    # comm
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc)
    #print(points_on_proc)
    if len(points_on_proc) > 0:
        values_on_proc = function.eval(points_on_proc, cells)
    else:
        values_on_proc = None
    return [points_on_proc,values_on_proc]

if __name__ == "__main__":
    
    from dolfinx import fem, mesh
    from mpi4py import MPI


    # Create a mesh and a first function

    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle
    )
    V_u = fem.VectorFunctionSpace(domain, ("CG", 1))
    V_f = fem.FunctionSpace(domain, ("CG", 1))
    u = fem.Function(V_u)
    u.interpolate(lambda x: [1 + x[0] ** 2, 2 * x[1] ** 2])
    
    x_s = np.linspace(0,1,10)
    y_s = 0.2 * np.ones_like(x_s)
    z_s = 0.0 * np.ones_like(x_s)
    points = np.array([x_s,y_s,z_s])
    values = evaluate_on_points(u,points)
    print(values)

