import gmsh
import numpy as np
import ufl
#import dolfinx.plotting
from dolfinx import Function, FunctionSpace

from mpi4py import MPI
proc = MPI.COMM_WORLD.rank

from dolfinx.io import extract_gmsh_geometry,extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh
from dolfinx.cpp.io import perm_gmsh
from dolfinx.cpp.mesh import to_type
from dolfinx.mesh import create_mesh

def generate_mesh_with_crack(Lx=1,Ly=1,Lcrack=.3,lc=.015,refinement_ratio=10,dist_min=.05,dist_max=.2):
    # For further documentation see
    # - gmsh tutorials, e.g. see https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/python/t10.py 
    # - dolfinx-gmsh interface https://github.com/FEniCS/dolfinx/blob/master/python/demo/gmsh/demo_gmsh.py
    # 
    gmsh.initialize()
    gdim = 2
    
    if proc == 0:
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
        p1 = model.geo.addPoint(0.0, 0.0, 0, lc)
        p2 = model.geo.addPoint(Lcrack, 0.0, 0, lc)
        p3 = model.geo.addPoint(Lx, 0, 0, lc)
        p4 = model.geo.addPoint(Lx, Ly, 0, lc)
        p5 = model.geo.addPoint(0, Ly, 0, lc)
        l1 = model.geo.addLine(p1, p2)
        l2 = model.geo.addLine(p2, p3)
        l3 = model.geo.addLine(p3, p4)
        l4 = model.geo.addLine(p4, p5)
        l5 = model.geo.addLine(p5, p1)
        cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4, l5])
        surface_1 = model.geo.addPlaneSurface([cloop1])
        
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [p2])
        #model.mesh.field.setNumber(1, "NNodesByEdge", 100)
        #model.mesh.field.setNumbers(1, "EdgesList", [2])
        #
        # SizeMax -                     /------------------
        #                              /
        #                             /
        #                            /
        # SizeMin -o----------------/
        #          |                |    |
        #        Point         DistMin  DistMax
    
        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", lc / refinement_ratio)
        model.mesh.field.setNumber(2, "LcMax", lc)
        model.mesh.field.setNumber(2, "DistMin", dist_min)
        model.mesh.field.setNumber(2, "DistMax", dist_max)
        model.mesh.field.setAsBackgroundMesh(2)
    
    
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(2)]
        model.addPhysicalGroup(2, surface_entities, tag=5)
        model.setPhysicalName(2, 2, "Rectangle surface")
        model.mesh.generate(gdim)
        
    
        # Sort mesh nodes according to their index in gmsh
        geometry_data = extract_gmsh_geometry(model, model_name="Rectangle")[:,0:2]
        topology_data = extract_gmsh_topology_and_markers(model, "Rectangle")
        
        # Broadcast cell type data and geometric dimension
        gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)
        # Extract the cell type and number of nodes per cell and broadcast
        # it to the other processors 
        gmsh_cell_type = list(topology_data.keys())[0]    
        properties = gmsh.model.mesh.getElementProperties(gmsh_cell_type)
        name, dim, order, num_nodes, local_coords, _ = properties
        cells = topology_data[gmsh_cell_type]["topology"]
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([gmsh_cell_type, num_nodes], root=0)
    else:        
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, geometry_data = np.empty([0, num_nodes]), np.empty([0, gdim])

    #Permute topology data from MSH-ordering to dolfinx-ordering
    ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
    gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm]
    
    # Create distributed mesh
    mesh = create_mesh(MPI.COMM_WORLD, cells, geometry_data[:, :gdim], ufl_domain)
    return mesh

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from pathlib import Path

    Lcrack = 0.5
    dist_min = .1
    dist_max = .3
    mesh = generate_mesh_with_crack(Lcrack=Lcrack,
                     Ly=.5,
                     lc=.1, # caracteristic length of the mesh
                     refinement_ratio=10, # how much it is refined at the tip zone
                     dist_min=dist_min, # radius of tip zone
                     dist_max=dist_max # radius of the transition zone 
                     )

    Path("output").mkdir(parents=True, exist_ok=True)    
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/elasticity-demo.xdmf", "w") as file:
        file.write_mesh(mesh)
    