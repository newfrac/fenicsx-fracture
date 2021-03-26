import gmsh
import numpy as np
import ufl
#import dolfinx.plotting
from dolfinx import Function, FunctionSpace
from dolfinx.io import (extract_gmsh_geometry, extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh
from mpi4py import MPI

def generate_mesh(Lx=1,Ly=1,lc=.015):
    # For further documentation see
    # - gmsh tutorials, e.g. see https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/python/t10.py 
    # - dolfinx-gmsh interface https://github.com/FEniCS/dolfinx/blob/master/python/demo/gmsh/demo_gmsh.py
    # 
    gmsh.initialize()
    model = gmsh.model()
    model.add("Rectangle")
    model.setCurrent("Rectangle")
    p1 = model.geo.addPoint(0.0, 0.0, 0, lc)
    p2 = model.geo.addPoint(Lx, 0, 0, lc)
    p3 = model.geo.addPoint(Lx, Ly, 0, lc)
    p4 = model.geo.addPoint(0, Ly, 0, lc)
    l1 = model.geo.addLine(p1, p2)
    l2 = model.geo.addLine(p2, p3)
    l3 = model.geo.addLine(p3, p4)
    l4 = model.geo.addLine(p4, p1)
    cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4])
    surface_1 = model.geo.addPlaneSurface([cloop1])

    model.geo.synchronize()
    surface_entities = [model[1] for model in model.getEntities(2)]
    model.addPhysicalGroup(2, surface_entities, tag=5)
    model.setPhysicalName(2, 5, "Rectangle surface")
    model.mesh.generate(2)
    # get mesh into fenics
    x = extract_gmsh_geometry(model, model_name="Rectangle")[:,0:2]
    gmsh_cell_id = model.mesh.getElementType("triangle", 1)
    topologies = extract_gmsh_topology_and_markers(model, "Rectangle")
    cells = topologies[gmsh_cell_id]["topology"]
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh_facet_id = model.mesh.getElementType("triangle", 1)
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, 2))
    return mesh

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from pathlib import Path

    Lcrack = 0.5
    dist_min = .1
    dist_max = .3
    mesh = generate_mesh(Lcrack=Lcrack,
                     Ly=.5,
                     lc=.1, # caracteristic length of the mesh
                     refinement_ratio=10, # how much it is refined at the tip zone
                     dist_min=dist_min, # radius of tip zone
                     dist_max=dist_max # radius of the transition zone 
                     )
    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='auto')
    #dolfinx.plotting.plot(mesh)
    #Path("output").mkdir(parents=True, exist_ok=True)    
    #plt.savefig("mesh.png")
