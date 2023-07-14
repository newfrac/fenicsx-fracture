import gmsh
import math
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI

def generate_mesh(Geometrical_data, Mesh_data):
    
    L, H, d, a = Geometrical_data.get('L'), Geometrical_data.get('H'), Geometrical_data.get('d'), Geometrical_data.get('a')
    m0, m1, m2 = Mesh_data.get('m0'), Mesh_data.get('m1'), Mesh_data.get('m2')
    
    gmsh.initialize()
    gdim = 2
    
    # .............. Parallel computation: in this case, we work with just one processor
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    
    # .............. Geometrical points in the mesh
    p1 = gmsh.model.geo.addPoint(0.,0.,0, m0)
    p2 = gmsh.model.geo.addPoint(H-a,0.,0, m1)
    p3 = gmsh.model.geo.addPoint(H-a/2,d/2,0, m1)
    p4 = gmsh.model.geo.addPoint(H-a/2,L/2,0, m2)
    p5 = gmsh.model.geo.addPoint(-a/2,L/2,0, m2)
    p6 = gmsh.model.geo.addPoint(-a/2,d/2,0, m1)

    #............... Geometrical lines
    l1  = gmsh.model.geo.addLine(p1,p2)
    l2  = gmsh.model.geo.addLine(p2,p3)
    l3  = gmsh.model.geo.addLine(p3,p4)
    l4  = gmsh.model.geo.addLine(p4,p5)
    l5  = gmsh.model.geo.addLine(p5,p6)
    l6  = gmsh.model.geo.addLine(p6,p1)

    #............... Geometrical loops and surfaces
    cloop1   = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4,l5,l6])
    surface1 = gmsh.model.geo.addPlaneSurface([cloop1])

    #............... Model composition
    gmsh.model.geo.synchronize()  

    #............... Physical groups: the domain and the vertical symmetric line 
    domain   = gmsh.model.addPhysicalGroup(gdim, [surface1], 1)
    hor_load = gmsh.model.addPhysicalGroup(gdim-1, [l4], 1)

    #...............  Mesh generation
    gmsh.model.mesh.generate(gdim)

    #............... Mesh importation in the dolfinx format
    mesh, _, facets = model_to_mesh(gmsh.model,mesh_comm,model_rank,gdim=gdim)

    gmsh.finalize()
    
    return mesh, facets
    
    
    
    
    
    
    
    
