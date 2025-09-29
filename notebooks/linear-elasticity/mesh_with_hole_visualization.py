import gmsh
from dolfinx.io import gmshio


def create_rectangle_mesh_gmsh(
    L: float,
    H: float,
    mesh_size: float,
    mesh_file: str = None,
    structured: bool = False,
    element_order: int = 1,
    hole_radius: float = None,
):
    """
    Create a rectangular mesh using gmsh and convert to DOLFINx.
    """
    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("rectangle")

    # Create rectangle points centered at origin
    p1 = gmsh.model.geo.addPoint(-L / 2, -H / 2, 0, mesh_size)  # bottom-left
    p2 = gmsh.model.geo.addPoint(L / 2, -H / 2, 0, mesh_size)  # bottom-right
    p3 = gmsh.model.geo.addPoint(L / 2, H / 2, 0, mesh_size)  # top-right
    p4 = gmsh.model.geo.addPoint(-L / 2, H / 2, 0, mesh_size)  # top-left

    # Create rectangle lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left

    # Create curve loop for rectangle
    rect_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # Create hole if requested
    if hole_radius is not None:
        # Center hole at origin (0, 0)
        cx, cy = 0.0, 0.0

        # Create circle for hole
        center = gmsh.model.geo.addPoint(cx, cy, 0, mesh_size / 2)

        # Create circle arc points
        p_right = gmsh.model.geo.addPoint(
            cx + hole_radius, cy, 0, mesh_size / 2
        )
        p_top = gmsh.model.geo.addPoint(cx, cy + hole_radius, 0, mesh_size / 2)
        p_left = gmsh.model.geo.addPoint(cx - hole_radius, cy, 0, mesh_size / 2)
        p_bottom = gmsh.model.geo.addPoint(
            cx, cy - hole_radius, 0, mesh_size / 2
        )

        # Create circle arcs
        arc1 = gmsh.model.geo.addCircleArc(p_right, center, p_top)
        arc2 = gmsh.model.geo.addCircleArc(p_top, center, p_left)
        arc3 = gmsh.model.geo.addCircleArc(p_left, center, p_bottom)
        arc4 = gmsh.model.geo.addCircleArc(p_bottom, center, p_right)

        # Create curve loop for hole
        hole_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])

        # Create surface with hole
        surface = gmsh.model.geo.addPlaneSurface([rect_loop, hole_loop])

        # Add physical group for hole boundary
        gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], 6, "hole")
    else:
        # Create surface without hole
        surface = gmsh.model.geo.addPlaneSurface([rect_loop])

    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # Configure mesh algorithm based on structured flag
    if structured:
        # For structured quad meshes
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

        # Calculate number of divisions based on mesh size
        n_x = max(2, int(L / mesh_size))
        n_y = max(2, int(H / mesh_size))

        # Set transfinite curves
        gmsh.model.mesh.setTransfiniteCurve(l1, n_x + 1)  # bottom
        gmsh.model.mesh.setTransfiniteCurve(l3, n_x + 1)  # top
        gmsh.model.mesh.setTransfiniteCurve(l2, n_y + 1)  # right
        gmsh.model.mesh.setTransfiniteCurve(l4, n_y + 1)  # left

        # Set transfinite surface for structured quad mesh
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.setRecombine(2, surface, 45)

        print(f"Creating structured quad mesh: {n_x}x{n_y} elements")
    else:
        # For unstructured triangle meshes
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
        gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Keep triangles

    # Add physical groups for boundary conditions
    gmsh.model.addPhysicalGroup(1, [l1], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [l2], 2, "right")
    gmsh.model.addPhysicalGroup(1, [l3], 3, "top")
    gmsh.model.addPhysicalGroup(1, [l4], 4, "left")
    gmsh.model.addPhysicalGroup(2, [surface], 5, "domain")

    # Set element order
    gmsh.model.mesh.setOrder(element_order)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Convert to DOLFINx mesh
    domain, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )

    # Clean up
    gmsh.finalize()

    return domain, cell_tags, facet_tags


# Create and visualize mesh
domain_hole, ct, ft = create_rectangle_mesh_gmsh(
    1,
    1.2,
    0.1,
    hole_radius=0.4,
    mesh_file="rectangle_hole",
)

print(f"   Vertices: {domain_hole.geometry.x.shape[0]}")
print(f"   Cells: {domain_hole.topology.index_map(2).size_local}")

# Visualize with pyvista
import pyvista
import dolfinx.plot

topology, cell_types, x = dolfinx.plot.vtk_mesh(domain_hole)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightblue")
plotter.camera_position = "xy"
plotter.add_title("Rectangle with Hole Mesh")

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("rectangle_with_hole_mesh.png")
