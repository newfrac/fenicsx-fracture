import mpi4py
import numpy as np
import dolfinx  # FEM in python
from dolfinx import fem
from typing import Optional
import ufl
import typing


def evaluate_at_points(
    points: np.ndarray | typing.Sequence, function: fem.Function
) -> Optional[np.ndarray]:
    """
    Evaluate the `function` at all `points`.

    Note:
        The points `p0,p1,..` should be ordered as `p0_x, p1_x, ..., p0_y, p1_y, ... p0_z, p1_z, ...`.

    Note:
        It is assumed that `points` is the same on all processes.

    Note:
        Values are only returned on process 0.
    """
    mesh = function.function_space.mesh

    if isinstance(points, list):
        points = np.array(points)
    if len(points.shape) == 1:
        if points.size < 3:
            # Pad point with zeros
            _points = np.zeros((3, 1), dtype=mesh.geometry.x.dtype)
            _points[: len(points), 0] = points
        else:
            if len(points) % 3 != 0 and mesh.geometry.dim == 2:
                # Pad points with extra 0
                _points = np.zeros(
                    (3, len(points) // 2), dtype=mesh.geometry.x.dtype
                )
                _points[:2, :] = np.array(points).reshape(2, -1)
            elif len(points) % 3 == 0:
                _points = np.zeros(
                    (3, len(points) // 3), dtype=mesh.geometry.x.dtype
                )
                _points[:, :] = np.array(points).reshape(3, -1)
            else:
                raise RuntimeError(
                    "Received list of points that cannot be formatted as a (n, 3) array"
                )
    else:
        _points = np.array(points, dtype=mesh.geometry.x.dtype)

    comm = mesh.comm
    if comm.rank == 0:
        input_points = _points.T
    else:
        input_points = np.empty((0, 3), dtype=points.dtype)

    owernship = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, input_points, 1e-6
    )
    values = function.eval(
        np.array(owernship.dest_points).reshape(-1, 3), owernship.dest_cells
    ).reshape(-1, function.function_space.dofmap.bs)
    if comm.rank != 0:
        assert np.allclose(owernship.dest_owners, 0)
    gathered_values = comm.gather(values, root=0)
    # print(f"src_owner rank {comm.rank}:", owernship.src_owner)
    # print(f"dest_cells rank {comm.rank}:", owernship.dest_cells)
    # print(f"dest_owner rank {comm.rank}:", owernship.dest_owners)
    src_counter = np.zeros(comm.size, dtype=np.int32)
    bs = function.function_space.dofmap.bs
    values = np.zeros((input_points.shape[0], bs), dtype=function.x.array.dtype)
    if comm.rank == 0:
        for i, owner in enumerate(owernship.src_owner):
            if owner == -1:
                print(f"Could not find point in mesh for {input_points[i]}")
                continue
            values[i] = gathered_values[owner][src_counter[owner]]
            src_counter[owner] += 1
    else:
        return None
    return values


if __name__ == "__main__":
    from mpi4py import MPI
    from dolfinx import fem
    import numpy as np
    from dolfinx.mesh import create_box, CellType, GhostMode
    from petsc4py import PETSc
    import matplotlib.pyplot as plt

    dtype = PETSc.ScalarType  # type: ignore

    import ufl

    comm = MPI.COMM_WORLD
    msh = create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
        [16, 16, 16],
        CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V)
    u.interpolate(lambda x: (x[0], x[1], x[2]))
    xs = np.linspace(0, 1, 100)
    ys = np.zeros_like(xs)
    zs = np.zeros_like(xs)
    points = np.array([xs, ys, zs])
    points_flat = points.reshape(3 * xs.size, -1)
    data = evaluate_at_points(points, u)
    if comm.rank == 0:
        plt.plot(xs, data[:, 0], "o", label="x")
        plt.plot(xs, data[:, 1], "o", label="y")
        plt.plot(xs, data[:, 2], ".", label="z")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.savefig("points.png")
