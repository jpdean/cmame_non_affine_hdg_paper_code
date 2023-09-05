import numpy as np
from hdg_stokes import Scheme, solve
from mpi4py import MPI
from utils import create_trap_mesh
from dolfinx import mesh
import ufl


def gamma_marker(x):
    "A marker function to identify the domain boundary"
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | \
        np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def u_e(x, module=ufl):
    "Expression for the exact velocity"
    u = (0.0 * x[0], 0.0 * x[1])
    if module == ufl:
        return ufl.as_vector(u)
    else:
        assert module == np
        return np.vstack(u)


def p_e(x, module=ufl):
    "Expression for the exact pressure"
    return c * (x[1]**3 - x[1]**2 / 2 + x[1] - 7 / 12)


if __name__ == "__main__":
    # Simulation parameters
    n = 8  # Number of cells in each direction
    k = 2  # Polynomial degree
    nu = 1.0  # Kinermatic viscosity
    c = 1.0e4  # Force parameter
    scheme = Scheme.DRW  # Numerical scheme (RW or DRW)

    # Create trapezium shaped mesh
    comm = MPI.COMM_WORLD
    msh = create_trap_mesh(
        comm, (n, n), ((0, 0), (1, 1)), offset_scale=0.25,
        ghost_mode=mesh.GhostMode.none)

    # Boundary conditions
    boundary_ids = {"gamma": 1}
    boundary_conditions = {"gamma": lambda x: u_e(x, module=np)}
    boundary_facets = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, gamma_marker)
    values = np.full_like(
        boundary_facets, boundary_ids["gamma"], dtype=np.intc)
    perm = np.argsort(boundary_facets)
    mt = mesh.meshtags(
        msh, msh.topology.dim - 1, boundary_facets[perm], values[perm])

    # Right-hand side
    x = ufl.SpatialCoordinate(msh)
    f = ufl.as_vector((0.0, c * (3 * x[1]**2 - x[1] + 1)))

    solve(k, nu, scheme, msh, mt, boundary_ids,
          boundary_conditions, f, u_e, p_e)
