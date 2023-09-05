import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh
import sys


def par_print(comm, string):
    "Simple function for printing in parallel"
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def norm_L2(comm, v, measure=ufl.dx):
    "Compute the L2-norm of a function"
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(ufl.inner(v, v) * measure)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * ufl.dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * ufl.dx)), op=MPI.SUM)


def normal_jump_error(msh, v):
    "Compute the normal jump error"
    n = ufl.FacetNormal(msh)
    return norm_L2(msh.comm, ufl.jump(v, n), measure=ufl.dS)


def create_trap_mesh(comm, n, corners, offset_scale=0.25,
                     ghost_mode=mesh.GhostMode.none):
    """Creates a trapezium mesh by creating a square mesh and offsetting
    the points by a fraction of the cell diameter. The offset can be
    controlled with offset_scale.
    Parameters:
        n: Number of elements in each direction
        corners: coordinates of the bottom left and upper right corners
        offset_scale: Fraction of cell diameter to offset the points
        ghost_mode: The ghost mode
    Returns:
        mesh: A dolfinx mesh object
    """
    if n[1] % 2 != 0:
        raise Exception("n[1] must be even")

    if comm.rank == 0:
        # Width of each element
        h = [(corners[1][i] - corners[0][i]) / n[i] for i in range(2)]

        x = []
        for j in range(n[1] + 1):
            for i in range(n[0] + 1):
                offset = 0
                if j % 2 != 0:
                    if i % 2 == 0:
                        offset = offset_scale * h[1]
                    else:
                        offset = - offset_scale * h[1]
                x.append([corners[0][0] + i * h[0],
                          corners[0][1] + j * h[1] + offset])
        x = np.array(x)

        cells = []
        for j in range(n[1]):
            for i in range(n[0]):
                node_0 = i + (n[0] + 1) * j
                node_1 = i + (n[0] + 1) * j + 1
                node_2 = i + (n[0] + 1) * (j + 1)
                node_3 = i + (n[0] + 1) * (j + 1) + 1

                cells.append([node_0, node_1, node_2, node_3])
        cells = np.array(cells)
    else:
        cells, x = np.empty([0, 3]), np.empty([0, 2])

    ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    partitioner = mesh.create_cell_partitioner(ghost_mode)
    msh = mesh.create_mesh(comm, cells, x, ufl_mesh, partitioner=partitioner)

    return msh
