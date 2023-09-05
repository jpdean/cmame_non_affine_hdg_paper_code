from dolfinx import mesh, fem, io
import ufl
from ufl import inner, grad, dot, div, outer
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.cpp.fem import compute_integration_domains
from utils import (norm_L2, domain_average, normal_jump_error,
                   par_print)
from enum import Enum


class Scheme(Enum):
    # Scheme from https://doi.org/10.1016/j.cma.2019.112619
    RW = 1
    # Scheme from "Design and analysis of an exactly divergence-free
    # hybridized discontinuous Galerkin method for incompressible
    # flows on meshes with quadrilateral cells" by
    # J. P. Dean, S. Rhebergen, and G. N. Wells
    DRW = 2


def solve(k, nu, scheme, msh, mt, boundary_ids,
          boundary_conditions, f, u_e, p_e):
    # Create a sub-mesh of the facets of msh to allow facet function
    # spaces to be created
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    facet_imap = msh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    facets = np.arange(num_facets, dtype=np.int32)
    facet_mesh, facet_mesh_to_msh = mesh.create_submesh(
        msh, fdim, facets)[0:2]

    # Create the function spaces for the problem
    if scheme == Scheme.RW:
        V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
        Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
    else:
        V = fem.FunctionSpace(msh, ("Discontinuous Raviart-Thomas", k + 1))
        Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
    Vbar = fem.VectorFunctionSpace(
        facet_mesh, ("Discontinuous Lagrange", k))
    Qbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

    # Create functions
    u_h = fem.Function(V)  # Cell velocity
    ubar_h = fem.Function(Vbar)  # Facet velocity
    ubar_h.name = "ubar"
    p_h = fem.Function(Q)  # Cell pressure
    p_h.name = "p"
    pbar_h = fem.Function(Qbar)  # Facet pressure
    pbar_h.name = "pbar"

    # Create integration measures
    all_facets_tag = 0
    all_facets = []
    num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
    for cell in range(msh.topology.index_map(tdim).size_local):
        for local_facet in range(num_cell_facets):
            all_facets.extend([cell, local_facet])
    facet_integration_entities = [(all_facets_tag, all_facets)]
    facet_integration_entities += compute_integration_domains(
        fem.IntegralType.exterior_facet, mt._cpp_object)
    dx_c = ufl.Measure("dx", domain=msh)
    ds_c = ufl.Measure(
        "ds", subdomain_data=facet_integration_entities, domain=msh)
    dx_f = ufl.Measure("dx", domain=facet_mesh)

    # We write the mixed domain forms as integrals over msh. Hence, we must
    # provide a map from facets in msh to cells in facet_mesh. This is the
    # 'inverse' of facet_mesh_to_msh, which we compute as follows:
    msh_to_facet_mesh = np.full(num_facets, -1)
    msh_to_facet_mesh[facet_mesh_to_msh] = np.arange(
        len(facet_mesh_to_msh))
    entity_maps = {facet_mesh: msh_to_facet_mesh}

    # Trial and test functions
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)
    pbar, qbar = ufl.TrialFunction(Qbar), ufl.TestFunction(Qbar)

    # Define finite element forms
    h = ufl.CellDiameter(msh)
    n = ufl.FacetNormal(msh)
    gamma = 16.0 * k**2 / h  # Scaled penalty parameter

    nu = fem.Constant(msh, PETSc.ScalarType(nu))
    a_00 = nu * inner(grad(u), grad(v)) * dx_c \
        - nu * inner(grad(u), outer(v, n)) * ds_c(all_facets_tag) \
        + nu * gamma * inner(outer(u, n), outer(v, n)) * ds_c(all_facets_tag) \
        - nu * inner(outer(u, n), grad(v)) * ds_c(all_facets_tag)
    a_01 = fem.form(- inner(p * ufl.Identity(msh.topology.dim),
                    grad(v)) * dx_c)
    a_02 = - nu * gamma * inner(
        outer(ubar, n), outer(v, n)) * ds_c(all_facets_tag) \
        + nu * inner(outer(ubar, n), grad(v)) * ds_c(all_facets_tag)
    a_03 = fem.form(inner(pbar * ufl.Identity(msh.topology.dim),
                          outer(v, n)) * ds_c(all_facets_tag),
                    entity_maps=entity_maps)
    a_10 = fem.form(inner(u, grad(q)) * dx_c -
                    inner(dot(u, n), q) * ds_c(all_facets_tag))
    a_20 = - nu * inner(grad(u), outer(vbar, n)) * ds_c(all_facets_tag) \
        + nu * gamma * inner(outer(u, n), outer(vbar, n)
                             ) * ds_c(all_facets_tag)
    a_30 = fem.form(inner(dot(u, n), qbar) *
                    ds_c(all_facets_tag), entity_maps=entity_maps)
    a_23 = fem.form(
        inner(pbar * ufl.Identity(tdim), outer(vbar, n)) *
        ds_c(all_facets_tag),
        entity_maps=entity_maps)
    a_32 = fem.form(- inner(dot(ubar, n), qbar) * ds_c,
                    entity_maps=entity_maps)
    a_22 = - nu * gamma * \
        inner(outer(ubar, n), outer(vbar, n)) * ds_c(all_facets_tag)

    L_2 = inner(fem.Constant(msh, [PETSc.ScalarType(0.0)
                                   for i in range(tdim)]),
                vbar) * ds_c(all_facets_tag)

    # Apply boundary conditions
    bcs = []
    for name, bc in boundary_conditions.items():
        bound_id = boundary_ids[name]
        bc_expr = bc
        bc_func = fem.Function(Vbar)
        bc_func.interpolate(bc_expr)
        facets = msh_to_facet_mesh[mt.indices[mt.values == bound_id]]
        dofs = fem.locate_dofs_topological(Vbar, fdim, facets)
        bcs.append(fem.dirichletbc(bc_func, dofs))

    # Compile forms
    a_00 = fem.form(a_00)
    a_02 = fem.form(a_02, entity_maps=entity_maps)
    a_20 = fem.form(a_20, entity_maps=entity_maps)
    a_22 = fem.form(a_22, entity_maps=entity_maps)

    L_0 = fem.form(inner(f, v) * dx_c)
    L_1 = fem.form(inner(fem.Constant(msh, 0.0), q) * dx_c)
    L_2 = fem.form(L_2, entity_maps=entity_maps)
    L_3 = fem.form(inner(fem.Constant(
        facet_mesh, PETSc.ScalarType(0.0)), qbar) * dx_f)

    # Define block structure
    a = [[a_00, a_01, a_02, a_03],
         [a_10, None, None, None],
         [a_20, None, a_22, a_23],
         [a_30, None, a_32, None]]
    L = [L_0, L_1, L_2, L_3]

    # Assemble matrix
    A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
    A.assemble()

    # Assemble vector
    b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

    # Setup solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    # Settings to handle the nullspace of constants
    opts["mat_mumps_icntl_6"] = 2
    opts["mat_mumps_icntl_14"] = 100
    ksp.setFromOptions()

    # Compute solution
    x = A.createVecRight()
    ksp.solve(b, x)

    # Recover solution
    u_offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    p_offset = u_offset + \
        Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    ubar_offset = \
        p_offset + Vbar.dofmap.index_map.size_local * \
        Vbar.dofmap.index_map_bs
    u_h.x.array[:u_offset] = x.array_r[:u_offset]
    u_h.x.scatter_forward()
    p_h.x.array[:p_offset - u_offset] = x.array_r[u_offset:p_offset]
    p_h.x.scatter_forward()
    ubar_h.x.array[:ubar_offset - p_offset] = \
        x.array_r[p_offset:ubar_offset]
    ubar_h.x.scatter_forward()
    pbar_h.x.array[:(len(x.array_r) - ubar_offset)] = \
        x.array_r[ubar_offset:]
    pbar_h.x.scatter_forward()

    # The scheme DRW uses a broken Raviart-Thomas space for the
    # velocity field. We interpolate this into a broken Lagrange
    # space of degree k + 1 (which can represent it exactly) for
    # artifact free visualisation
    V_vis = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))
    u_vis = fem.Function(V_vis)
    u_vis.name = "u"
    u_vis.interpolate(u_h)

    # Write solution to file
    vis_files = [("u.bp", u_vis), ("p.bp", p_h),
                 ("ubar.bp", ubar_h), ("pbar.bp", pbar_h)]
    for file_name, func in vis_files:
        with io.VTXWriter(msh.comm, file_name, func) as f:
            f.write(0.0)

    # Compute error in solution
    x = ufl.SpatialCoordinate(msh)
    xbar = ufl.SpatialCoordinate(facet_mesh)

    e_u = norm_L2(msh.comm, u_h - u_e(x))
    e_ubar = norm_L2(msh.comm, ubar_h - u_e(xbar))

    p_h_avg = domain_average(msh, p_h)
    p_e_avg = domain_average(msh, p_e(x))
    e_p = norm_L2(msh.comm, (p_h - p_h_avg) - (p_e(x) - p_e_avg))
    pbar_h_avg = domain_average(facet_mesh, pbar_h)
    pbar_e_avg = domain_average(facet_mesh, p_e(xbar))
    e_pbar = norm_L2(msh.comm, (pbar_h - pbar_h_avg) -
                     (p_e(xbar) - pbar_e_avg))

    e_div_u = norm_L2(msh.comm, div(u_h))
    e_jump_u = normal_jump_error(msh, u_h)

    # Print errors
    comm = msh.comm
    par_print(comm, f"e_u = {e_u}")
    par_print(comm, f"e_ubar = {e_ubar}")
    par_print(comm, f"e_p = {e_p}")
    par_print(comm, f"e_pbar = {e_pbar}")
    par_print(comm, f"e_div_u = {e_div_u}")
    par_print(comm, f"e_jump_u = {e_jump_u}")
