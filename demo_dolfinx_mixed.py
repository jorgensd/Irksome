from mpi4py import MPI
import dolfinx
import ufl
from irksome import GaussLegendre, Dt, MeshConstant
from irksome.tools import get_stage_space
from irksome.backends.dolfinx import dirichletbc
from ufl import pi, atan, div, grad, inner, dx

butcher_tableau = GaussLegendre(3)
N = 25

x0 = 0.0
x1 = 1.0
y0 = 0.0
y1 = 1.0
msh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[x0, y0], [x1, y1]], [N, N])
msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
Constant = lambda val: dolfinx.fem.Constant(msh, val)

#V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (2,)))
import basix
el_u = basix.ufl.element("Lagrange", msh.basix_cell(), 3, shape=(2, ))
el_p = basix.ufl.element("Lagrange", msh.basix_cell(), 2)
U = dolfinx.fem.functionspace(msh, el_u)
Q = dolfinx.fem.functionspace(msh, el_p)
V = ufl.MixedFunctionSpace(U, Q)

MC = MeshConstant(msh, backend="dolfinx")
t = MC.Constant(0.0)
dt = MC.Constant(1.0/N)
(x, y) = ufl.SpatialCoordinate(msh)

uexact = ufl.as_vector([x*t + y**2, -y*t+t*(x**2)])
pexact = x + y * t

u_rhs = Dt(uexact) - div(grad(uexact)) + grad(pexact)
p_rhs = -div(uexact)

u = dolfinx.fem.Function(U)
p = dolfinx.fem.Function(Q)
(v, q) = ufl.TestFunctions(V)
F = (inner(Dt(u), v)*dx
         + inner(grad(u), grad(v))*dx
         - inner(p, div(v))*dx
         - inner(div(u), q)*dx
         - inner(u_rhs, v)*dx
         - inner(p_rhs, q)*dx)

u_bndry = dolfinx.fem.Function(U)
boundary_dofs = dolfinx.fem.locate_dofs_topological(U, msh.topology.dim-1, boundary_facets)

# Dirichlet BCs (irksome style with UFL expression)
bc = dirichletbc(uexact, boundary_dofs, U)
bcs = [bc]

# Initial conditions
bc_expr = dolfinx.fem.Expression(uexact, U.element.interpolation_points)
u.interpolate(bc_expr)
p_avg = msh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(pexact*dx)), op=MPI.SUM)
p.interpolate(dolfinx.fem.Expression(pexact - p_avg, Q.element.interpolation_points))

# Get the function space for the stage-coupled problem and a function to hold the stages we're computing::

Vbig = get_stage_space(V, butcher_tableau.num_stages, backend="dolfinx")

# Get the variational form and bcs for the stage-coupled variational problem::
# Fnew, bcnew = getForm(F, butcher_tableau, t, dt, u, k, bcs=bcs, backend="dolfinx")

from irksome.stage_derivative import StageDerivativeTimeStepper
from irksome.tools import AI
solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "snes_monitor": None, "snes_error_if_not_converged": True,
                   "ksp_monitor": None, "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True,
                   "snes_linesearch_type": "none", "snes_atol": 1e-6, "snes_rtol": 1e-6,}
from irksome.backends.dolfinx import MixedCoefficientList       
coeffs = MixedCoefficientList([u, p])
breakpoint()
linear_stepper = StageDerivativeTimeStepper(F, butcher_tableau, t, dt, coeffs, bcs=bcs, Fp=None, bc_type="DAE", splitting=AI,
                        solver_parameters=solver_parameters, backend="dolfinx")



from irksome.backends.dolfinx import norm

vtx_writer = dolfinx.io.VTXWriter(msh.comm, "u_mixed.bp", [u])
while (float(t) < 1.0):
    if (float(t) + float(dt) > 1.0):
        dt.assign(1.0 - float(t))

    linear_stepper.advance()
   
    t.assign(float(t) + float(dt))
    print(float(t), norm(u - uexact, norm_type="L2", mesh=msh))

    vtx_writer.write(float(t))

vtx_writer.close()




# from irksome.backends.dolfinx import create_variational_problem, create_variational_solver
# problem = create_variational_problem(Fnew, k, bcs=bcnew)
# solver = create_variational_solver(problem, solver_parameters=solver_parameters)
# solver.solve()
# breakpoint()