from mpi4py import MPI
import dolfinx
import ufl
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from ufl.algorithms.ad import expand_derivatives


butcher_tableau = GaussLegendre(1)
ns = butcher_tableau.num_stages


N = 100
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

msh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[x0,y0],[x1,y1]], [N, N] )
V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
MC = MeshConstant(msh)
dt = MC.Constant(10.0/N)
t = MC.Constant(0.0)
x, y = ufl.SpatialCoordinate(msh)
S = dolfinx.fem.Constant(msh, 2.0)
C = dolfinx.fem.Constant(msh, 1000.0)
B = (x- dolfinx.fem.Constant(msh, x0))*(x- dolfinx.fem.Constant(msh, x1))*(y- dolfinx.fem.Constant(msh, y0))*(y- dolfinx.fem.Constant(msh, y1))/C
R = (x * x + y * y) ** 0.5
uexact = B * ufl.atan(t)*(ufl.pi / 2.0 - ufl.atan(S * (R - t)))
rhs = expand_derivatives(ufl.diff(uexact, t)) - ufl.div(ufl.grad(uexact))

u_exact_expr = dolfinx.fem.Expression(uexact, V.element.interpolation_points)
u = dolfinx.fem.Function(V)
u.interpolate(u_exact_expr)
v = ufl.TestFunction(V)

F = ufl.inner(Dt(u), v)*ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - ufl.inner(rhs, v)*ufl.dx

msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
bc = dolfinx.fem.dirichletbc(0.0, boundary_dofs, V)

lu_params = {"ksp_type": "preonly","pc_type": "lu"}
stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=[bc], solver_parameters=lu_params)




breakpoint()