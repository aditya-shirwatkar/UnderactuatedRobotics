import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import eq, MathematicalProgram, Solve, Variable
from pydrake.math import sin, cos
# from pydrake.solvers.ipopt import IpoptSolver

# q = [x, y, theta]
# dq = [dx, dy, dtheta]
# ddq = [-(u1 + u2)sin(theta)/m, (u1 + u2)*cos(theta) - mg/m, (u1 - u2)l/2/I]

N = 284  # Note: I had to do a manual "line search" to find this.

prog = MathematicalProgram()

# Create decision variables
u = np.empty((2, N-1), dtype=Variable)
x = np.empty((3, N), dtype=Variable)
dx = np.empty((3, N), dtype=Variable)

# Discrete-time approximation of the double integrator.
dt = 0.01
A = np.eye(2) + dt*np.mat('0 1; 0 0')
print(A)
B = dt*np.mat('0; 1')
print(B)
m = 0.2
l = 0.5
g = 9.81
r = l/2
f = 10
I = m * (l**2)/12

for n in range(N):
  if n < N - 1:  
    u[:,n] = prog.NewContinuousVariables(2, 'u' + str(n))
  dx[:,n] = prog.NewContinuousVariables(3, 'dx' + str(n))
  x[:,n] = prog.NewContinuousVariables(3, 'x' + str(n))

# Add constraints
x0 = [-2., 0., 0.]
prog.AddBoundingBoxConstraint(x0, x0, x[:,0])

def constraint_evaluator1(x):
    return dt*np.array([(-(x[3,n] + x[4,n])*sin(x[2,n])/m), 
                    (x[3,n] + x[4,n])*cos(x[2,n]) - m*g/m,
                    (x[3,n] - x[4,n])*r/I])

def constraint_evaluator2(x):
    return dt*np.array([x[0,n], x[1,n], x[2,n]])

for n in range(N-1):
    # Will eventually be prog.AddConstraint(x[:,n+1] == A@x[:,n] + B@u[:,n])
    # See drake issues 12841 and 8315
    # prog.AddConstraint(eq(dx[0,n+1], dx[0,n] + dt*(-(u[0,n] + u[1,n])*sin(x[2,n])/m)))
    # prog.AddConstraint(eq(dx[1,n+1], dx[1,n] + dt*((u[0,n] + u[1,n])*cos(x[2,n]) - m*g/m)))
    # prog.AddConstraint(eq(dx[2,n+1], dx[2,n] + dt*((u[0,n] - u[1,n])*r/I)))

    prog.AddConstraint(constraint_evaluator1,
                        lb=np.array([-2.5, -2.5, -np.pi, -f*m*g, -f*m*g]),
                        ub=np.array([2.5, 2.5, np.pi, f*m*g, f*m*g]),
                        vars=[x[0, n], x[1, n], x[2, n], u[0, n], u[1, n]])

    prog.AddConstraint(constraint_evaluator2,
                        lb=np.array([-(f**2)]*3),
                        ub=np.array([f**2]*3),
                        vars=[dx]
                        )
    prog.AddConstraint(eq(x[1,n+1], x[1,n] + dt*dx[1,n]))
    prog.AddConstraint(eq(x[2,n+1], x[2,n] + dt*dx[2,n]))

    # prog.AddBoundingBoxConstraint(-f*m*g, f*m*g, u[:,n])
    # prog.AddBoundingBoxConstraint(-2.5, 2.5, x[0,n])
    # prog.AddBoundingBoxConstraint(-2.5, 2.5, x[1,n])
    # prog.AddBoundingBoxConstraint(-np.pi, np.pi, x[2,n])

xf = [0., 0., 0.]
prog.AddBoundingBoxConstraint(xf, xf, x[:, N-1])

# solver = IpoptSolver()
# result = solver.Solve(prog)

result = Solve(prog)

x_sol = result.GetSolution(x)
assert(result.is_success()), "Optimization failed"

plt.figure()
plt.plot(x_sol[0,:])
plt.plot(x_sol[1,:])
plt.plot(x_sol[2,:])
# plt.xlabel('q')
# plt.ylabel('qdot')
plt.show()
