import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_custom_envs
from pydrake.all import eq, MathematicalProgram, Solve, Variable
from pydrake.math import sin, cos
from pydrake.solvers.ipopt import IpoptSolver
import time as tt
# q = [x, y, theta]
# dq = [dx, dy, dtheta]
# ddq = [-(u1 + u2)sin(theta)/m, (u1 + u2)*cos(theta) - mg/m, (u1 - u2)l/2/I]

N = 100  # Note: I had to do a manual "line search" to find this.

prog = MathematicalProgram()

# Create decision variables
u = np.empty((2, N), dtype=Variable)
x = np.empty((3, N), dtype=Variable)
dx = np.empty((3, N), dtype=Variable)

# Discrete-time approximation of the double integrator.
T = 2.
dt = T/N
env = gym.make('Quadrotor2D-trajopt-v0', dt=dt)
# A = np.eye(2) + dt*np.mat('0 1; 0 0')
# print(A)
# B = dt*np.mat('0; 1')
# print(B)
m = env.m
l = env.l
g = 9.81
r = l/2
f = 10
I = m * (l**2)/12

for n in range(N):
   
  u[:,n] = prog.NewContinuousVariables(2, 'u' + str(n))
  dx[:,n] = prog.NewContinuousVariables(3, 'dx' + str(n))
  x[:,n] = prog.NewContinuousVariables(3, 'x' + str(n))

s = env.reset().reshape(6, 1)
print(s)
# Add constraints
x0 = [s[0], s[1], s[2]]
prog.AddBoundingBoxConstraint(x0, x0, x[:,0])
x1 = [0]*3
prog.AddBoundingBoxConstraint(x1, x1, dx[:, 0])

def constraint_evaluator1(z):
    return np.array([(-(z[6] + z[7])*sin(z[2])/m)*dt + z[3] - z[8], 
                    ((z[6] + z[7])*cos(z[2]) - m*g/m)*dt + z[4] - z[9],
                    ((z[6] - z[7])*r/I)*dt + z[5] - z[10]
                    ])

def constraint_evaluator2(z):
    return np.array([z[3]*dt + z[0] - z[6], z[4]*dt + z[1] - z[7], z[5]*dt + z[2] - z[8]])

for n in range(N-1):
    # Will eventually be prog.AddConstraint(x[:,n+1] == A@x[:,n] + B@u[:,n])
    # See drake issues 12841 and 8315
    # prog.AddConstraint(eq(dx[0,n+1], dx[0,n] + dt*(-(u[0,n] + u[1,n])*sin(x[2,n])/m)))
    # prog.AddConstraint(eq(dx[1,n+1], dx[1,n] + dt*((u[0,n] + u[1,n])*cos(x[2,n]) - m*g/m)))
    # prog.AddConstraint(eq(dx[2,n+1], dx[2,n] + dt*((u[0,n] - u[1,n])*r/I)))
    prog.AddQuadraticCost(sum(x[:,n]**2) + sum(dx[:,n]**2) + sum(u[:,n]**2))
    prog.AddConstraint(constraint_evaluator1,
                        lb=np.array([0]*3),
                        ub=np.array([0]*3),
                        vars=[x[0, n], x[1, n], x[2, n], 
                              dx[0, n], dx[1, n], dx[2, n],
                              u[0, n], u[1, n],
                              dx[0, n+1], dx[1, n+1], dx[2, n+1],])

    prog.AddConstraint(constraint_evaluator2,
                        lb=np.array([0]*3),
                        ub=np.array([0]*3),
                        vars=[x[0, n], x[1, n], x[2, n], 
                              dx[0, n], dx[1, n], dx[2, n],
                              x[0, n+1], x[1, n+1], x[2, n+1],])

    # prog.AddConstraint(eq(x[1,n+1], x[1,n] + dt*dx[1,n]))
    # prog.AddConstraint(eq(x[2,n+1], x[2,n] + dt*dx[2,n]))

    prog.AddBoundingBoxConstraint([-f*m*g]*2, [f*m*g]*2, u[:,n])
    prog.AddBoundingBoxConstraint(-5, 5, x[0,n])
    prog.AddBoundingBoxConstraint(-5, 5, x[1,n])
    prog.AddBoundingBoxConstraint(-np.pi, np.pi, x[2,n])

xf = [0., 0., 0.]
prog.AddBoundingBoxConstraint(xf, xf, x[:, N-1])
uf = [m*g/2, m*g/2]
prog.AddBoundingBoxConstraint(uf, uf, u[:, N-1])
x1 = [0]*3
prog.AddBoundingBoxConstraint(x1, x1, dx[:, N-1])

solver = IpoptSolver()
result = solver.Solve(prog)

# result = Solve(prog)

x_sol = result.GetSolution(x)
dx_sol = result.GetSolution(dx)
u_sol = result.GetSolution(u)
assert(result.is_success()), "Optimization failed"
time = np.linspace(0, T, N)
# print(len(time))
plt.figure()
plt.subplot(311)
plt.plot(time, x_sol[0,:], 'r', label='x')
plt.plot(time, x_sol[1,:], 'g', label='y')
plt.plot(time, x_sol[2,:], 'b', label='theta')
plt.subplot(312)
plt.plot(time, dx_sol[0,:], 'r', label='dx')
plt.plot(time, dx_sol[1,:], 'g', label='dy')
plt.plot(time, dx_sol[2,:], 'b', label='theta')
plt.subplot(313)
plt.plot(time, u_sol[1,:], 'c', label='u2')
plt.plot(time, u_sol[0,:], 'm', label='u2')
# plt.xlabel('q')
# plt.ylabel('qdot')
plt.show()

done = False
j = 0
while not done:
    a = [x_sol[0,j], x_sol[1,j], x_sol[2,j], 
         dx_sol[0,j], dx_sol[1,j], dx_sol[2,j],
         u_sol[0,j], u_sol[1,j]]
    ns, c, done,_ = env.step(a)
    # s = ns.reshape(6, 1)
    env.render()
    tt.sleep(0.05)
    j += 1
env.close()