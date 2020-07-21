import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import gym
import gym_custom_envs
from pydrake.all import (
  DiagramBuilder, DynamicProgrammingOptions, FittedValueIteration, 
  LinearSystem, Simulator
)
env = gym.make("DoubleIntegrator-v0")
from IPython import get_ipython
from pydrake.systems.primitives import LogOutput

# xdot = Ax + Bu
# y = Cx + Du
def DoubleIntegrator():
  return LinearSystem(A=np.mat('0 1; 0 0'),
                      B=np.mat('0; 1'),
                      C=np.eye(2),
                      D=np.zeros((2,1)))

plant = DoubleIntegrator()
simulator = Simulator(plant)
options = DynamicProgrammingOptions()

qbins = np.linspace(-5., 5., 31)
qdotbins = np.linspace(-5., 5., 31)
state_grid = [set(qbins), set(qdotbins)]

input_limit = 1.
input_grid = [set(np.linspace(-input_limit, input_limit, 9))]
timestep = 0.01

[Q, Qdot] = np.meshgrid(qbins, qdotbins)

print('\n ===================== \n')
print('Q = ')
print(Q)
print('Qdot =')
print(Qdot)
# print('Ugrid = ')
# print(input_grid)
print('\n ===================== \n')

def draw(iteration, mesh, cost_to_go, policy):
    # Drawing is slow, don't draw every frame.
    if iteration % 20 != 0:
        return
    plt.suptitle("iteration " + str(iteration))
    J = np.reshape(cost_to_go, Q.shape)
    surf = ax1.plot_surface(Q, Qdot, J, rstride=1, cstride=1, cmap=cm.jet)

    Pi = np.reshape(policy, Q.shape)
    surf2 = ax2.plot_surface(Q, Qdot, Pi, rstride=1, cstride=1, cmap=cm.jet)

    if plt.get_backend() != u"template":
        fig.canvas.draw()
        plt.pause(1e-10)
    fig.savefig('img/quad/'+str(iteration)+'.png')
    surf.remove()
    surf2.remove()

options.visualization_callback = draw

def solve():
    policy, cost_to_go = FittedValueIteration(simulator, cost_function, state_grid,
                                              input_grid, timestep, options)

    J = np.reshape(cost_to_go, Q.shape)
    surf = ax1.plot_surface(Q, Qdot, J, rstride=1, cstride=1, cmap=cm.jet)
    Pi = np.reshape(policy.get_output_values(), Q.shape)
    surf2 = ax2.plot_surface(Q, Qdot, Pi, rstride=1, cstride=1, cmap=cm.jet)
    return policy

def simulate():
    # Animate the resulting policy.
    builder = DiagramBuilder()
    plant = builder.AddSystem(DoubleIntegrator())

    vi_policy = builder.AddSystem(policy)
    builder.Connect(plant.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0), plant.get_input_port(0))
    diagram = builder.Build()
    simulator = Simulator(diagram)

    logger = LogOutput(plant.get_output_port(0), builder)
    logger.set_name("logger")

    simulator.get_mutable_context().SetContinuousState(env.reset())

    simulator.AdvanceTo(10. if get_ipython() is not None else 5.)

    return logger


def min_time_cost(context):
    x = context.get_continuous_state_vector().CopyToVector()
    if x.dot(x) < .05:
        return 0.
    return 1.

def quadratic_regulator_cost(context):
    x = context.get_continuous_state_vector().CopyToVector()
    u = plant.EvalVectorInput(context, 0).CopyToVector()
    return x.dot(x) + u.dot(u)

# Minimum Time
cost_function = min_time_cost
options.convergence_tol = 0.001

# Quadratic Cost
cost_function = quadratic_regulator_cost
options.convergence_tol = 0.1

fig = plt.figure(figsize=(9, 4))
ax1, ax2 = fig.subplots(1, 2, subplot_kw=dict(projection='3d'))
ax1.set_xlabel("q")
ax1.set_ylabel("qdot")
ax1.set_title("Quadratic-Cost-to-Go")

ax2.set_xlabel("q")
ax2.set_ylabel("qdot")
ax2.set_title("Bang-Bang-Policy")

policy = solve()
