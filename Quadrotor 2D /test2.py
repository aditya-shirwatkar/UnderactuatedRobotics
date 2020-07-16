import math
import numpy as np

import matplotlib.pyplot as plt
from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType
# from pydrake.systems.pyplot_visualizer import PyPlotVisualizer
from pydrake.all import DirectCollocation, DirectTranscription, PiecewisePolynomial, Solve
from pydrake.systems.scalar_conversion import TemplateSystem

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html


# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("Quadrotor2D_")
def Quadrotor2D_(T):

    class Impl(LeafSystem_[T]):

        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter)
            # two inputs (thrust)
            self.DeclareVectorInputPort("u", BasicVector_[T](2))
            # six outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](6),
                                         self.CopyStateOut)
            # three positions, three velocities
            self.DeclareContinuousState(3, 3, 0)

            # parameters based on [Bouadi, Bouchoucha, Tadjine, 2007]
            self.length = 0.25  # length of rotor arm
            self.mass = 0.486  # mass of quadrotor
            self.inertia = 0.00383  # moment of inertia
            self.gravity = 9.81  # gravity

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            y = output.SetFromVector(x)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()
            q = x[:3]
            qdot = x[3:]
            qddot = np.array([
                -np.sin(q[2]) / self.mass * (u[0] + u[1]),
                np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                self.length / self.inertia * (u[0] - u[1])
            ])
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot)))

    return Impl

# class SimpleContinuousTimeSystem(LeafSystem):
#     def __init__(self):
#         LeafSystem.__init__(self)
        
#         self.DeclareContinuousState(3, 3, 0)                                  # 6 state variable.
#         self.DeclareVectorInputPort("u", BasicVector(2), self.CopyStateOut)   # 6 input
#         self.DeclareVectorOutputPort("x", BasicVector(6), self.CopyStateOut)  # 6 output.
#         self.length = 0.25  # length of rotor arm
#         self.mass = 0.486  # mass of quadrotor
#         self.inertia = 0.00383  # moment of inertia
#         self.gravity = 9.81  # gravity

#     # xdot(t) = f(x,u,t)
#     def DoCalcTimeDerivatives(self, context, derivatives):
#         # x = context.get_continuous_state_vector().GetAtIndex(0)
#         # xdot = -x + x**3
#         # derivatives.get_mutable_vector().SetAtIndex(0, xdot)
#         x = context.get_continuous_state_vector().CopyToVector()
#         u = self.EvalVectorInput(context, 0).CopyToVector()
#         q = x[:3]
#         qdot = x[3:]
#         qddot = np.array([
#             -np.sin(q[2]) / self.mass * (u[0] + u[1]),
#             np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
#             self.length / self.inertia * (u[0] - u[1])
#         ])
#         derivatives.get_mutable_vector().SetFromVector(
#             np.concatenate((qdot, qddot)))

#     # y = x
#     def CopyStateOut(self, context, output):
#         x = context.get_continuous_state_vector().CopyToVector()
#         output.SetFromVector(x)
        
from pydrake.examples import quadrotor
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.examples.pendulum import PendulumPlant
from pydrake.examples.quadrotor import QuadrotorPlant
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import GenerateHtml
from pydrake.systems.primitives import LogOutput
from pydrake.math import *
Quadrotor2D = Quadrotor2D_[None]  # Default instantiation
N = 200
# max_dt = 0.5
# max_tf = N * max_dt
plant = QuadrotorPlant()
print(plant)
context = plant.CreateDefaultContext()
print(context)
# params = context.get_numeric_parameter(0)

dircol = DirectCollocation(plant,
                           context,
                           num_time_samples=N,
                           minimum_timestep=0.01,
                           maximum_timestep=0.05,
                           assume_non_continuous_states_are_fixed=True)

dircol.AddEqualTimeIntervalsConstraints()
thrust_limit = 10  # N*m.
u = dircol.input()
print(u)
x0 = dircol.initial_state()
xf = dircol.final_state()
dircol.AddConstraintToAllKnotPoints(-thrust_limit <= u[0])
dircol.AddConstraintToAllKnotPoints(u[0] <= thrust_limit)
dircol.AddConstraintToAllKnotPoints(-thrust_limit <= u[1])
dircol.AddConstraintToAllKnotPoints(u[1] <= thrust_limit)
# dircol.AddConstraintToAllKnotPoints()
dircol.AddBoundingBoxConstraint(-2., -2., x0[0])
dircol.AddBoundingBoxConstraint(-1, -1., x0[1])
dircol.AddBoundingBoxConstraint(-np.pi/3, -np.pi/3, x0[2])

dircol.AddBoundingBoxConstraint(0, 0, xf[0])
dircol.AddBoundingBoxConstraint(0, 0, xf[1])
dircol.AddBoundingBoxConstraint(0, 0, xf[2])

result = Solve(dircol)
assert result.is_success()

x_trajectory = dircol.ReconstructStateTrajectory(result)
u_trajectory = dircol.ReconstructInputTrajectory(result)
x_knots = np.hstack([
    x_trajectory.value(t) for t in np.linspace(x_trajectory.start_time(),
                                               x_trajectory.end_time(), N)
])
u_knots = np.hstack([
    u_trajectory.value(t) for t in np.linspace(u_trajectory.start_time(),
                                               u_trajectory.end_time(), N)
])

time = np.linspace(x_trajectory.start_time(),x_trajectory.end_time(), N)
# print(x_knots)
plt.subplot(311)
plt.plot(time, x_knots[0, :], 'r')
plt.plot(time, x_knots[1, :], 'g')
plt.plot(time, x_knots[2, :], 'b')
plt.subplot(312)
plt.plot(time, x_knots[3, :], 'r')
plt.plot(time, x_knots[4, :], 'g')
plt.plot(time, x_knots[5, :], 'b')
plt.subplot(313)
plt.plot(time, u_knots[0, :], 'c')
plt.plot(time, u_knots[1, :], 'm')
# plt.plot(time, x_knots[0, :])
# # Plot the energy contours.
# nq = 151
# nqd = 151
# mgl = params.mass() * params.gravity() * params.length()
# q = np.linspace(-0.5, 0.5, nq)
# qd = np.linspace(-.5, 2, nqd)
# Q, QD = np.meshgrid(q, qd)
# Energy = .5 * params.mass() * params.length()**2 * QD**2 + mgl * np.cos(Q)
# plt.contour(Q,
#            QD,
#            Energy,
#            alpha=0.5,
#            linestyles="dashed",
#            colors="black",
#            linewidths=0.5)

# plt.set_xlabel("theta")
# plt.set_ylabel("thetadot")
# plt.axis([-0.5, 0.5, 0, 2])
# plt.set_title("Limit Cycle of the Rimless Wheel (w/ contours of "
#              "constant energy)")
plt.show()            

# builder = DiagramBuilder()

# # First add the pendulum.
# quadrotor = builder.AddSystem(QuadrotorPlant())
# quadrotor.set_name("quadrotor")

# controller1 = builder.AddSystem(PidController(kp=[10.], ki=[10.], kd=[10.]))
# controller1.set_name("controller1")


# controller2 = builder.AddSystem(PidController(kp=[10.], ki=[10.], kd=[10.]))
# controller2.set_name("controller2")


# # Now "wire up" the controller to the plant.
# builder.Connect(quadrotor.get_state_output_port(), controller1.get_input_port_estimated_state())
# builder.Connect(controller1.get_output_port_control(), quadrotor.get_input_port())
# builder.Connect(quadrotor.get_state_output_port(), controller2.get_input_port_estimated_state())
# builder.Connect(controller2.get_output_port_control(), quadrotor.get_input_port())

# # Make the desired_state input of the controller an input to the diagram.
# builder.ExportInput(controller1.get_input_port_desired_state())
# builder.ExportInput(controller2.get_input_port_desired_state())

# # Log the state of the pendulum.
# logger = LogOutput(quadrotor.get_state_output_port(), builder)
# logger.set_name("logger")

# diagram = builder.Build()
# diagram.set_name("diagram")

# html_str = GenerateHtml(diagram)
# f = open("Diagram.html","w")
# f.write(html_str)
# f.close()

# import webbrowser
# new = 2 # open in a new tab, if possible
# webbrowser.open_new_tab('Diagram.html')

# import matplotlib.pyplot as plt

# # Set up a simulator to run this diagram.
# simulator = Simulator(diagram)
# context = simulator.get_mutable_context()

# # We'll try to regulate the pendulum to a particular angle.
# desired_angle = 0.

# # First we extract the subsystem context for the pendulum.
# pendulum_context = diagram.GetMutableSubsystemContext(quadrotor, context)
# # Then we can set the pendulum state, which is (theta, thetadot).
# pendulum_context.get_mutable_continuous_state_vector().SetFromVector([-2., 0.2, 0.,
#                                                                       0., 0., 0.])

# # The diagram has a single input port (port index 0), which is the desired_state.
# context.FixInputPort(0, [0]*6)
# context.FixInputPort(1, [0]*6)

# # Reset the logger only because we've written this notebook with the opportunity to 
# # simulate multiple times (in this cell) using the same logger object.  This is 
# # often not needed.
# logger.reset()

# # Simulate for 10 seconds.
# simulator.AdvanceTo(20)

# # Plot the results.
# t = logger.sample_times()
# plt.figure()
# plt.title('PID Control of the Quadrotor')
# # Plot theta.
# plt.subplot(211)
# plt.plot([t[0], t[-1]], [desired_angle, desired_angle], 'g' )
# plt.xlabel('time (seconds)')
# plt.plot(t, logger.data()[2,:],'r.-')
# plt.ylabel('theta (rad)')
# # Plot theta_dot
# plt.subplot(212)
# plt.plot(t, logger.data()[5,:],'b.-')
# plt.plot([t[0], t[-1]], [0, 0], 'g' )
# plt.xlabel('time (seconds)')
# plt.ylabel('theta_dot (rad)')
# # Draw a line for the desired angle.
# plt.show()
