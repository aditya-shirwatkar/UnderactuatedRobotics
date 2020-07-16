import numpy as np
import matplotlib.pyplot as plt
# from IPython import get_ipython

# from underactuated.jupyter import AdvanceToAndVisualize, SetupMatplotlibBackend
# plt_is_interactive = SetupMatplotlibBackend()

from pydrake.all import (ConstantVectorSource, DiagramBuilder,
                         PlanarSceneGraphVisualizer, SceneGraph, SignalLogger,
                         Simulator)
from pydrake.examples.compass_gait import (CompassGait, CompassGaitGeometry,
                                           CompassGaitParams)

builder = DiagramBuilder()
compass_gait = builder.AddSystem(CompassGait())

hip_torque = builder.AddSystem(ConstantVectorSource([0.0]))
builder.Connect(hip_torque.get_output_port(0), compass_gait.get_input_port(0))

scene_graph = builder.AddSystem(SceneGraph())
CompassGaitGeometry.AddToBuilder(
    builder, compass_gait.get_floating_base_state_output_port(), scene_graph)
# visualizer = builder.AddSystem(
#     PlanarSceneGraphVisualizer(scene_graph, xlim=[-1., 5.], ylim=[-1., 2.], 
#                                show=plt_is_interactive))
# builder.Connect(scene_graph.get_pose_bundle_output_port(),
#                 visualizer.get_input_port(0))

logger = builder.AddSystem(SignalLogger(14))
builder.Connect(compass_gait.get_output_port(1), logger.get_input_port(0))

diagram = builder.Build()
simulator = Simulator(diagram)

context = simulator.get_mutable_context()
context.SetAccuracy(1e-4)
context.SetContinuousState([0., 0., 0.4, -2.])

duration = 8.0 # sets a shorter duration during testing
# AdvanceToAndVisualize(simulator, visualizer, duration)

plt.figure()
plt.plot(logger.data()[4, :])
plt.plot(logger.data()[11, :])
# plt.xlabel("left leg angle")
# plt.ylabel("left leg angular velocity")
plt.show()