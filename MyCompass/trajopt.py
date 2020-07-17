import copy
import numpy as np

from pydrake.common.containers import namedview
from pydrake.systems.framework import (BasicVector, LeafSystem, PublishEvent,
                                       PortDataType, UnrestrictedUpdateEvent,
                                       WitnessFunctionDirection)
from pydrake.systems.pyplot_visualizer import PyPlotVisualizer

BipedState = namedview(
    "BipedState", ["l_tip","q1", "q2", "q3", "q4", "q5", "r_tip", 
                "l_tipdot","q1dot", "q2dot", "q3dot", "q4dot", "q5dot", "r_tipdot"])

class MyCompassGait(LeafSystem):

    def __init__(self):
        LeafSystem.__init__(self)

        self.DeclareVectorInputPort("u", BasicVector(4))
        self.DeclareContinuousState(BasicVector(np.zeros(14)), 7, 7, 0)

        self.DeclareVectorOutputPort("state", BasicVector(10), self.CopyStateOut)

        # Parameters from Geyer05, p.23
        self.mass = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # kg
        self.length = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # m
        self.gravity = 9.81  # m/s^2
        self.inertia = self.mass * self.length
        
        self.left_impact_witness = self.MakeWitnessFunction(
            "impact", WitnessFunctionDirection.kPositiveThenNonPositive,
            self.left_foot_height, UnrestrictedUpdateEvent(self.switch)
        )
        self.right_impact_witness = self.MakeWitnessFunction(
            "impact", WitnessFunctionDirection.kPositiveThenNonPositive,
            self.right_foot_height, UnrestrictedUpdateEvent(self.switch)
        )

        # self.touchdown_witness = self.MakeWitnessFunction(
        #     "touchdown", WitnessFunctionDirection.kPositiveThenNonPositive,
        #     self.foot_height, UnrestrictedUpdateEvent(self.touchdown))
        # self.takeoff_witness = self.MakeWitnessFunction(
        #     "takeoff", WitnessFunctionDirection.kPositiveThenNonPositive,
        #     self.leg_compression, UnrestrictedUpdateEvent(self.takeoff))
        # self.apex_witness = self.MakeWitnessFunction(
        #     "apex", WitnessFunctionDirection.kPositiveThenNonPositive,
        #     self.apex, PublishEvent(self.publish_apex))
    
    def left_foot_height(self, context):
        s = BipedState(context.get_continuous_state_vector().CopyToVector())
        h1 = self.length[0]*np.sin(s.q1)
        h2 = self.length[1]*np.sin(s.q2) + h1
        h3 = self.length[2]*np.sin(s.q3) + h2
        h4 = self.length[3]*np.sin(s.q4) + h3
        h5 = self.length[4]*np.sin(s.q5) + h4
        return h5

    def right_foot_height(self, context):
        s = BipedState(context.get_continuous_state_vector().CopyToVector())
        h1 = self.length[0]*np.sin(s.q5)
        h2 = self.length[1]*np.sin(s.q4) + h1
        h3 = self.length[2]*np.sin(s.q3) + h2
        h4 = self.length[3]*np.sin(s.q2) + h3
        h5 = self.length[4]*np.sin(s.q1) + h4
        return h5

    # def touchdown(self, context, event, state):
    #     s = BipedState(
    #         context.get_mutable_continuous_state_vector().CopyToVector())

    #     # print("touchdown")

    #     # Update rdot and thetadot to match xdot and ydot, using
    #     # x = -r*sin(theta), z = r*cos(theta)
    #     #  => xdot = -rdot*s - r*c*thetadot, zdot = rdot*c - r*s*thetadot
    #     #  => xdot*c + zdot*s = -r*thetadot
    #     # r^2 = x^2 + z^2
    #     #  => 2r*rdot = 2x*xdot + 2z*zdot
    #     #  => rdot = -xdot*sin(theta) + zdot*cos(theta)
    #     # (matches Geyer05 Eq. 2.24 up to the symbol changes)
    #     s.r = self.r0
    #     s.rdot = -s.xdot * np.sin(s.theta) + s.zdot * np.cos(s.theta)
    #     s.thetadot = -(s.xdot * np.cos(s.theta) +
    #                    s.zdot * np.sin(s.theta)) / self.r0
    #     state.get_mutable_continuous_state().get_mutable_vector().SetFromVector(
    #         s[:])

    # def leg_compression(self, context):
    #     s = BipedState(context.get_continuous_state_vector().CopyToVector())
    #     return self.r0 - s.r

    # def takeoff(self, context, event, state):
    #     s = BipedState(
    #         context.get_mutable_continuous_state_vector().CopyToVector())

    #     # print("takeoff")

    #     # Setup flight state (these lines aren't strictly required, since we
    #     # choose to also integrate x and z in stance below).
    #     s.z = self.r0 * np.cos(s.theta)
    #     s.xdot = -s.rdot * np.sin(s.theta) - self.r0 * s.thetadot * np.cos(
    #         s.theta)
    #     s.zdot = s.rdot * np.cos(s.theta) - self.r0 * s.thetadot * np.sin(
    #         s.theta)

    #     # Update theta to commanded leg angle.
    #     s.theta = self.EvalVectorInput(context, 0).GetAtIndex(0)
    #     s.thetadot = 0
    #     s.r = self.r0
    #     s.rdot = 0

    #     state.get_mutable_continuous_state().get_mutable_vector().SetFromVector(
    #         s[:])

    # def apex(self, context):
    #     return context.get_continuous_state_vector().GetAtIndex(5)  # zdot

    def publish_apex(self, context, event):
        # TODO(russt): provide an option to terminate here instead, pending
        # resolution of #4447.
        # print("apex")
        if self.last_apex is None:
            s = BipedState(
                context.get_mutable_continuous_state_vector().CopyToVector())
            self.last_apex = s.z

    def apex_velocity_from_dimensionless_system_energy(self, Etilde, z):
        E = Etilde * self.mass * self.gravity * self.r0
        # E = 0.5*m*v^2 + m*g*z
        xdot = np.sqrt(2. / self.mass * (E - self.mass * self.gravity * z))
        return xdot

    def energy_flight(self, context):
        s = BipedState(
            context.get_mutable_continuous_state_vector().CopyToVector())
        return (0.5 * self.mass * (s.xdot**2 + s.zdot**2) +
                self.mass * self.gravity * s.z)

    def energy_stance(self, context):
        s = BipedState(
            context.get_mutable_continuous_state_vector().CopyToVector())
        return (0.5 * self.mass * (s.rdot**2 + s.r**2 * s.thetadot**2) +
                self.mass * self.gravity * s.r * np.cos(s.theta) +
                0.5 * self.stiffness * (self.r0 - s.r)**2)

    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        y = output.SetFromVector(x)

    def DoGetWitnessFunctions(self, context):
        return [self.touchdown_witness, self.takeoff_witness, self.apex_witness]

    def DoCalcTimeDerivatives(self, context, derivatives):
        s = BipedState(context.get_continuous_state_vector().CopyToVector())
        sdot = BipedState(np.zeros(10))
        sdot[0:5] = s[5:10]

        if (self.left_foot_height(context) < 0):
            # then we're in "stance"
            sdot.rdot = (self.stiffness / self.mass * (self.r0 - s.r) +
                         s.r * s.thetadot**2 - self.gravity * np.cos(s.theta))
            sdot.thetadot = (self.gravity / s.r * np.sin(s.theta) -
                             2 * s.rdot * s.thetadot / s.r)

            # Integrate x and z also, just for the sake of visualization (all
            # the integrated values except x will be overwritten in the
            # take-off reset).
            # x = -r*sin(theta), y = r*cos(theta) =>
            sdot.xdot = (-sdot.rdot * np.sin(s.theta) -
                         2 * s.rdot * s.thetadot * np.cos(s.theta) +
                         s.r * s.thetadot**2 * np.sin(s.theta) -
                         s.r * sdot.thetadot * np.cos(s.theta))
            sdot.zdot = (sdot.rdot * np.cos(s.theta) -
                         2 * s.rdot * s.thetadot * np.sin(s.theta) -
                         s.r * sdot.thetadot * np.sin(s.theta) -
                         s.r * s.thetadot**2 * np.cos(s.theta))

        else:
            sdot.xdot = 0
            sdot.zdot = -self.gravity
            sdot.rdot = 0
            sdot.thetadot = 0

        derivatives.get_mutable_vector().SetFromVector(sdot[:])
