import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path

class Quad2DEnv_trajopt(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }

    def __init__(self, dt=0.02):
        self.gravity = 10.
        self.m = 0.5
        self.l = 0.5 
        self.I = self.m*(self.l**2)/12
        self.thrust_max = 15.0
        self.dt = dt  # seconds between state updates
        self.kinematics_integrator = 'euler'

        self.x_threshold = 2.5
        self.y_threshold = 2.5
        self.O_threshold = np.pi

        self.goal = np.array([0.0 ,0.0, 0.0])

        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.y_threshold * 2,
                         np.finfo(np.float32).max,
                         self.O_threshold * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        ahigh = np.array([self.thrust_max,
                          np.finfo(np.float32).max,
                          self.thrust_max,
                          np.finfo(np.float32).max],
                         dtype=np.float32)
        self.action_space = spaces.Box(-ahigh, ahigh, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, X):

        x, y, theta, x_dot, y_dot, theta_dot = self.state
        # theta = (((theta+np.pi) % (2*np.pi))) - np.pi
        x = X[0]
        y = X[1]
        theta = X[2]
        x_dot = X[3]
        y_dot = X[4]
        theta_dot = X[5]
        u1 = X[6]
        u2 = X[7]
        # dt = self.dt
        # m = self.m
        # l = self.l
        # I = self.I
        # g = self.gravity
        # # u1 = act[0]
        # # u2 = act[1]
        # costheta = np.cos(theta)
        # sintheta = np.sin(theta)
        # theta = np.arctan2(sintheta, costheta)

        # x_ddot = -((u1 + u2)*sintheta)/m
        # y_ddot = (((u1 + u2)*costheta) - (m*g))/m
        # theta_ddot = ((u1 - u2)*l/2)/I

        # theta_dot = theta_dot + theta_ddot*dt
        # y_dot = y_dot + y_ddot*dt
        # x_dot = x_dot + x_ddot*dt

        # theta = theta + theta_dot * dt
        # theta = np.arctan2(sintheta, costheta)
        # y = y + y_dot * dt
        # x = x + x_dot * dt

        self.state = (x, y, theta, x_dot, y_dot, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or y < -self.y_threshold
            or y > self.y_threshold
        )

        done2 = bool(
            abs(x) < 0.01
            and abs(y) < 0.01
            and abs(theta) < 0.01
            and abs(x_dot) < 0.01
            and abs(y_dot) < 0.01
            and abs(theta_dot) < 0.01
        )

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        if done2:
            done = True

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [np.random.uniform(low=-self.x_threshold/1.01, high=self.x_threshold/1.01),
                      np.random.uniform(low=-self.y_threshold/1.01, high=self.y_threshold/1.01),
                      np.random.uniform(low=-self.O_threshold, high=self.O_threshold),
                      0,0,0]
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        x, y, theta, x_dot, y_dot, theta_dot = self.state
        # theta = (((theta+np.pi) % (2*np.pi))) - np.pi
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        theta = np.arctan2(sintheta, costheta)

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = y * scale + screen_height/ 2.0 
        cartx = x * scale + screen_width / 2.0
        polewidth = 10.0
        polelen = scale * (2 * self.l)
        fan_width = 10.0
        fan_height = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -polelen / 2, polelen / 2, polewidth/2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.pole_trans = rendering.Transform()
            pole.add_attr(self.pole_trans)
            self.viewer.add_geom(pole)
            l, r, t, b = -fan_width / 2, fan_width / 2, fan_height - fan_width / 2, -fan_width / 2
            axle1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            axle1.set_color(.5, .5, .8)
            self.axle1_trans = rendering.Transform()
            axle1.add_attr(self.axle1_trans)
            self.viewer.add_geom(axle1)

            axle2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            axle2.set_color(.5, .5, .8)
            self.axle2_trans = rendering.Transform()
            axle2.add_attr(self.axle2_trans)
            self.viewer.add_geom(axle2)

            trackx = rendering.Line((0, - polelen/4), (0, polelen/4))
            trackx.set_color(255, 0, 0)
            self.trackx_trans = rendering.Transform()
            trackx.add_attr(self.trackx_trans)
            self.viewer.add_geom(trackx)

            tracky = rendering.Line((- polelen/4, 0), (polelen/4, 0))
            tracky.set_color(255, 0, 0)
            self.tracky_trans = rendering.Transform()
            tracky.add_attr(self.tracky_trans)
            self.viewer.add_geom(tracky)

            goal = rendering.Line((-polelen/4 + screen_width/2, screen_height/2), (polelen/4 + screen_width/2, screen_height/2))
            goal.set_color(0, 255, 0)
            self.viewer.add_geom(goal)

            goal = rendering.Line((screen_width/2, screen_height/2 -polelen/4), (screen_width/2, screen_height/2 + polelen/4))
            goal.set_color(0, 255, 0)
            self.viewer.add_geom(goal)

            self._pole_geom = pole

        if self.state is None:
            return None

        self.pole_trans.set_translation(cartx,
                                        carty)
        self.pole_trans.set_rotation(theta)

        self.trackx_trans.set_translation(cartx,
                                          carty)
        self.tracky_trans.set_translation(cartx,
                                          carty)
        self.axle1_trans.set_translation(cartx + (polelen/2*np.cos(theta)),
                                         carty + (polelen/2*np.sin(theta)))
        self.axle1_trans.set_rotation(theta)
        self.axle2_trans.set_translation(cartx - (polelen/2*np.cos(theta)),
                                        carty - (polelen/2*np.sin(theta)))
        self.axle2_trans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
