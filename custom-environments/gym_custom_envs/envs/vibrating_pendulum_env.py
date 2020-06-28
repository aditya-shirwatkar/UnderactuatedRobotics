import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path

class VibPenEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 1.0
        self.h = 1.  # amplitude of x vibration
        self.w = 1.  # frequency of x vibration
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_max = 10.0
        self.torque_max = 10.0
        self.dt = 0.005  # seconds between state updates
        self.kinematics_integrator = 'euler'

        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        ahigh = np.array([self.torque_max], dtype=np.float32)
        self.action_space = spaces.Box(-ahigh, ahigh, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):

        x, x_dot, theta, theta_dot = self.state
        dt = self.dt
        w = self.w
        t = act[0]
        m = self.masspole
        l = self.length
        g = self.gravity
        torque = act[1]
        self.last_u = torque
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        h = self.h
        sinwt = np.sin(w*t)
        x = h*sinwt
        sm = act[2]
        theta_ddot = (torque/(m*(l**2))) + (((w**2)*x/l)*costheta)+((-g/l)*sintheta)
        theta_ddot += (1 - theta_dot)*2 + sm*2
        theta_dot = theta_dot + theta_ddot*dt
        theta = theta + theta_dot * dt

        self.state = (x, x_dot, theta, theta_dot)

        done = False

        reward = 0.0 # you will need to change this if you want to train RL agents
        
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.steps_beyond_done = None
        self.state = np.array([0, 0, 0, 0])
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 200  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            
            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2]+np.pi)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
