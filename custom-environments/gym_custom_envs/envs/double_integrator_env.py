import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
from os import path

class DoubIntEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }

    def __init__(self):
        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 1.0
        # self.h = 1.  # amplitude of x vibration
        # self.w = 1.  # frequency of x vibration
        # self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = (self.masspole * self.length)
        # self.force_max = 10.0
        # self.torque_max = 10.0
        # self.dt = 0.005  # seconds between state updates
        # self.kinematics_integrator = 'euler'
        #
        # # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        #
        # # Angle limit set to 2 * theta_threshold_radians so failing observation
        # # is still within bounds.
        # high = np.array([self.x_threshold * 2,
        #                  np.finfo(np.float32).max,
        #                  self.theta_threshold_radians * 2,
        #                  np.finfo(np.float32).max],
        #                 dtype=np.float32)
        # ahigh = np.array([self.torque_max], dtype=np.float32)
        # self.action_space = spaces.Box(-ahigh, ahigh, dtype=np.float32)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.dt = 0.01
        self.map_limit = 2
        self.dense_parameter = 20
        mapq, mapdq = np.meshgrid(np.linspace(-self.map_limit, self.map_limit, self.dense_parameter)
                                 , np.linspace(-self.map_limit, self.map_limit, self.dense_parameter))
        # self.map = np.array([mapq, mapdq])
        self.observation_space_low = np.array([-self.map_limit, -self.map_limit])
        self.observation_space_high = np.array([self.map_limit, self.map_limit])
        self.seed()
        self.viewer = None
        self.state = None
        self.ep_limit = 1000
        self.umin = -1.
        self.umax = 1.
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_discrete(self, s):
        dis_state = (((s - self.observation_space_low)
                     /(self.observation_space_high - self.observation_space_low))
                    *self.dense_parameter)
        # print(s, dis_state.astype(np.int))
        return dis_state.astype(np.int)

    def step(self, act):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        # if act == 0:
        #     u = -1.0 
        q, q_dot = self.state
        dt = self.dt
        u = act
        q_dot = q_dot + dt*u
        q = q + dt*q_dot
        # print('q,qdot', [q, q_dot], 'discrete', self.get_discrete(np.array([q, q_dot])))
        self.last_u = u
        self.state = (np.array([q, q_dot]))

        done = False
        # condition = np.isclose([q, q_dot], [0 , 0])
        if (abs(q) <= 0.1) and (abs(q_dot) <= 0.1):
            cost = 10.0
            done = True
        else:
            cost = -abs(q) - abs(q_dot)
            self.ep_count +=1

        if self.ep_count >= self.ep_limit:
            done = True

        if abs(q) >= (self.map_limit - 0.1):
            cost = -abs(q) - abs(q_dot)
            done = True

        return np.array(self.state), cost, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        states = [np.array([self.map_limit - 0.1, 0])
                  , np.array([-self.map_limit + 0.1, 0])
                  # ,np.array([self.map_limit, -self.map_limit]),
                  # ,np.array([-self.map_limit, -self.map_limit])
                  ]
        self.state = (random.choice(states))
        self.ep_count = 0
        # self.state = np.array([self.map_limit, 0])
        # print(states, self.state)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.map_limit * 2
        scale = screen_width/world_width
        carty = 200  # TOP OF CART
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

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self.goal = rendering.Line((screen_width/2, 0), (screen_width, screen_height))
            self.goal.set_color(0, 0, 255)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
