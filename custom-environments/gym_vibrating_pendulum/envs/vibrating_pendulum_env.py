# import gym
# from gym import spaces
# from gym.utils import seeding
# import numpy as np
# from os import path
#
# class VibPenEnv(gym.Env):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 30
#     }
#
#     def __init__(self, m=1., g=-9.81, l=1., h=0.5, w=2):
#         self.max_speed = 8
#         self.max_torque = 2.
#         self.force = 0.2
#         self.dt = .05
#         self.g = g
#         self.m = m
#         self.l = l
#         self.w = w
#         self.pmax = h
#         self.viewer = None
#         self.pmax_speed = h*w
#
#         high = np.array([1., 1., self.max_speed, self.pmax, self.pmax_speed], dtype=np.float32)
#         self.action_space = spaces.Box(
#             low=-self.max_torque,
#             high=self.max_torque, shape=(1,),
#             dtype=np.float32
#         )
#         self.vibrator_space = spaces.Box(
#             low=-self.force,
#             high=self.force, shape=(1,),
#             dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=-high,
#             high=high,
#             dtype=np.float32
#         )
#
#         self.seed()
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def step(self, u):
#         th, thdot, p, pdot = self.state  # th := theta
#
#         g = self.g
#         m = self.m
#         l = self.l
#         h = self.pmax
#         w = self.w
#         dt = self.dt
#
#         u = np.clip(u, -self.max_torque, self.max_torque)[0]
#         # f = np.clip(f, -self.force, self.force)[0]
#         self.last_u = u  # for rendering
#         # self.last_f = f  # for rendering
#         costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
#
#         newpdot = pdot + dt*(-(w**2)*p)
#         newp = p + newpdot*dt
#         # newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
#         newthdot = thdot + ((u/(m*(l**2))) +
#                             (((w**2)*p/l)*np.cos(th))
#                             +((-g/l)*np.sin(th)))*dt
#         newth = th + newthdot * dt
#         newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
#
#
#         self.state = np.array([newth, newthdot, newp, newpdot])
#         return self._get_obs(), -costs, False, {}
#
#     def reset(self):
#         high = np.array([np.pi, 1, self.pmax, self.pmax_speed])
#         # self.state = self.np_random.uniform(low=-high, high=high)
#         self.last_u = None
#         self.state = np.array([np.pi, 0,0,self.pmax_speed])
#         return self._get_obs()
#
#     def _get_obs(self):
#         theta, thetadot, p, pdot = self.state
#         return np.array([np.cos(theta) + p, np.sin(theta), thetadot, p, pdot])
#
#     def render(self, mode='human'):
#         if self.viewer is None:
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.Viewer(500, 500)
#             self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
#             rod = rendering.make_capsule(1, .2)
#             rod.set_color(.8, .3, .3)
#             self.pole_transform = rendering.Transform()
#             rod.add_attr(self.pole_transform)
#             self.viewer.add_geom(rod)
#             axle = rendering.make_circle(.05)
#             axle.set_color(0, 0, 0)
#             self.axle_transform = rendering.Transform()
#             axle.add_attr(self.axle_transform)
#             self.viewer.add_geom(axle)
#             fname = path.join(path.dirname(__file__), "assets/clockwise.png")
#             self.img = rendering.Image(fname, 1., 1.)
#             self.imgtrans = rendering.Transform()
#             self.img.add_attr(self.imgtrans)
#
#         self.viewer.add_onetime(self.img)
#         self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
#         self.pole_transform.set_translation(self.state[2], 0)
#         self.axle_transform.set_translation(self.state[2], 0)
#         self.imgtrans.set_translation(self.state[2], 0)
#
#         if self.last_u:
#             self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
#
#         return self.viewer.render(return_rgb_array=mode == 'rgb_array')
#
#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None
#
#
# def angle_normalize(x):
#     return (((x+np.pi) % (2*np.pi)) - np.pi)

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

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
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
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

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
        # print(act[1], (-g/l)*sintheta, act[2], ((w**2)*x/l)*costheta,theta_ddot)
        # print((act[1]/(m*(l**2))) + ((-g/l)*sintheta), (act[2]/(m*(l**2))) + (((w**2)*x/l)*costheta))

        theta = theta + theta_dot * dt

        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        # temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # if self.kinematics_integrator == 'euler':
        #     x = x + self.tau * x_dot
        #     x_dot = x_dot + self.tau * xacc
        #     theta = theta + self.tau * theta_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        # else:  # semi-implicit euler
        #     x_dot = x_dot + self.tau * xacc
        #     x = x + self.tau * x_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        #     theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # done = bool(
        #     x < -self.x_threshold
        #     or x > self.x_threshold
        #     or theta < -self.theta_threshold_radians
        #     or theta > self.theta_threshold_radians
        # )

        done = False

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            # if self.steps_beyond_done == 0:
            #     logger.warn(
            #         "You are calling 'step()' even though this "
            #         "environment has already returned done = True. You "
            #         "should always call 'reset()' once you receive 'done = "
            #         "True' -- any further steps are undefined behavior."
            #     )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
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
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            # self.img = rendering.Image(fname, 1., 1.)
            # self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

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
        # self.imgtrans.set_translation(cartx, carty)
        #
        # if self.last_u:
        #     self.imgtrans.scale = (-2*self.last_u, np.abs(2*self.last_u))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None