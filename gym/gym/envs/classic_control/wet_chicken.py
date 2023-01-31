import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class WetChickenEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.width = 5.0
        self.length = 5.0
        self.max_speed = 1.0

        self.action_space = spaces.Box(
            low=-self.max_speed,
            high=self.max_speed, shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=self.width, shape=(2,),
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        x_cord, y_cord = self.state
        reward = -(self.length-y_cord)
        drift = 3*x_cord*(1/self.width)
        turbulence = 3.5 - drift
        new_x_cord = np.clip(x_cord+ action[0], 0, self.width)
        new_y_cord = y_cord+action[1]-1+drift+turbulence*np.random.uniform(low=-1, high=1)
        if new_y_cord > self.length:
            new_y_cord = 0.0
            new_x_cord = 0.0
        if new_y_cord<0.0:
            new_y_cord = 0.0
        self.state = np.array([new_x_cord, new_y_cord])
        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        x_cord, y_cord = self.state 
        return np.array([x_cord, y_cord])

    def reset(self, state='random'):
        self.last_u = None
        if type(state) ==str:
            # unif across space
            self.state = np.random.uniform(low=0.0, high=5.0, size=2)
            # OG start state
            #self.state = np.random.uniform(low=0.0, high=1.0, size=2)
            return self.state
        else:
            self.state = state
            return state


    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
