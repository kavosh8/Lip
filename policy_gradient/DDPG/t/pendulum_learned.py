import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import keras
import sys, numpy
class NewPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,model_name):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self.model_name=model_name
        self.model = self.create_model()

    def create_model(self):
        state_input=keras.layers.Input(shape=(3,))
        action_input=keras.layers.Input(shape=(1,))
        added = keras.layers.Concatenate()([state_input, action_input])
        added=keras.layers.Dense(32,activation='relu')(added)
        added=keras.layers.Dense(32,activation='relu')(added)
        #added=keras.layers.Dense(32,activation='relu')(added)
        next_state=keras.layers.Dense(3)(added)
        model = keras.models.Model(inputs=[state_input, action_input], outputs=next_state)
        model.compile(optimizer='adam',
              loss='mse')
        #cwd = os.getcwd()
        return model    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        #print(self.state)
        #print("obs",self._get_obs())
        #print("obs",self.observe)
        th, thdot = self.state # th := theta
        g = 10.
        m = 1.
        l = 1.
        dt = self.dt
        unclipped_u=u
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)


        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        self.state = np.array([newth, newthdot])
        #print(self.state)
        #print("next actual obs",self._get_obs())#****
        #print(u)
        #sys.exit(1)
        self.model.load_weights("t/"+self.model_name+".h5")
        next_observe=self.model.predict(x=[numpy.array(self.observe).reshape(1,3),numpy.array(unclipped_u).reshape(1,1)])
        #print("next predicted obs",next_observe[0])#****
        self.observe=next_observe[0]
        #print(numpy.linalg.norm(self._get_obs()-self.observe))
        #sys.exit(1)


        return self.observe, -costs, False, {}

    def _reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.model.load_weights("t/"+self.model_name+".h5")
        self.observe=self._get_obs()
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)