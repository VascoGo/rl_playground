import gymnasium as gym
import numpy as np

class BEMS(gym.Env):
    def __init__(self):

        # 3 Actions: ([is_charging, is_using])
        self.action_space = gym.spaces.MultiBinary(2)

        # State (battery capacity, is_charging, is_using)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32), high=np.array([1.0,1.0,1.0], dtype=np.float32), dtype=np.float32)

        self._agent_state = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    def _get_obs(self):
        return self._agent_state

    def _get_info(self):
        return {
            "bettery_life": 1.0
        }

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self._agent_state = [np.random.random(), 0.0, 0.0]

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        
        return super().step(action)