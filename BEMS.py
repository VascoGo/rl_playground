import gymnasium as gym
import numpy as np

class BEMS(gym.Env):
    def __init__(self, decay_rate=0.03, charge_rate=0.05, task_queue=10):

        # Environment personalizable variables
        self._decay_rate = decay_rate
        self._charge_rate = charge_rate
        self._task_queue = task_queue

        # Environment constant variables
        self.REWARD_COMPLETE = 15.0
        self.PENALTY_DEADLINE = 12.0
        self.PENALTY_STEP = 0.02
        self.PENALTY_BATTERY_EMPTY = 12.0
        self.PENALTY_BATTERY_LOW = 0.2
        self.PENALTY_OVERCHARGE = 0.15
        self.PENALTY_CHARGE_USE = 0.3
        self.PENALTY_CHARGE_TOGGLE = 0.1

        # Actions
        self.action_space = gym.spaces.MultiDiscrete([2, self._task_queue + 1])

        # State
        self.observation_space = gym.spaces.Dict({
            "status": gym.spaces.Box(low=np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0, self._task_queue - 1, 20.0], dtype=np.float32), dtype=np.float32),
            "tasks": gym.spaces.Box(high=20, low=0, shape=(self._task_queue,), dtype=np.float32),
            "task_mask": gym.spaces.MultiBinary(self._task_queue)
        }
        )

        self._agent_state = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    def _get_obs(self):
        return self._agent_state

    def _get_info(self):
        return {
            "bettery_life": 1.0
        }

    def _get_reward(self, action):
        reward = - self.PENALTY_STEP

        current_status = self._agent_state["status"]
        soc = current_status[0]
        was_charging = current_status[1]
        is_using = current_status[2]
        active_slot = int(current_status[3])
        remaining_steps = current_status[4]

        if action[0] == 1.0 and was_charging == 0.0:
            reward -= self.PENALTY_CHARGE_TOGGLE

        if action[0] == 1.0 and is_using == 1.0:
            reward -= self.PENALTY_CHARGE_USE

        if action[0] == 1.0 and soc >= 1.0:
            reward -= self.PENALTY_OVERCHARGE

        if is_using == 1.0 and soc <= self._decay_rate:
            reward -= self.PENALTY_BATTERY_EMPTY
        elif soc < 0.10:
            reward -= self.PENALTY_BATTERY_LOW

        if is_using == 1.0 and remaining_steps <= 1.0 and soc > self._decay_rate:
            reward += self.REWARD_COMPLETE

        for i in range(self._task_queue):
            if self._agent_state["task_mask"][i] == 1:
                deadline = self._agent_state["tasks"][i]
                steps_needed = remaining_steps if (is_using == 1.0 and i == active_slot) else 20.0
                if deadline < steps_needed:
                    reward -= self.PENALTY_DEADLINE

        return reward

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        tasks = np.zeros(self._task_queue, dtype=np.float32)
        task_mask = np.zeros(self._task_queue, dtype=np.int8)
        for i in range(self._task_queue):
            prob = self.np_random.random()
            if prob > 0.5:
                tasks[i] = 200 * prob
                task_mask[i] = 1 

        self._agent_state = {
            "status": np.array([1.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            "tasks": tasks,
            "task_mask": task_mask
        }

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):

        reward = self._get_reward(action)

        truncated = False

        if action[0] == 1.0:
            self._agent_state["status"][1] = 1.0
            self._agent_state["status"][0] = self._agent_state["status"][0] + self._charge_rate
            if self._agent_state["status"][0] > 1.0:
                self._agent_state["status"][0] = 1.0
        else:
            self._agent_state["status"][1] = 0.0

        if self._agent_state["status"][2] == 1.0:
            self._agent_state["status"][0] = self._agent_state["status"][0] - self._decay_rate if  self._agent_state["status"][0] >= self._decay_rate else 0
            self._agent_state["status"][4] -= 1
            if self._agent_state["status"][4] == 0:
                finished_task = int(self._agent_state["status"][3]) 
                self._agent_state["status"][2] = 0
                self._agent_state["status"][3] = -1

                self._agent_state["tasks"][finished_task] = 0
                self._agent_state["task_mask"][finished_task] = 0

        for i in range(self._task_queue):
            if (self._agent_state["task_mask"][i] == 1):
                self._agent_state["tasks"][i] -= 1
                if self._agent_state["tasks"][i] == 0:
                    self._agent_state["task_mask"][i] = 0
                    if int(self._agent_state["status"][3]) == i:
                        self._agent_state["status"][2] = 0                        
                        self._agent_state["status"][3] = -1
                        self._agent_state["status"][4] = 0

        if action[1] != self._task_queue and self._agent_state["status"][2] == 0 and self._agent_state["status"][0] > 0.0:
            index = action[1]
            if self._agent_state["task_mask"][index] == 1:
                self._agent_state["status"][2] = 1.0
                self._agent_state["status"][3] = index
                self._agent_state["status"][4] = 20

        if (self._agent_state["status"][0] == 0.0 and self._agent_state["status"][2] == 1.0):
            self._agent_state["status"][2] = 0.0
            self._agent_state["status"][3] = -1.0
            self._agent_state["status"][4] = 0.0

        terminated = False

        if np.count_nonzero(self._agent_state["task_mask"]) == 0:
            terminated = True
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()