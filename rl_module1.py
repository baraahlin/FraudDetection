import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FeatureGroupingEnv(gym.Env):
    def __init__(self, X, y, num_groups=3):
        super().__init__()

        self.X = X
        self.y = y
        self.num_features = X.shape[1]
        self.num_groups = num_groups

        self.current_idx = 0

        self.action_space = spaces.Discrete(num_groups)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        self.assignments = np.zeros(self.num_features)

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        self.assignments = np.random.randint(0, self.num_groups, size=self.num_features)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.current_idx / self.num_features,
            np.mean(self.assignments)
        ], dtype=np.float32)

    def step(self, action):
        self.assignments[self.current_idx] = action
        self.current_idx += 1

        done = self.current_idx >= self.num_features

        reward = 0
        if done:
            reward = self._compute_reward()

        return self._get_obs(), reward, done, False, {}

    def _compute_reward(self):
        # simple reward: encourage balanced grouping
        counts = np.bincount(self.assignments.astype(int))
        return -np.std(counts)
    


from stable_baselines3 import PPO

def run_rl_grouping(X, y):
    env = FeatureGroupingEnv(X, y)

    model = PPO("MlpPolicy", env, verbose=0)

    # ⚡ FAST training
    model.learn(total_timesteps=2000)

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    return env.assignments.astype(int)