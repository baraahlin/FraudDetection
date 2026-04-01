# with mfo
#RL (grouping) → MFO (optimize features inside groups) → classifier
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

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

    # =========================
    # 🔥 NEW REWARD FUNCTION
    # =========================
    def _compute_reward(self):

        selected_features = []
        total_redundancy = 0
        active_groups = 0

        for g in range(self.num_groups):
            idx = np.where(self.assignments == g)[0]

            if len(idx) == 0:
                continue

            active_groups += 1
            selected_features.extend(idx)

            # redundancy (correlation)
            if len(idx) > 1:
                subset = self.X[:, idx]
                corr = np.corrcoef(subset, rowvar=False)
                redundancy = np.nanmean(np.abs(corr))
                total_redundancy += redundancy

        if len(selected_features) == 0:
            return -10

        selected_features = np.unique(selected_features)
        X_sub = self.X[:, selected_features]

        # quick model (fast)
        model = LogisticRegression(max_iter=200, solver='liblinear')

        #model.fit(X_sub, self.y)
        #y_pred = model.predict(X_sub)
        '''
        # sample for fast evaluation
        idx = np.random.choice(len(self.y), size=min(500, len(self.y)), replace=False)

        X_sample = X_sub[idx]
        y_sample = self.y[idx]
        '''
        # ensure consistent sampling
        n_samples = X_sub.shape[0]
        sample_size = min(500, n_samples)

        idx = np.random.choice(n_samples, size=sample_size, replace=False)

        X_sample = X_sub[idx, :]
        y_sample = self.y[idx]

        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_sample)
        

        # confusion matrix
        #tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
        tn, fp, fn, tp = confusion_matrix(y_sample, y_pred).ravel()

        # G-mean
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        gmean = np.sqrt(sensitivity * specificity)

        # penalties
        redundancy_penalty = total_redundancy / (active_groups + 1e-6)
        complexity_penalty = len(selected_features) / self.num_features

        # final reward
        reward = (2.0 * gmean) - (0.5 * redundancy_penalty) - (0.2 * complexity_penalty)

        return reward
    

from stable_baselines3 import PPO

def run_rl_grouping(X, y):
    env = FeatureGroupingEnv(X, y)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.001,
        ent_coef=0.05  # encourage exploration
    )

    #model.learn(total_timesteps=3000)
    model.learn(total_timesteps=1500)

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    return env.assignments.astype(int)