import numpy as np
import random
import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete
from scipy.stats import norm

class SequentialInferenceEnv(gym.Env):
    """
    Sequential Inference Task:
    In each trial, the environment first randomly sets the correct answer (target A or target B).
    The agent can take one of three actions:
        0: choose A (predict target = A)
        1: choose B (predict target = B)
        2: sample a stimulus from the current target distribution

    When the agent chooses action=2 ("sample"), the environment draws a one-hot stimulus
    according to the current trial's sampling distribution and returns it as the observation.
    A sampling cost and urgency cost are applied for each sample action.

    When the agent chooses action=0 or action=1, the environment treats this as a decision:
      • action=0 is interpreted as "predict target A"
      • action=1 is interpreted as "predict target B"
    If the agent's decision matches the correct_answer, reward = 1.0; otherwise, reward = 0.0.
    Then the environment advances to the next trial (or ends the episode if all trials are done).

    Each episode consists of num_trials trials. In each trial, the agent may sample up to max_samples times.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, num_trials=1, max_samples=np.inf, num_stimuli=8, seed=None, stimuli_logLR=None, reward=1.0, sampling_cost=0.01, urgency_cost=0.001, max_steps=1000):
        super(SequentialInferenceEnv, self).__init__()

        self.num_trials = num_trials
        self.max_samples = max_samples
        self.num_stimuli = num_stimuli
        self.stimuli_logLR = stimuli_logLR
        self.reward = reward
        self.sampling_cost = sampling_cost  # Cost applied each time agent samples
        self.urgency_cost = urgency_cost # slope of urgency cost
        self.max_steps = max_steps  # Maximum steps before truncation

        self.pA = None   
        self.pB = None  
        self.current_trial = 0
        self.stimuli_probs = None
        self.step_count = 0  # Track current step count

        self.set_random_seed(seed)

        # Action space: 0 = choose A (predict A), 1 = choose B (predict B), 2 = sample
        self.action_space = Discrete(3)

        # Observation space: a single one-hot vector of length num_stimuli
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(self.num_stimuli,),
                                     dtype=np.float32)

    def _generate_episode_stimuli(self):
        """
        At the start of each episode, randomly generate (or use provided) a set of
        log likelihood ratios for the 8 stimuli. Then find skew-normal parameters
        so that the probability ratios match those logLRs, yielding pA and pB.
        """
        # If no stimuli_logLR was provided, generate num.stimuli/2 positive and num.stimuli/2 negative values
        if self.stimuli_logLR is None:
            logLR_pos = np.random.uniform(0.05, 0.95, size=self.num_stimuli // 2)
            logLR_pos.sort()
            logLR_neg = -logLR_pos[::-1]
            self.stimuli_logLR = np.concatenate([logLR_neg, logLR_pos])
        else:
            self.stimuli_logLR = np.array(self.stimuli_logLR, dtype=np.float64)

        # symmetric Gaussian parameters
        mu = 2.2
        sigma = 5.0
        x_grid = np.linspace(-16, 16, 1200 + 1)

        # calculate the left and right Gaussian PDFs
        pdf_left = norm.pdf(x_grid, -mu, sigma)
        pdf_right = norm.pdf(x_grid, mu, sigma)
        log_ratio = np.log10(pdf_right / pdf_left)

        # interpolate the x values based on stimuli_logLR
        x_targets = np.interp(self.stimuli_logLR, log_ratio, x_grid)

        # corresponding probabilities (use the right PDF as pB, the left PDF as pA)
        pB_raw = norm.pdf(x_targets, mu, sigma)
        pA_raw = norm.pdf(x_targets, -mu, sigma)

        # normalize the probabilities
        pB = pB_raw / pB_raw.sum()
        pA = pA_raw / pA_raw.sum()

        self.pA = pA.astype(np.float64)
        self.pB = pB.astype(np.float64)

    def reset_trial(self):
        """
        Start a new trial:
          1. Randomly set correct_answer ∈ {A, B}.
          2. Choose the sampling distribution: pA if correct_answer=A, else pB.
          3. Clear any record of previously sampled stimuli in this trial.
          4. Initialize sample_count=0.
          5. Return an all-zero one-hot vector as the initial observation (dummy observation).
        """
        self.correct_answer = np.random.randint(0, 2)
        self.stimuli_probs = self.pA if (self.correct_answer == 0) else self.pB

        # Record of all sampled indices this trial
        self.trial_stimuli_idx = []
        # Record of all sampled one-hot vectors (for debugging / info)
        self.trial_stimuli = []
        self.sample_count = 0

        # Return dummy observation (all zeros)
        obs = np.zeros(self.num_stimuli, dtype=np.float32)

        return obs

    def reset(self, *, seed=None, options=None):
        """
        Reset the entire episode:
          1. Optionally reset the random seed.
          2. Set current_trial to 0.
          3. Re-generate the episode-level distributions (pA, pB).
          4. Initialize the first trial.
        Returns (obs, info).
        """
        if seed is not None:
            self.set_random_seed(seed)
        
        self.current_trial = 1
        self.step_count = 0  # Reset step counter
        self._generate_episode_stimuli()

        init_obs = self.reset_trial()
        info = {
            'trial': self.current_trial,
            'correct_answer': self.correct_answer,
            'mask': self.get_action_mask(),
            'stimuli_so_far': list(self.trial_stimuli_idx),
            'current_stimulus': -1,  # No stimulus sampled yet
            'reward': 0.0
        }
        return init_obs, info

    def step(self, action):
        """
        Take one step in the environment.
        If action == 2 (sample):
          • If sample_count < max_samples, draw an index according to stimuli_probs,
            create a one-hot vector, record it, and return (obs=one_hot, reward=-sampling_cost-urgency_cost, done=False).
          • If sample_count >= max_samples, return (obs=all zeros, reward=-sampling_cost-urgency_cost, done=False).

        If action == 0 (choose A ≡ predict A) or action == 1 (choose B ≡ predict B):
          • Compute reward = 1 if prediction matches correct_answer, else 0.
          • Apply urgency cost based on number of samples taken.
          • Advance to next trial (or end episode if current_trial == num_trials-1).
          • Return the next trial's initial observation (all zeros) or an all-zero obs if done.

        Returns: obs, reward, done, truncated, info
        """
        # Increment step counter
        self.step_count += 1
        terminated = False 
        truncated  = (self.step_count > self.max_steps)
        done = terminated or truncated

        if truncated:
            obs = np.zeros(self.num_stimuli, dtype=np.float32)
            reward = 0.0
            info = {
                'trial': self.current_trial,
                'correct_answer': self.correct_answer,
                'mask': self.get_action_mask(),
                'stimuli_so_far': list(self.trial_stimuli_idx),
                'current_stimulus': self.trial_stimuli_idx[-1] if self.trial_stimuli_idx else -1,
                'reward': reward
            }
            return obs, reward, True, True, info
        
        # --- Case 1: sample action ---
        if action == 2:
            # Calculate urgency cost based on current sample count
            current_urgency_cost = self.urgency_cost * self.sample_count
            
            if self.sample_count >= self.max_samples:
                # Cannot sample more; return zero observation
                obs = np.zeros(self.num_stimuli, dtype=np.float32)
                # reward = - current_urgency_cost  # only urgency costs
                reward = -self.sampling_cost
                info = {
                    'trial': self.current_trial,
                    'correct_answer': self.correct_answer,
                    'mask': self.get_action_mask(),
                    'stimuli_so_far': list(self.trial_stimuli_idx),
                    'current_stimulus': self.trial_stimuli_idx[-1] if self.trial_stimuli_idx else -1,
                    'reward': reward
                }
                return obs, reward, done, truncated, info
            
            else:
                # Perform a random draw from current distribution
                idx = np.random.choice(self.num_stimuli, p=self.stimuli_probs)
                obs = np.zeros(self.num_stimuli, dtype=np.float32)
                obs[int(idx)] = 1.0

                # Record this sample
                self.trial_stimuli_idx.append(int(idx))
                self.trial_stimuli.append(obs.copy())
                self.sample_count += 1

                reward = -self.sampling_cost - current_urgency_cost  # Apply both sampling and urgency costs
                info = {
                    'trial': self.current_trial,
                    'correct_answer': self.correct_answer,
                    'mask': self.get_action_mask(),
                    'stimuli_so_far': list(self.trial_stimuli_idx),
                    'current_stimulus': int(idx),
                    'reward': reward
                }
                return obs, reward, done, truncated, info

        # --- Case 2: decision action (choose A or B) ---
        elif action in [0, 1]:
            # Map action → predicted target: action=0 → predict A (0); action=1 → predict B (1)
            predicted = action

            reward = self.reward if (predicted == self.correct_answer) else 0.0

            # Move to next trial or end episode
            if self.current_trial >= self.num_trials:
                done = True
                obs = np.zeros(self.num_stimuli, dtype=np.float32)
            else:
                self.current_trial += 1
                done = False
                obs = self.reset_trial()
            
            info = {
                'trial': self.current_trial,
                'correct_answer': self.correct_answer,
                'mask': self.get_action_mask(),
                'stimuli_so_far': list(self.trial_stimuli_idx),
                'current_stimulus': self.trial_stimuli_idx[-1] if self.trial_stimuli_idx else -1,
                'reward': reward
            }

            return obs, reward, done, truncated, info

        else:
            raise ValueError(f"Invalid action {action}. Must be 0 (choose A), 1 (choose B), or 2 (sample).")

    def get_action_mask(self):
        """
        Return a boolean mask of which actions are valid.
        Action 2 (sample) is disabled when:
        - sample_count >= max_samples, or
        - step_count >= max_steps
        Actions 0 and 1 (decisions) remain available to force a choice.
        """
        mask = np.ones((self.action_space.n,), dtype=bool)
        if self.step_count >= self.max_steps:
            mask[2] = False
        return mask

    def set_random_seed(self, seed):
        """
        Set the random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def render(self, mode='human'):
        """
        Simple render: print current trial status, correct answer, and samples so far.
        """
        print(f"Trial {self.current_trial + 1}/{self.num_trials}, "
              f"Correct Answer: {self.correct_answer}, "
              f"Samples: {self.trial_stimuli_idx}")
    
    def one_hot_coding(self, num_classes, labels = None):
            """
            One-hot code nodes.
            """
            if labels is None:
                labels_one_hot = np.zeros((num_classes,))
            else:
                labels_one_hot = np.eye(num_classes)[labels]

            return labels_one_hot

    def close(self):
        pass

    
class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        self.init_prev_variables()

        new_observation_shape = (
            self.env.observation_space.shape[0] +
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        obs_wrapped = self.wrap_obs(obs)

        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        self.init_prev_variables()

        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs,
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward
        ])
        return obs_wrapped

if __name__ == "__main__":
    
    env = SequentialInferenceEnv(num_trials=1, max_samples=20, seed=None, max_steps=1000)
    env = MetaLearningWrapper(env)
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial observation:", obs)
    total_reward = 0
    done = False
    truncated = False
    
    while not done and not truncated:

        # Use action mask to select valid actions
        mask = env.env.get_action_mask()
        valid_actions = np.where(mask)[0]
        action = np.random.choice(valid_actions)

        action_names = {0: "Choose A", 1: "Choose B", 2: "Sample"}
        print(f"Step {env.env.step_count} (Trial {env.env.current_trial}):")
        print(f"  Action: {action} ({action_names[action]})")
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Action Mask: {info['mask']} (A, B, Sample)")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Done: {done}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")
        print()
    
    if truncated:
        print("Episode finished due to truncation (max steps reached).")
    else:
        print("Episode finished naturally.")
    print(f"Final total reward: {total_reward:.4f}")
    print(f"Total steps taken: {env.env.step_count}")



