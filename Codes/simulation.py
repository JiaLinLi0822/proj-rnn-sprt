import torch
import numpy as np 
import pandas as pd
import gymnasium as gym
import os

# from environment import *
from env import *
from a2c import BatchMaskA2C
from net import *
from replaybuffer import *


if __name__ == '__main__':

    data = []

    env = gym.vector.SyncVectorEnv([
        lambda: 
        # MetaLearningWrapper(
            SequentialInferenceEnv(
            num_trials=1,
            max_samples=10000,
            max_steps=10000,
            reward = 1.0,
            sampling_cost=0.01,
            urgency_cost=0.00,
            num_stimuli=8,
            stimuli_logLR=np.array([-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9]),
            seed=20250805,
            )
        # )
        for i in range(1)
    ])

    path = os.getcwd() + '/results/exp_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1500000/'
    net = torch.load(path + 'net.pth')
    net.eval()  # Set network to evaluation mode

    # Run simulation with the vectorized environment
    num_episodes = 50000  # Number of episodes to simulate per environment in batch
    batch_size = 1  # Number of environments to simulate in parallel

    for episode in range(num_episodes):

        if episode % 1000 == 0:
            print(f'Simulating episode {episode}...')
        
        buffer = BatchReplayBuffer()
            
        # initialize a trial
        dones = np.zeros(batch_size, dtype = bool) # no reset once turned to 1
        mask = torch.ones(batch_size)
        states_hidden = None

        obs, info = env.reset()
        obs = torch.Tensor(obs)
        action_mask = torch.tensor(np.stack(info['mask'])) # (batch_size, action_dim), bool

        episode_data = []

        episode_data.append({
            'episode': episode,
            'correct_answer': info['correct_answer'][0],
            'stimuli_so_far': info['stimuli_so_far'][0],
            'current_stimulus': info['current_stimulus'][0],
        })

        # iterate through a trial
        while not all(dones):
            # step the net
            action, policy, log_prob, entropy, value, states_hidden = net(
                obs, states_hidden, action_mask,
            )
            value = value.view(-1) # (batch_size,)
            # step the env
            obs, reward, done, truncated, info = env.step(action)
            obs = torch.Tensor(obs) # (batch_size, feature_dim)
            reward = torch.Tensor(reward) # (batch_size,)
            action_mask = torch.tensor(np.stack(info['mask'])) # (batch_size, action_dim), bool

            # record the action and reward to the last step
            episode_data[-1]['action'] = action.item()
            episode_data[-1]['reward'] = round(reward.item(), 4)
            episode_data[-1]['policy'] = policy.tolist()[0]
            episode_data[-1]['hidden_state'] = states_hidden.detach().cpu().numpy().tolist()[0]

            if done == False:
                # Append trial data to episode_data
                episode_data.append({
                    'episode': episode,
                    'correct_answer': info['correct_answer'][0],
                    'stimuli_so_far': info['stimuli_so_far'][0],
                    'current_stimulus': info['current_stimulus'][0].tolist(),
                })

            # update mask and dones
            # note: the order of the following two lines is crucial
            dones = np.logical_or(dones, done)
            mask = (1 - torch.Tensor(dones)) # keep 0 once a batch is done

        # concatenate episode data
        data.extend(episode_data)

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_json(path + "data.json", orient="records", lines=True)
    print("done")