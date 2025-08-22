import os
import argparse
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from net import *
from env import *
# from oldenv import *
from a2c import *


# note: vectorizing wrapper only works under this protection
if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    parser.add_argument('--jobid', type = str, default = 'reward=1.0_sample_cost=0.01_urgency_cost=0.0_logLR=[-0.9,0.9]_max_samples=10_max_steps=10000_epNum=1500000', help = 'job id')
    parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results'), help = 'path to store results')

    # nework parameters
    parser.add_argument('--hidden_size', type = int, default = 64, help = 'lstm hidden size')

    # environment parameters
    parser.add_argument('--num_trials', type = int, default = 1, help = 'number of trials per episode')
    parser.add_argument('--max_samples', type = int, default = 10, help = 'number of samples per trial')
    parser.add_argument('--max_steps', type = int, default = 10000, help = 'maximum steps per trial')
    parser.add_argument('--reward', type = float, default = 1.0, help = 'reward for correct answer')
    parser.add_argument('--sampling_cost', type = float, default = 0.01, help = 'sampling cost')
    parser.add_argument('--urgency_cost', type = float, default = 0.00, help = 'urgency cost')
    parser.add_argument('--num_stimuli', type = int, default = 8, help = 'number of stimuli')

    # training parameters
    parser.add_argument('--num_episodes', type = int, default = 1500000, help = 'training episodes')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch_size')
    parser.add_argument('--gamma', type = float, default = 1.0, help = 'temporal discount')
    parser.add_argument('--lamda', type = float, default = 1.0, help = 'generalized advantage estimation coefficient')
    parser.add_argument('--beta_v', type = float, default = 0.05, help = 'value loss coefficient')
    parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
    parser.add_argument('--max_grad_norm', type = float, default = 1.0, help = 'gradient clipping')

    args = parser.parse_args()

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # set environment
    seeds = [random.randint(0, 1000) for _ in range(args.batch_size)]
    
    num_batches = args.num_episodes // args.batch_size
    lr_schedule = two_phase_linear(
        total = num_batches,
        start = args.lr,
        final = 1e-5,
    )
    entropy_schedule = two_phase_linear(
        total = num_batches,
        start = args.beta_e,
        final = 1e-3,
    )

    env = gym.vector.SyncVectorEnv([
        lambda: 
        # MetaLearningWrapper(
            SequentialInferenceEnv(
                num_trials = args.num_trials,
                max_samples = args.max_samples,
                num_stimuli = args.num_stimuli,
                reward = args.reward,
                sampling_cost = args.sampling_cost,
                urgency_cost = args.urgency_cost,
                stimuli_logLR = np.array([-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9]),
                max_steps = args.max_steps,
                seed = seeds[i],
            )
            # )
        for i in range(args.batch_size)
    ])

    # set network
    net = SharedGRURecurrentActorCriticPolicy(
        feature_dim = env.single_observation_space.shape[0],
        action_dim = env.single_action_space.n,
        gru_hidden_dim = args.hidden_size,
    )

    # set model
    model = BatchMaskA2C(
        net = net,
        env = env,
        lr = args.lr,
        batch_size = args.batch_size,
        gamma = args.gamma,
        lamda = args.lamda,
        beta_v = args.beta_v,
        beta_e = args.beta_e,
        max_grad_norm = args.max_grad_norm,
        lr_schedule = lr_schedule,
        entropy_schedule = entropy_schedule,
    )

    # train network
    data = model.learn(
        num_episodes = args.num_episodes,
        print_frequency = 10
    )
    
    # save net and data
    model.save_net(os.path.join(exp_path, f'net.pth'))
    model.save_data(os.path.join(exp_path, f'data_training.p'))

    num_segments = 100
    N = len(data['loss'])  # or len(data['reward']) if same length
    segment_size = N // num_segments

    # average the episode reward and loss
    avg_episode_reward = [
        np.mean(data['episode_reward'][i*segment_size:(i+1)*segment_size])
        for i in range(num_segments)
    ]
    avg_loss = [
        np.mean(data['loss'][i*segment_size:(i+1)*segment_size])
        for i in range(num_segments)
    ]
    episode_percentage = np.linspace(0, 100, num_segments)

    # plot the episode reward
    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(episode_percentage, avg_episode_reward)
    plt.xlabel('Episodes Percentage')
    plt.ylabel('Average Episode Reward')
    plt.title('Training Reward (Averaged)')
    plt.savefig(os.path.join(exp_path, f'training_reward.png'))
    plt.show()
    plt.close()

    # plot the loss
    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(episode_percentage, avg_loss)
    plt.xlabel('Episodes Percentage')
    plt.ylabel('Average Loss')
    plt.title('Training Loss (Averaged)')
    plt.savefig(os.path.join(exp_path, f'training_loss.png'))
    plt.show()
    plt.close()