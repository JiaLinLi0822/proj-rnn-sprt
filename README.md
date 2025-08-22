# Recurrent neural network learns early commitment as an optimal strategy for decision-making in naturalistic environments

This repository contains the implementation of a recurrent neural network (RNN) model for studying decision-making in sequential sampling tasks, specifically in the context of Sequential Probability Ratio Tests (SPRT). This work is submitted to a NeurIPS ogInterp workshop: Interpreting Cognition in Deep Learning Models.

## Overview

The project investigates how recurrent neural networks can learn optimal decision-making strategies in sequential inference tasks where agents must balance accuracy with sampling costs. The model uses an Actor-Critic reinforcement learning approach with GRU-based recurrent networks to learn when to sample more evidence versus when to make a decision.

## Key Features

- **Sequential Inference Environment**: A gymnasium-compatible environment simulating SPRT tasks
- **Recurrent Neural Networks**: GRU-based Actor-Critic architecture with action masking
- **Multiple Environment Configurations**: Three different cost structures representing different decision boundaries
- **Comprehensive Analysis**: Tools for analyzing decision weights, sampling patterns, and neural dynamics

## Project Structure

```
proj-rnn-sprt/
├── Codes/                      # Main source code
│   ├── training.py            # Main training script
│   ├── a2c.py                 # A2C trainer implementation
│   ├── env.py                 # Sequential inference environment
│   ├── net.py                 # Neural network architectures
│   ├── simulation.py          # Model evaluation and simulation
│   ├── utils.py               # Utility functions and schedulers
│   ├── replaybuffer.py        # Replay buffer for training
│   ├── analysis.ipynb         # Analysis notebook
│   └── submit.sh              # Job submission script
├── Results/                   # Training results and saved models
├── Figures/                   # Generated figures and plots
└── OriginalSPRT/             # Original SPRT analysis scripts
```

## Installation and Dependencies

### Required Python Packages

The project requires the following Python packages:

```bash
# Core ML/RL libraries
torch>=1.12.0
gymnasium>=0.26.0
numpy>=1.21.0

# Data analysis and visualization
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Additional utilities
pickle-mixin
tqdm
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd proj-rnn-sprt
```

2. Create a conda environment (recommended):
```bash
conda create -n deeprl python=3.8
conda activate deeprl
```

3. Install dependencies:
```bash
pip install torch gymnasium numpy pandas matplotlib seaborn scipy scikit-learn
```

## Training the Model

### Basic Training

To train a model with default parameters:

```bash
cd Codes
python training.py
```

### Advanced Training with Custom Parameters

The training script supports extensive customization through command-line arguments:

```bash
python training.py \
    --jobid "custom_experiment" \
    --path "../Results" \
    --hidden_size 64 \
    --num_trials 1 \
    --max_samples 10000 \
    --max_steps 10000 \
    --reward 1.0 \
    --sampling_cost 0.01 \
    --urgency_cost 0.00 \
    --num_episodes 1500000 \
    --lr 1e-3 \
    --batch_size 128 \
    --gamma 1.0 \
    --lamda 1.0 \
    --beta_v 0.05 \
    --beta_e 0.05 \
    --max_grad_norm 1.0
```

### Parameter Descriptions

#### Environment Parameters
- `--num_trials`: Number of trials per episode (default: 1)
- `--max_samples`: Maximum samples per trial (default: 10)
- `--max_steps`: Maximum steps before truncation (default: 10000)
- `--reward`: Reward for correct decisions (default: 1.0)
- `--sampling_cost`: Cost per sample (default: 0.01)
- `--urgency_cost`: Urgency cost coefficient (default: 0.00)
- `--num_stimuli`: Number of stimulus types (default: 8)

#### Network Parameters
- `--hidden_size`: GRU hidden layer size (default: 64)

#### Training Parameters
- `--num_episodes`: Total training episodes (default: 1500000)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size for parallel environments (default: 128)
- `--gamma`: Temporal discount factor (default: 1.0)
- `--lamda`: GAE lambda coefficient (default: 1.0)
- `--beta_v`: Value loss coefficient (default: 0.05)
- `--beta_e`: Entropy regularization coefficient (default: 0.05)
- `--max_grad_norm`: Gradient clipping threshold (default: 1.0)

### Batch Training with Different Configurations

Use the provided submission script to train multiple models in parallel:

```bash
cd Codes
./submit.sh
```

The script runs three predefined configurations:
1. **env1**: Infinite horizon (low sampling cost, high max_samples)
2. **env2**: Infinite horizon (high sampling cost, high max_samples)  
3. **env3**: Soft boundary (low sampling cost, low max_samples)

## Reproducing Results

### 1. Train Models

Run the batch training script to generate all three environment configurations:

```bash
cd Codes
chmod +x submit.sh
./submit.sh
```

This will create results in:
- `Results/exp_001_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1500000/`
- `Results/exp_002_reward=1.0_sample_cost=0.05_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10000_max_steps=10000_epNum=1500000/`
- `Results/exp_003_reward=1.0_sample_cost=0.01_urgency_cost=0.00_logLR=[-0.9,0.9]_max_samples=10_max_steps=10000_epNum=1500000/`

### 2. Generate Simulation Data

After training, run simulations to collect behavioral data:

```bash
cd Codes
python simulation.py
```

### 3. Analyze Results

Open and run the analysis notebook:

```bash
cd Codes
jupyter notebook analysis.ipynb
```

The notebook will generate all the figures showing in the article:
- Decision boundaries and sampling patterns
- Neural network decision weights
- Comparison with optimal SPRT behavior
- Psychometric curves and response time distributions

## Experimental Configurations

The project studies three main experimental environments:

### Environment 1: Infinite Horizon with low sampling cost
- **Sampling Cost**: 0.01 (low)
- **Max Samples**: 10000 (effectively unlimited)
- **Behavior**: Near-optimal SPRT with evidence accumulation

### Environment 2: Infinite Horizon with high sampling cost
- **Sampling Cost**: 0.05 (high)
- **Max Samples**: 10000 (effectively unlimited)
- **Behavior**: Early decision-making due to high sampling costs

### Environment 3: Time constraint with low sampling cost
- **Sampling Cost**: 0.01 (low)  
- **Max Samples**: 10 (limited)
- **Behavior**: Strategic sampling within constraints

## Outputs

### Training Outputs
Each training run produces:
- `net.pth`: Trained neural network model
- `data_training.p`: Training metrics (loss, reward, episode length)
- `training_reward.png`: Training reward curve
- `training_loss.png`: Training loss curve

### Simulation Outputs  
- `data.json`: Detailed behavioral data including decisions, sampling patterns, and (hidden states)neural activations

### Analysis Outputs
Generated figures include:
- Decision weight analysis
- Sampling probability curves
- Accuracy vs. evidence strength
- Response time distributions
- Neural dynamics visualization

## Model Architecture

The model uses a **SharedGRURecurrentActorCriticPolicy** with:
- **Input**: 8-dimensional one-hot stimulus vectors
- **Recurrent Layer**: GRU with 64 hidden units (configurable)
- **Actor Head**: Linear layer outputting action probabilities (Choose A, Choose B, Sample)
- **Critic Head**: Linear layer outputting state values
- **Action Masking**: Prevents invalid actions (e.g., sampling when limit reached)