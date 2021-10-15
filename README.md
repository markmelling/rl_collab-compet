[//]: # (Image References)

# Collaboration and Competition

### Introduction

This project uses the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment to demonstrate two agents collaborating.

![Trained Agent](images/tennis_trained.gif)

In the environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Solving the Environment

This project uses the two agent Tennis Unity environment to show a solution using the Twin Delayed Deep Deterministic (TD3) algorithm (https://arxiv.org/abs/1802.09477v3)

The solutions successfully achieves a score of > 0.5 on average for 100 episodes.

The DDPG algorithm was also tried, but this failed to reach the target score given the number of episodes used in training. From previous experiments DDPG does appear to take much longer to train, which may be the reason why a solution was not achieved.

## Getting Started

### Installing the dependencies
To run the code either from the command line or a Jupyter Notebook you need to make sure you have the Unity environment, Python and the required libraries installed.

#### Installing Python
The easiest way to install Python, or the correct version of Python, is to use conda, and the easiest way to install conda is to install miniconda.

If you don't have conda installed then follow the instructions here: [miniconda](https://docs.conda.io/en/latest/miniconda.html)

With conda installed, create and activate a conda environment which uses python 3.6 by entering the following commands in a terminal window

- For Linux or Mac
```shell
conda create --name drlnd python=3.6
source activate drlnd
```

- For Windows
```shell
conda create --name drlnd python=3.6
activate drlnd
```

#### Install the required python libraries
Clone this repository if you've not done it already, then navigate to the python subdirectory and install the dependencies by entering the following commands in a terminal window

```shell
git clone https://github.com/markmelling/rl_collab-compet
cd rl_collab-compet
pip install .

```
#### Create an IPython kernel if using a Jupyter notebook
python -m ipykernel install --user --name drlnd --display-name "drlnd"


#### Install the pre-prepared Unity environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in rl_collab-compet and unzip (or decompress) the file. 


## Instructions

You can train and evaluate the agents either from the notebook `Tennis.ipynb` or from the command line (I've not tried it on Windows)

For the notebook follow the instructions in `Tennis.ipynb`.

For the command line to train an agent you can run:
```
python train.py -m train -a 'td3 -n 'The_name_of_your_choosing'
```

The best model weights are stored in a file with the name (-n) with a .pth extension. This will be updated each time the best score is improved
The results of intermidiate evaluations are stored in a csv file of the same base name with a .csv extension

To run an evaluation test from the command line:

```

python train.py -m test -a 'td3 -n 'The_name_of_your_choosing'
```

This will return the average score for 100 episodes


### Pre-trained models
Pre-trained model weights that solve the environment have been saved for both agents
- `Tennis_TD3_Trained-0.pth`
- `Tennis_TD3_Trained-1.pth`

## Source code
Outside of the `Tennis.ipynb` notebook there are the following python files:
- run_agent.py (can be either run from the command line or from a notebook)
All other source code files are in the lib folder

- environments.py - provides a wrapper around a unity environment
- ddpg_agent.py - DDPG_Agent class implements the DDPG algorithm
- td3_agent.py - TD3_Agent class implements the TD3 algorithm
- model.py 
  - Implementations of a Deterministic Actor Critic Neural Network 
  - Implementations of a Neural Network supporting TD3
- replay_buffer.py - experience replay buffer
- utils - various useful functions and noise classes
