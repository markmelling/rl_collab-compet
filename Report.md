[image1]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/ddpg_learning_rate.png
[image2]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/td3_learning_rate.png 
[image3]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/ddpg_test_scores.png
[image4]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/td3_test_scores.png

# Reinforcement Learning - Collaboration and Competition

## Introduction

This project uses the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment to demonstrate two agents collaborating.


## Solving the Environment

This project uses the two agent Tennis Unity environment to show a solution using the Twin Delayed Deep Deterministic (TD3) algorithm (https://arxiv.org/abs/1802.09477v3)

The solutions successfully achieves a score of > 0.5 on average for 100 episodes.

The DDPG algorithm was also tried, but this failed to reach the target score given the number of episodes used in training. From previous experiments DDPG does appear to take much longer to train, which may be the reason why a solution was not achieved.

## Implementation

The code to train or test the implemented model can either be run from the command line or from a Jupyter notebook (see `Tennis.ipynb`)

`run_agent.py` in the root of this repo is used to test or evaluate the model.

All other source files are in the lib folder

- `environments.py` - provides a wrapper around a unity environment
- `ddpg_agent.py` - DDPG_Agent class implements the DDPG algorithm
- `td3_agent.py` - TD3_Agent class implements the TD3 algorithm
- `model.py` 
  - Implementations of a Deterministic Actor Critic Neural Network 
  - Implementations of a Neural Network supporting TD3
- `replay_buffer.py` - experience replay buffer
- `utils.py` - various useful functions and noise classes


The agents are implemented as classes (DDPG_Agent and TD3_Agent), they both inherit from BaseAgent which just supports saving and loading the models.

The agent is responsible for creating the appropriate neural network (defined in model.py)

The agents provide a simple interface:

- act(state) 
This returns an action (which when training will include some noise)

- step(self, state, action, reward, next_state, done)
Adds experience to the replay buffer and then samples the replay buffer to learn from experiences

- learn(experiences, gamma)
Implements the relevent algorithm for updating the neural network and performing a soft update on the target network. 

## Learning Algorithms

There are two learning algorithms implemented, Deep Deterministic Policy Gradient and Twin Delayed Deep Deterministic

Initially DDPG was attempted, but this failed to 'solve' the environment in the length of training carried out.

### Deep Deterministic Policy Gradient
Deep Deterministic Policy Gradient is a model-free, off-policy algorithm for learning continuous actions. It combines Deterministic Policy Gradient and Deep Q-Network. 

DDPG uses an experience replay buffer along with a target network, to stabilize learning.

DDPG uses an Actor-Critic algorithm where a value based function and policy based function are merged.
The basic idea is to split the model in two: one for computing an action based on a state and another one to produce the Q values of the action.


- the actor inputs the state and determines the best action
- the critic evaluates the state and action (from the actor) by computing the Q-value 
     
Gradient ascent (not descent) is used to maximise the Q-value and update the weights.

For DDPG it also uses a target Actor-Critic network to add stability to the training. The target network's weights are gradually updated from the network (see the `soft_update method` of the agent class).

An experience replay buffer (`ReplayBuffer`) is used to learn from previous experiences. These samples are randomly sampled and used to learn from.

An Ornstein-Uhlenbeck process is used for generating noise to implement better exploration by the Actor network.


#### Hyper-pararmeters

- Replay buffer size: 1e6
- Replay batch (sample) size:  128 
- Discount factor (gamma): 0.99
- Soft update rate (tau): 5e-3
- layer initialization: orthogonal,  weight scale: 1e-3 
- Optimizer: Adam, learning_rate: 1e-3

#### Neural network architecture
- Actor network has 2 hidden layers with 400 and 300 units respectively
- Actor learning rate: 1e-3
- Critic network has 2 hidden layers with 400 and 300 units respectively
- Critic learning rate: 1e-3

 
### Twin Delayed Deep Deterministic 
TD3 builds on DDPG, like DDPG it is an model-free, off-policy algorithm that supports continuous action spaces. DDPG can over estimate Q-values which leads to the policy breaking, to tackle this TD3 introduces three improvements:
- TD3 uses 2 Q-learning networks and in calculating the Bellman Optimality Equation it takes the minimium of these two networks `target = rewards + (gamma * (1 - dones) * torch.min(q_1, q_2))`
- The policy (and target network) are updated less frequently (I followed the recommended one policy update for two Q function updates)
- Adds noise to the target action to make it harder to exploit Q-function errors

A gaussian process is used for generating noise.

#### Hyper-pararmeters

- Replay buffer size: 1e6
- Replay batch (sample) size:  128 
- Discount factor (gamma): 0.99
- Soft update rate (tau): 5e-3
- layer initialization: orthogonal,  weight scale: 1e-3 
- Optimizer: Adam, learning_rate: 1e-3

- noise: 0.2,
- noise clipping: 0.5
- delay in updating the policy and target network: 2

#### Neural network architecture
- Actor network has 2 hidden layers with 400 and 300 units respectively
- Actor learning rate: 1e-3
- Critic network has 2 hidden layers with 400 and 300 units respectively
- Critic learning rate: 1e-3


## Plot of rewards

### TD3

Using the pre-trained model weights the TD3 agent produced an average score of 2.35 over 100 episodes

An average score of 0.5+ is achieved after 496 episodes


#### Plot of rewards during training
![Learning rate](images/tennis_td3_learning_rate.png)

#### Episode scores over 100 test episodes
![test scores](images/tennis_td3_test_scores.png)

#### Model weights
The model weights for a TD3 based solutions that scores 0.5+ is stored in `Tennis_TD3_Trained-0.pth` and `Tennis_TD3_Trained-1.pth` for each agent.


### Future work
The length of time that it takes to train a model is considerable on my current setup and really is a barrier to testing and experimenting. I need to investigate both improved local versions (faster computer) and 'in the cloud' options, both in terms of the reduction in time for training and cost.

Other future work worth considering:
#### Additional algorithms
Add support for other well know algorithms and compare their performance (e.g. PPO and A2C)

#### Replay buffer
Experiment with prioritised experience replay buffer.


## Glossary 

#### Model-free
A model-free algorithm does not use a model of the environment. That is it doesn't use a function which predicts state transitions or rewards.
Q-learning is an example of a model-free algorithm.
#### Off-policy (from Richard Sutton's book)
In Q-learning, the agent learns an optimal policy with the help of a greedy policy and behaves using policies of other agents. Q-learning is called off-policy because the updated policy is different from the behaviour policy. In other words, it estimates the reward for future actions and appends a value to the new state without actually following any greedy policy.
#### Experience Replay
As experiences (state, action, reward, next state) with the environment happen they are stored and then subsequently sampled to learn from.
