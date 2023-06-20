# Project PPO_Reacher Continuous Control

### Project Details

This is the second project of the Udacity Deep Reinforcement Learning nanodegree, dealing with continuous control.
For this project, the Reacher environment was solved using the PPO algorithm with distributed training over 20 parallel agents.

The essence of this problem is to solve the movement of a double jointed arm, so that its hand moves into a target location. For each timestep where the hand is in the target location, a reward of +0.1 is received. Else a reward of +0.0 is received.

The environment is considered solved when the average score accross all 20 agents reaches 30 for 100 episodes in a row.

Other specifications for each agent:
State size : 33
Action size : 4
Each action range : [-1,1]

 
### Getting started

To run this code, Python 3.6 is required, along with the dependencies found in [requirements.txt](https://github.com/Matlal033/Project_DDQN_Banana_Navigation/edit/main/requirements.txt).
Creating a virtual environment with those specifications is recommended.

download

### Instructions

#### To train the agent from scratch

Launch *PPO_agent.py* from the command window.

#### To watch a trained agent

To watch a train agent, we can provide as first system argument the filepath to the saved weights while lunching the main script.
Launch *PPO_agent.py [fullpath to trained weights]* from the command window.
