# Project PPO_Reacher Continuous Control

### Project Details

This is the second project of the Udacity Deep Reinforcement Learning nanodegree, dealing with continuous control.
For this project, the Reacher environment was solved using the PPO algorithm with distributed training over 20 parallel agents.

The essence of this problem is to solve the movement of a double jointed arm, so that its hand stays in a moving target location. For each timestep where the hand is in the target location, a reward of +0.1 is received. Else a reward of +0.0 is received.

The main specifications for each agent are: \
State size: 33 (informations about position, rotation, velocity and angular velocities) \
Action size: 4 (Torque value for each join) \ 
Each action range: [-1,1] 

The environment is considered solved when the average score accross all 20 agents reaches 30 for 100 episodes in a row.

![](images/Reacher_g1.gif)

### Getting started

To run this code, Python 3.6 is required, along with the dependencies found in [requirements.txt](requirements.txt).
Creating a virtual environment with those specifications is recommended.

You will also need to download the unity environnment compressed file from one of the following links, and extract it under the `Project_PPO_Reacher_Continuous_Control/` folder :

- Linux : [click here][(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX : [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit) : [click here][(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit) : [click here][(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Instructions

#### To train the agent from scratch

Launch *PPO_agent.py* from the command window.

#### To watch a trained agent

First, in the *PPO_agent.py* file, make sure that the path to the UnityEnvironment is correctly mapped to *Reacher.exe*
To watch a train agent, we launch the same main script *PPO_agent.py" and we add a system argument to the filepath to the saved weights.
E.g.: Launch *PPO_agent.py [filepath to trained weights]* from the command window.
