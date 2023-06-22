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

To run this code, Python 3.6 is required, along with the dependencies found in [requirements.txt](requirements.txt). \
Creating a virtual environment with those specifications is recommended.

You will also need to download the unity environnment compressed file from one of the following links, and extract it under the `Project_PPO_Reacher_Continuous_Control/` folder :

- Linux : [click here][(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX : [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit) : [click here][(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit) : [click here][(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Instructions

#### To train the agent from scratch

In *main.py* file, make sure the path to the UnityEnvironment points to Reacher.exe.
E.g.: `env = UnityEnvironment(file_name='Reacher_Windows_x86_64\Reacher.exe')`
Then, launch `main.py` from the command window.

### To train the agent from a previous checkpoint
In the command window, pass as a first argument the filepath to the checkpoint.
E.g.: `main.py "checkpoints\checkpoint_temp_actor_critdic.pth"`

#### To watch a trained agent

First, in the *watch_agent.py* file,  make sure the path to the UnityEnvironment points to Reacher.exe.
Then, from the command window, launch *watch_agent.py*  file the filepath to the checkpoint as the first argument.
E.g.: `watch_agent.py "checkpoints\checkpoint_temp_actor_critdic.pth"`
