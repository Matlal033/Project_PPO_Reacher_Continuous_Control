# Project Report

### Learning Algorithm

The PPO algorithm was used to solve this project.
It is an on-policy algorithm, from the family of policy gradient methods, that is able to tackle complex environments, continuous action spaces and distributed training.

##### Hypermarameters :
    episode = 2000
    discount_rate = .99
    gae_lambda = 0.95
    surrogate_clip = 0.2
    surrogate_clip_decay = 0.999
    beta = 1e-2 #entropy coefficient
    beta_decay = 0.995
    tmax = 400000 #timesteps while collecting trajectories
    SGD_epoch = 10 
    LR = 3e-4
    adam_epsilon = 1e-5
    batch_size = 2000
    hidden_size = 1024
    gradient_clip = 5 
    rollout_size = 2000

##### Actor neural network structure :

| Layer | type | Input size | Output size | Activation |
|-------|------|------------|-------------|------------|
|1 | Fully Connected | 33 (state size) | 1024 | ReLU |
|2  | Fully Connected | 1024 | 1024 | ReLU |
|3  | Fully Connected | 1024 | 4 (action size) | tanh |

##### Value neural network structure :

| Layer | type | Input size | Output size | Activation |
|-------|------|------------|-------------|------------|
|1 | Fully Connected | 33 (state size) | 1024 | ReLU |
|2  | Fully Connected | 1024 | 1024 | ReLU |
|3  | Fully Connected | 1024 | 1 | None |

### Plot of rewards


### Ideas for future work

