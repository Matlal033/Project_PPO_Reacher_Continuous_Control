# Project Report

### Learning Algorithm

The **PPO algorithm** was used to solve this project.
It is an on-policy algorithm, from the family of policy gradient methods, that is able to tackle complex environments, continuous action spaces and distributed training.

One of the key elements characterizing the PPO algorithm is the surrogate clipped objective. \
To train the neural net, we compare the probabilities of taking a given action according to a previous policy, versus the probabilities of taking that same action with the actual policy by dividing them together. \
Then, we clip the ratio of probabilities within a range close to one (e.g. [0.8 to 1,2]), and we multiply it by the advantages of doing that action in that state space, thus obtaining the clipped objective (The advantages are the delta of rewards by taking a given action instead of the average action). \
Clipping the ratio decreases the steps while updating the policy and makes it more stable. \
That clipped objective (clipped ratio * advantages) is then compared to the unclipped objective (ratio * advantages), and the smallest objective value is the one used. The reason behind that choice is to stay more conservative with the policy updates.

Since the goal is to maximize the expected returns, and that a loss function tries to minimize a value, we use the negative value of the objective during training.
Entropy is also included has a regularization term in the loss function, thereby increasing the stochasticity of the policy, helping with exploration, and potentially helping with faster convergence to a good policy.

#### Hypermarameters :
|Hyperparameter|Value|
|--------------|-----|
|episode | 2000 |
|discount_rate | .99 |
|tau | 0.95 |
|surrogate_clip | 0.2 |
|surrogate_clip_decay | 1 |
|beta | 1e-2 |
|beta_decay | 1 |
|n_epoch | 4 |
|learning_rate | 1e-4 |
|adam_epsilon | 3e-4 |
|batch_size | 500 |
|hidden_size | 128 |
|gradient_clip | 5 |
|rollout_size | 500 |

#### Neural networks

Without sharing weights, both the actor and critic neural networks have the same structure for the 3 first layers, with state size as input, and 128 as all hidden sizes. \
The actor outputs 4 values for the actions as probability distributions with a learnable standard deviation. \
the critic outputs a single number that estimates the value of a given state. \
An adam optimizer is used to train the neural networks.

##### Actor neural network structure :

| Layer | type | Input size | Output size | Activation |
|-------|------|------------|-------------|------------|
|1 | Fully Connected | 33 (state size) | 128 | ReLU |
|2  | Fully Connected | 128 | 128 | ReLU |
|3 | Fully Connected | 128 | 128 | ReLU |
|4  | Fully Connected | 128 | 4 (action size) | tanh |

##### Value neural network structure :

| Layer | type | Input size | Output size | Activation |
|-------|------|------------|-------------|------------|
|1 | Fully Connected | 33 (state size) | 128 | ReLU |
|2  | Fully Connected | 128 | 128 | ReLU |
|3  | Fully Connected | 128 | 128 | ReLU |
|4 | Fully Connected | 1024 | 1 | None |

##### Other parameters
`self.std = nn.Parameter(torch.ones(1, action_size))`

### Plot of rewards

In **121** episodes, the game was solved by achieving an average score of at least +30 over the next 100 episodes.

![](images/Reacher_plot1.png)

```
Episode  217  avg score:  34.563499227445575
Episode  218  avg score:  33.79699924457818
Episode  219  avg score:  34.288499233592304
Episode  220  avg score:  34.00399923995137
Episode  221  avg score:  34.84549922114238

Environment solved in 121 episodes!     Average Score: 30.166
```

![](images/Reacher_g1.gif)

### Ideas for future work

Since the learning was extremely sensible to the hyperparameters, and that multiple plateaus below 30 points were reached in previous experiments, it would be interesting to further investigate the impact of entropy at different stages of training, to better encourage exploration when agent is not learning anymore. \
Changing the structure of the neural net also had a major impact. Trying out larger and deeper hidden layers would be an interesting area to investigate further. 
