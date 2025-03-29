# Reinforcement learning on Gymnasium

## Motivation

Experimenting with RL on Gymnasium is an ideal starting point to delve into the algorithms that underlie agent training. Gymnasium allows focus on the agent by taking care of the environment implementation. Here, I go through the Gymnasium framework to explore how best to use it to experiment with RL models.

## Environment

The process of RL on Gymnasium starts with setting up the environment which can be done either:

- Initializing existing environment on the library
- Sub-classing an type of environment and modifying it for your purposes

Once initialized, `obs, info = env.reset()` will yield the first observation from the environment.

### Wrappers

These allow us to modify an existing environment without having to alter the underlying code directly. The following code sets a **time limit** on the Markov Decision Process (MDP) using the `TimeLimit` wrapper and exemplifies how wrappers are to be used:

```python
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
env = TimeLimit(env, max_episode_steps=100)
```

## Agent

The agent acts on the environment using `env.step(action)`. This function returns a tuple of length 5 containing:

- **observation** (*ObsType*) – “State” in the RL problem.
    
    In the sample `MiniGrid` environment, the default value of `obs` is a dictionary with a **Partially Observable Grid** from the perspective of the agent. By default this yields a `7 x 7` grid (`agent_view_size = 7`)where each **pixel** in the grid has:
    
    - `object type` (e.g., `2=wall`, `8=agent`, `5=goal`).
    - `color` (encoded as an integer).
    - `state` (usually `0`, not commonly used).
    
    However, using a wrapper we can get a picture of the entire grid instead.
    
- **reward** (*SupportsFloat*) – The reward as a result of taking the action.
- **terminated** (*bool*) – Whether the agent reaches the terminal state (as defined under the MDP of the task).
- **truncated** (*bool*) – Whether the truncation condition outside the scope of the MDP is satisfied such as a TimeLimit.
- **info** (*dict*)
    
    Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). This might, for instance, contain: metrics that describe the agent’s performance state, variables that are hidden from observations, or individual reward terms that are combined to produce the total reward. In OpenAI Gym <v26, it contains “TimeLimit.truncated” to distinguish truncation and termination, however this is deprecated in favour of returning terminated and truncated variables.
    

Then the path taken by the agent can be rendered either as:

1. A series of frames set up as follows:
    
    ```python
    fig, axes= plt.subplots(1, len(frames), figsize=(len(frames), 5))
    ```
    
2. An animation using `matplotlib.animation` 

## Training the agent

We use **`Stable-Baselines3`** for the basic implementation of the agent. The module provides a set of reliable implementations of reinforcement learning algorithms in PyTorch. Here is a useful website to help select an model for your RL problem: https://github.com/bennylp/RL-Taxonomy

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms as follows:

```python
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# To evaluate the model
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    frames.append(env.render())
```

The following are the arguments that can be passed into the model to initialize it: 

- **policy** (*ActorCriticPolicy*) – The policy model to use (MlpPolicy, CnnPolicy, …)
- **env** (*Env | [VecEnv](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#stable_baselines3.common.vec_env.VecEnv) | str*) – The environment to learn from (if registered in Gym, can be str)
- **n_steps** (*int*) – The number of steps to run for each environment per update

There are a lot more arguments which can be found out at:  https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO