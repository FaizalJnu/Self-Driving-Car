# AI Brain Project

This project implements a Deep Q-Learning (DQN) based AI brain using PyTorch. It's designed to learn and make decisions in an environment through reinforcement learning.

## Features

- Neural Network architecture for Q-value approximation
- Experience Replay for efficient learning
- Deep Q-Learning algorithm implementation
- Action selection using softmax distribution
- Model saving and loading capabilities

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- torch
- kivy

## Installation

1. Ensure you have Python 3.x installed.
2. Install the required packages:
   ```
   pip install numpy torch kivy
   ```

## Usage

1. Import the `Dqn` class from the main script:
   ```python
   from main import Dqn
   ```

2. Initialize the DQN agent:
   ```python
   input_size = <your_input_size>
   nb_action = <number_of_possible_actions>
   gamma = <discount_factor>
   
   agent = Dqn(input_size, nb_action, gamma)
   ```

3. Use the agent in your environment:
   ```python
   state = <current_state>
   action = agent.select_action(state)
   
   # After taking the action and observing the result:
   new_state = <new_state>
   reward = <observed_reward>
   
   next_action = agent.update(reward, new_state)
   ```

4. Save and load the model:
   ```python
   # To save
   agent.save()
   
   # To load
   agent.load()
   ```

## Components

### Network

The `Network` class defines the neural network architecture used for Q-value approximation. It consists of two fully connected layers.

### ReplayMemory

The `ReplayMemory` class implements experience replay, storing and sampling past experiences for batch learning.

### Dqn

The `Dqn` class ties everything together, implementing the Deep Q-Learning algorithm. It handles action selection, learning, and model updates.

## Customization

You can customize various aspects of the AI brain:

- Adjust the network architecture in the `Network` class
- Modify the learning rate in the `Dqn` class initialization
- Change the temperature parameter in the `select_action` method
- Adjust the replay memory size and batch size in the `Dqn` class
