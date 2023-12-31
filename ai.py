import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#architecture of the Neural Network

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        # to be able to use the tools of the module
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # full connections need to be defined. one for input layer and one for hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def  forward(self, state):
        # represents hidden neurons and relu is
        # rectifier function that activates hidden neurons       
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# implementin experience replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    #event is a tuple of 4 elements as mentioned
    def push(self, event):
        self.memory.append(event)
        # amking sure our memory has more than 
        # capacity
        if len(self.memory) > self.capacity:
            #we reomove the oldest transition
            del self.memory[0]

    def sample(self, batch_size):
        # we take samples randomly of fixed batch size
        # from the memory
        # zip(*list) simply reshapes the list.
        # forex if list = ((1,2),(3,4),(5,6))
        # zip(*list) = ((1,2,3),(4,5,6))
        samples = zip(*random.sample(self.memory, batch_size))
        #sample converted to torch variable
        # now each batch will be a pytorch variable
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

#implementing Deep Q Learning

class Dqn():

    def __init__(self,input_size, nb_action, gamma):
        self.gamma = gamma
        # reward mean of last 100
        # this is a sliding window
        self.reward_window = []
        # neural network
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # initializing a tensor class and initializing a fake dimension
        # along with it. 0 since python too starts with 0
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    # deciding which action to play and when?

    def select_action(self, state):
        # generating probabilities of the entities entered
        # the entities will be neural networks we're working
        # with
        probs = F.softmax(self.model(Variable(state, volatile = True))*75) # t is temp and is equal to 7
        # softmax{[1,2,3]} = [0.04,0.11,0.85] => softmax{[1,2,3]*3} = [0,0.02,0.98]
        # tempreture parameter is the way we tell which parameter we get to use
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]

    # training the model in forward and backward propagation
    # basically training the deep learning model

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Convert batch_action to a tensor of type int64
        batch_action = batch_action.type(torch.int64)
        
        # Compute the Q-values for the current state and selected actions
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        # Compute the Q-values for the next state
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        # Compute the target Q-values
        target = self.gamma * next_outputs + batch_reward
        
        # Calculate the loss using smooth L1 loss
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Zero the gradients, perform backpropagation, and update the model
        self.optimizer.zero_grad()

        # Calculate the loss using smooth L1 loss
        td_loss.backward(retain_graph = True)
        self.optimizer.step()

    # update function when the ai reaches a new state
    # also will integrate the action function to take 
    # a new action at every update
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        # starting all over again now
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            #make learning happen
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action    
        self.last_state = new_state
        self.last_reward = reward

        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    # to create a save the model
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain1.pth')
        
    def load(self):
        if os.path.isfile('last_brain1.pth'):
            print("=> loading checkpoint..")
            checkpoint = torch.load('last_brain1.pth')
            self.model.load_state_dict(checkpoint['state_dict'])   
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("no checkpoint found..")
        
       
                 

 


            

        
