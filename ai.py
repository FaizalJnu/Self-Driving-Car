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
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) # t is temp and is equal to 7
        # softmax{[1,2,3]} = [0.04,0.11,0.85] => softmax{[1,2,3]*3} = [0,0.02,0.98]
        # tempreture parameter is the way we tell which parameter we get to use
        action = probs.multinomial()
        return action.data[0,0]

    # training the model in forward and backward propagation
    # basically training the deep learning model

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # creating and killing the fake action using unqueeze and squeeze
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        # temporal difference loss
        # also using hoober loss
        # smooth_ll_loss is the hoober loss and 
        # is the best loss predictor in DL
        td_loss = F.smooth_l1_loss(outputs, target)
        # reinitialize the optimizer at each iteration
        # of the loop
        self.optimizer.zero_grad()
        #backpropagating
        td_loss.backward(retain_variables = True)
        self.optimizer.step()

    # update function when the ai reaches a new state
    # also will integrate the action function to take 
    # a new action at every update
       
                 




            

        
