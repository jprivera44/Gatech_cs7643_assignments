"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn

#JP's edit to include the functional
import torch.nn.functional as F


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        #weights
        #w_ig
        #w_hg
        #w_io
        #w_ho

        #Glorot Initialization, why are we using a 6 here?
        limit_i = np.sqrt(6/(input_size + hidden_size))

        #2nd limit for the hidden
        limit_h = np.sqrt(6/(hidden_size + hidden_size))


        # i_t: input gate
        #starting with the weights and biases needed here
        #w_ii
        self.w_ii = nn.Parameter(torch.FloatTensor(input_size,hidden_size).uniform_(-limit_i,limit_i))
        #b_ii
        self.b_ii = nn.Parameter(torch.zeros(hidden_size))
        #w_hi
        self.w_hi = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size).uniform_(-limit_h,limit_h))
        #b_hi
        self.b_hi = nn.Parameter(torch.zeros(hidden_size))


        # f_t: the forget gate
        self.w_if = nn.Parameter(torch.FloatTensor(input_size,hidden_size).uniform_(-limit_i,limit_i))

        self.b_if = nn.Parameter(torch.zeros(hidden_size))

        self.w_hf = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size).uniform_(-limit_h,limit_h))

        self.b_hf = nn.Parameter(torch.zeros(hidden_size))


        # g_t: the cell gate
        self.w_ig = nn.Parameter(torch.FloatTensor(input_size,hidden_size).uniform_(-limit_i,limit_i))

        self.b_ig = nn.Parameter(torch.zeros(hidden_size))

        self.w_hg = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size).uniform_(-limit_h,limit_h))

        self.b_hg = nn.Parameter(torch.zeros(hidden_size))


        # o_t: the output gate
        # Below I am assuming like the gates before it, the io, is input/hidden and that the ho is hidden/hidden
        self.w_io = nn.Parameter(torch.FloatTensor(input_size,hidden_size).uniform_(-limit_i,limit_i))

        self.b_io = nn.Parameter(torch.zeros(hidden_size))

        self.w_ho = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size).uniform_(-limit_h,limit_h))

        self.b_ho = nn.Parameter(torch.zeros(hidden_size))


        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        
        #INIT
        #x has shape of [4,3,2]
        batch_size = x.shape[0]
        batch_size,sequence_length,_ = x.size()

        #Removing this for now
        #h_t, c_t = None, None

        #Initialize if the values are none
        if init_states is None:
          h_t = torch.zeros(batch_size,self.hidden_size,device = x.device)
          c_t = torch.zeros(batch_size,self.hidden_size,device = x.device)


        #For loop going through each item in the batch
        for t in range(sequence_length):
          #Setting up an item from the batch
          x_t = x[:,t,:]
          
          #I_T
          i_t = torch.sigmoid(x_t @ self.w_ii + self.b_ii + h_t @ self.w_hi + self.b_hi)

          #F_T
          f_t = torch.sigmoid(x_t @ self.w_if + self.b_if + h_t @ self.w_hf + self.b_hf)

          #G_t
          g_t = torch.tanh(x_t @ self.w_ig + self.b_ig + h_t @ self.w_hg + self.b_hg)

          #O_T
          o_t = torch.sigmoid(x_t @ self.w_io + self.b_io + h_t @ self.w_ho + self.b_ho)

          #c_t
          c_t =torch.mul(f_t,c_t) + torch.mul(i_t,g_t)

          #h_t
          h_t = torch.mul(o_t,torch.tanh(c_t))
                  
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)




