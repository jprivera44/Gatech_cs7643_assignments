"""
S2S Encoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        #Embedding layer
        self.embedding = nn.Embedding(input_size,emb_size)

        #Recurrent layer
        #Guessing on the number of recurrent layers to be just 1 right now.
        #self.recurrent = nn.LSTM(emb_size,encoder_hidden_size,1,dropout)
        #Need to have the code handle both types of network, the RNN & LSTM
        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size, encoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(emb_size, encoder_hidden_size, batch_first=True)

        #Linear layer
        #says that the size of the output of the 2nd layer, must match a contraint related to the decoder
        self.linear_layer = nn.Sequential(
          nn.Linear(encoder_hidden_size,encoder_hidden_size),
          nn.ReLU(),
          nn.Linear(encoder_hidden_size,encoder_hidden_size)
        )
        #self.linear_layer = nn.Linear(encoder_hidden_size,decoder_hidden_size)

        #Dropout layer
        self.dropout = nn.Dropout(dropout)

        #Tahn layer
        self.tanh = nn.Tanh()


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the weights coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################
        #INITIAL PRINT
        print("The input is",input)

        #EMBEDDING LAYER
        embed_out = self.embedding(input)
        print("Output from the embedding layer is",embed_out)

        #DROPOUT LAYER
        embed_drop = self.dropout(embed_out)

        #RECURRENT LAYER
        #Need to check between LSTM and RNN
        #Here I am setting up the return to be a tuple if the model called is an LSTM
        if self.model_type == 'LSTM':
          output, (hidden,cell) = self.rnn(embed_drop)

        else:
          output, hidden = self.rnn(embed_drop)
          cell = None


        #TRANSFORM HIDDEN
        hidden = self.linear_layer(hidden)

        #TANH ACTIVATION LAYER
        hidden = self.tanh(hidden)


        #output, hidden = None, None

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # Do not apply any linear layers/Relu for the cell state when model_type is #
        # LSTM before returning it.                                                 #

        return output, hidden
