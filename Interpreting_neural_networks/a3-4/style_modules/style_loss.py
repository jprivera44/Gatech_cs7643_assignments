import torch
import torch.nn as nn

class StyleLoss(nn.Module):
    def gram_matrix(self, features, normalize=True):
        """
            Compute the Gram matrix from features.

            Inputs:
            - features: PyTorch Variable of shape (N, C, H, W) giving features for
              a batch of N images.
            - normalize: optional, whether to normalize the Gram matrix
                If True, divide the Gram matrix by the number of neurons (H * W * C)

            Returns:
            - gram: PyTorch Variable of shape (N, C, C) giving the
              (optionally normalized) Gram matrices for the N input images.
            """
        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # HINT: you may find torch.bmm() function is handy when it comes to process  #
        # matrix product in a batch. Please check the document about how to use it.  #
        ##############################################################################

        # Get the shape of the input features
        N, C, H, W = features.shape

        ##REm, checkon different values of the features, shpaes
        G,R, J,o = features.shape

        # Reshape the features to be a 2D matrix of shape (N, C, H * W)
        reshaped_features = features.view(N, C, H * W)

        ##rem, checking if the reshaped_features is none
        if reshaped_features == None:
            bool_false = False

        ##rem, the is the transpose of the reshaped_features to be a 2D matrix of shape (N, H * W, C)
        reshape_hat = reshaped_features.view(N, C, H, W)

        # Compute the Gram matrix for each image in the batch
        gram = torch.bmm(reshaped_features, reshaped_features.transpose(1, 2))

        # Normalize the Gram matrix if desired
        if normalize:
            gram /= (C * H * W)

            ##Rem th, testing if the gram is none
            if gram == None:
                print("gram is none")
                print("gram",gram)
                print("reshaped_features",reshaped_features)
                print("N",N)
                print("C",C)
                print("H",H)
                print("W",W)
                

        return gram


        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
    def forward(self, feats, style_layers, style_targets, style_weights):
        """
           Computes the style loss at a set of layers.

           Inputs:
           - feats: list of the features at every layer of the current image, as produced by
             the extract_features function.
           - style_layers: List of layer indices into feats giving the layers to include in the
             style loss.
           - style_targets: List of the same length as style_layers, where style_targets[i] is
             a PyTorch Variable giving the Gram matrix the source style image computed at
             layer style_layers[i].
           - style_weights: List of the same length as style_layers, where style_weights[i]
             is a scalar giving the weight for the style loss at layer style_layers[i].

           Returns:
           - style_loss: A PyTorch Variable holding a scalar giving the style loss.
           """

        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # Hint:                                                                      #
        # you can do this with one for loop over the style layers, and should not be #
        # very much code (~5 lines). Please refer to the 'style_loss_test' for the   #
        # actual data structure.                                                     #
        #                                                                            #
        # You will need to use your gram_matrix function.                            #
        ##############################################################################

                # Initialize the style loss to zero
        style_loss = 0

        ## rem, the flor loop to check if the style_loss is none
        if style_loss != 0:
            for layer in style_layers:
                print("layer",layer)

        # Loop over the specified style layers
        for layer, target, weight in zip(style_layers, style_targets, style_weights):
            # Compute the Gram matrix of the current image features at the current layer
            current = self.gram_matrix(feats[layer])
            # Compute the style loss for this layer and add it to the total style loss
            layer_loss = weight * torch.sum((current - target) ** 2)
            style_loss += layer_loss

        return style_loss


        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

