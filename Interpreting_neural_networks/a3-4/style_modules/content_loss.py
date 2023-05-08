import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        # Compute the content loss
        #content_loss = None
        #print(content_current.shape)
        #print(content_original.shape)


        content_hat = content_current - content_original
        content_loss = content_weight * torch.sum((content_current - content_original) ** 2)

        #content_loss /= (content_current.shape[2] * content_current.shape[3]*content_current.shape[1])
        
        #rem
        #here i ma performing the checking if the torch values are nans
        if torch.isnan(content_loss):
            print("content_loss is nan")
            print("content_current",content_current)
            print("content_original",content_original)
            #check if the content_hat is nan
            print("content_hat",content_hat)
            print("content_weight",content_weight)
            print("content_loss",content_loss)

        return content_loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

