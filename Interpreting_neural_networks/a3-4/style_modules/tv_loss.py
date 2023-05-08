import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total varation loss function                               #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        # Compute the total variation of the image

        #print(img.shape)
        #tv_loss = tv_weight * (
         #   torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
          #  torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        #)
        torch_p = None
        h,w = img.shape[2],img.shape[3]
        #this section is for debugging
        #the torch_p is used to print the shape of the tensor, so that we can know the shape of the tensor
            
        tv_h = torch.sum((img[:, :, 1:, :] - img[:, :, :-1, :])**2,dim = [1,2,3])

        tv_aa = torch.sum((img[:, :, 1:, 1:] - img[:, :, :-1, :-1])**2,dim = [1,2,3])
        tv_w = torch.sum((img[:, :, :, 1:] - img[:, :, :, :-1])**2,dim = [1,2,3])
        if torch_p != None:
            print("tv_h",tv_h.shape)
            print("tv_w",tv_w.shape)
            print("tv_aa",tv_aa.shape)

        tv_loss = tv_weight * torch.mean(tv_h + tv_w)
        tv_loss_hat   = tv_weight * torch.mean(tv_h + tv_w - tv_aa)

        #tv_loss /= (img.shape[2] * img.shape[3]*img.shape[1])

        return tv_loss


        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################