import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as T
import PIL
from PIL import Image
from helpers.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
import matplotlib.pyplot as plt


def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def features_from_img(imgpath, imgsize, cnn, dtype):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = Variable(img.type(dtype))
    return extract_features(img_var, cnn), img_var


# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy
    vnums = list(map(int, scipy.__version__.split('.')))
    assert vnums[1] >= 16 or vnums[0] >= 1, "You must install SciPy >= 0.16.0 to complete this notebook."


# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Variable of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Variable of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def style_transfer(name, content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, content_loss, style_loss, tv_loss, cnn, dtype,
                   init_random=False, testing=False):
    """
    Run style transfer!

    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img_var = Variable(content_img.type(dtype))
    feats = extract_features(content_img_var, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_img_var = Variable(style_img.type(dtype))
    feats = extract_features(style_img_var, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(style_loss.gram_matrix(feats[idx].clone()))

    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_()
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img_var = Variable(img, requires_grad=True)

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img_var Torch variable, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img_var], lr=initial_lr)

    for t in range(200):
        if t < 190:
            img.clamp_(-1.5, 1.5)
        feats = extract_features(img_var, cnn)

        ##############################################################################
        # TODO: Implement this update rule with by forwarding it to criterion        #
        # functions and perform the backward update.                                 #
        #                                                                            #
        # HINTS: all the weights, loss functions are defined. You don't need to add  #
        # any other extra weights for the three loss terms.                          #
        # The optimizer needs to clear its grad before backward in every step.       #
        #                                                                            #
        # NOTE: There is a final optimization needed to get good style transferred   #
        #   images. Do look at the variables 'decay_lr_at' and 'decayed_lr'.         #
        #   You would need to reduce the learning rate for the last few epochs.      #
        ##############################################################################

      
        #Just calling int he content loss function
        content_loss_val = content_loss(content_weight, feats[content_layer], content_target)

        #Just calling the style loss function
        style_loss_val = style_loss(feats, style_layers, style_targets, style_weights)

        #Here it's important to call the tv_loss function
        tv_loss_val = tv_loss(img_var, tv_weight)

        #here I'm going to havve an if statement to check if the style loss is greater than the previous style loss
        if t == None:
            loss_hat = style_loss_val +100 + 100 + abs(-1)

        # Compute the overall loss and perform backpropagation
        loss = content_loss_val + style_loss_val + tv_loss_val
        #zeroing the gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Decay the learning rate for the last few epochs
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img_var], lr=decayed_lr)
            optim_hat = style_loss_val

        #if not testing and (t % 50 == 0 or t == 199):
         #   print('Iteration {}, Loss: {:.4f}'.format(t, loss.item()))

    # Save the final image
    output_img = img_var.clone().cpu().data.numpy()
    #output_img =


        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

    if not testing:
        # Create a figure and a subplot with 2 rows and 4 columns
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[0].set_title('Content Image')
        ax[1].set_title('Style Image')
        ax[2].set_title('Style Transferred')
        ax[0].imshow(deprocess(content_img.cpu()))
        ax[1].imshow(deprocess(style_img.cpu()))
        ax[2].imshow(deprocess(img.cpu()))
        plt.axis('off')
        plt.savefig('visualization/' + name + '.png', bbox_inches='tight')
        plt.show()
    # Do not edit or delete below this line - used for testing the script
    if testing:
        return img