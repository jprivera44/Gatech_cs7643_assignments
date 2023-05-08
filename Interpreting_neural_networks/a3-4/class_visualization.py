import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from helpers.image_utils import preprocess, deprocess, SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage import gaussian_filter1d
import random
import numpy as np
import torch
from helpers.data_utils import load_images
import torchvision.models as models
from tqdm import tqdm

class ClassVisualization:


    @staticmethod
    def jitter(X, ox, oy):
        """
        Helper function to randomly jitter an image.

        Inputs
        - X: PyTorch Tensor of shape (N, C, H, W)
        - ox, oy: Integers giving number of pixels to jitter along W and H axes

        Returns: A new PyTorch Tensor of shape (N, C, H, W)
        """
        if ox != 0:
            left = X[:, :, :, :-ox]
            right = X[:, :, :, -ox:]
            X = torch.cat([right, left], dim=3)
        if oy != 0:
            top = X[:, :, :-oy]
            bottom = X[:, :, -oy:]
            X = torch.cat([bottom, top], dim=2)
        return X

    @staticmethod
    def blur_image(X, sigma=1.0):
        X_np = X.cpu().clone().numpy()
        X_np = gaussian_filter1d(X_np, sigma, axis=2)
        X_np = gaussian_filter1d(X_np, sigma, axis=3)
        X.copy_(torch.Tensor(X_np).type_as(X))
        return X

    def create_class_visualization(self, target_y, class_names, model, **kwargs):
        """
        Generate an image to maximize the score of target_y under a pretrained model.

        Inputs:
        - target_y: Integer in the range [0, 25) giving the index of the class
        - model: A pretrained CNN that will be used to generate the image
        - dtype: Torch datatype to use for computations

        Keyword arguments:
        - l2_reg: Strength of L2 regularization on the image
        - learning_rate: How big of a step to take
        - num_iterations: How many iterations to use
        - blur_every: How often to blur the image as an implicit regularizer
        - max_jitter: How much to gjitter the image as an implicit regularizer
        - show_every: How often to show the intermediate result
        - generate_plots: to plot images or not (used for testing)
        """

        model.eval()

        # model.type(dtype)
        l2_reg = kwargs.pop('l2_reg', 1e-3)
        #origingal learning rate was 25
        learning_rate = kwargs.pop('learning_rate', 35)
        num_iterations = kwargs.pop('num_iterations', 150)
        blur_every = kwargs.pop('blur_every', 10)
        max_jitter = kwargs.pop('max_jitter', 16)
        show_every = kwargs.pop('show_every', 25)
        generate_plots = kwargs.pop('generate_plots', True)

        # Randomly initialize the image as a PyTorch Tensor, and also wrap it in
        # a PyTorch Variable.
        img = torch.randn(1, 3, 224, 224).mul_(1.0)
        img_var = Variable(img, requires_grad=True)

        ########################################################################
        # TODO: Use the model to compute the gradient of the score for the     #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. Don't forget the #
        # L2 regularization term. We use this function in our loop below.      #
        ########################################################################

            

        #Function calls: This function below is being called rn, about 100 times.
        def compute_gradient(model, img_var):
            pass
            #steps:
            #1. Perform the forwrd pass
            #2. Calculate the score of class y
            #3. Calcualte the graidient for class y with respect to the input
            #4. Alter image pixels, with gradient ascent.

            """"""
            #STEP 1: Forward pass
            #Removing the option for torch.no_grad() to leave the computation for gradients
            #removing forward, since below is equivalent
            output = model(img_var)


            #STEP 2: Calculate score
            #not sure why we need to zero out the gradients first
            score = output[0,target_y] - l2_reg * torch.sum(img_var * img_var)
            #model.zero_grad()
            #print("Score: ", score.item())


            #STEP 3: Calculate gradient
            score.backward()
            #prforming the calculation
            ##added in the portion to select the data from the grad
            grad_tensor = img_var.grad.data
            #img_var.data = img_var.data + learning_rate * gradient 


            #STEP 3.5: Regularization
            #so far this hasn't worked so now I'm going to add in the L2 regularization
            #taking out the item below potentially impacts the output, its hard to tell
            grad_tensor += 2 * l2_reg * img_var.data


            #STEP 4: Gradient Ascent
            #now that I know the gradient, I can use to go up the function
            ##I'm also setting to the img_var.data instead of the self.image variable
            #this code here creates a totally new variable, and negates the build up of the computation graph
            #img_var.data  = img_var.data + learning_rate * grad_tensor / torch.norm(grad_tensor)
            img_var.data += learning_rate * grad_tensor / torch.norm(grad_tensor)

            #adding in the portion to zero out the gradients before hand
            #If i don't zero out the gradients the image just saturates
            img_var.grad.data.zero_()

            return score.item()
            #the things that tripped me up
            # - subtracting the l2_reg term from the score along witht he squared image
            # - not zeroing out the gradients before hand
            #Things to check,
            #why am I not setting the gradient to zero before hand? 
            #why am I only editing the img_var vs. img
            

            

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        score_placeholder = 0
        for t in tqdm(range(num_iterations), desc="Processing image", leave=True):
            # Randomly jitter the image a bit; this gives slightly nicer results
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.copy_(self.jitter(img, ox, oy))

            score_placeholder = compute_gradient(model, img_var)

            # Undo the random jitter
            img.copy_(self.jitter(img, -ox, -oy))

            # As regularizer, clamp and periodically blur the image
            for c in range(3):
                lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
                hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
                img[:, c].clamp_(min=lo, max=hi)
            if t % blur_every == 0:
                self.blur_image(img, sigma=0.5)

            # Periodically show the image
            if generate_plots:

                if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
                    plt.imshow(deprocess(img.clone().detach()))
                    class_name = class_names[target_y]
                    plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
                    plt.gcf().set_size_inches(4, 4)
                    plt.axis('off')
                    plt.savefig('visualization/class_visualization_iter_{}'.format(t + 1), bbox_inches='tight')

        final_score = score_placeholder
        print('Final score: ', final_score)
        return deprocess(img.cpu()), final_score

if __name__ == '__main__':

    #creating a value to check on the total scores
    total_scores = []

    # Check for GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT').to(device)
    for param in model.parameters():
        param.requires_grad = False
    

    cv = ClassVisualization()
    X, y, labels, class_names = load_images(num=5)
    visuals = []

    #for loop is only for 5 images
    for target in tqdm(y, desc="Creating class visualization", leave=True):

        
        #CALL GRADIENT CALCULATION FUNCTION
        #this is where I call the image to be generated that maximizes the score of target_y
        out, image_score = cv.create_class_visualization(target, class_names, model, generate_plots=False)

        #I then add the image to the list of visual objects
        visuals.append(out)
        total_scores.append(image_score)


    #printing out the final scores
    print("Total scores: ", sum(total_scores))

    # Create a figure and a subplot with 2 rows and 4 columns
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)

    #PLOTTING THE IMAGES
    # Loop over the subplots and plot an image in each one
    for i in tqdm(range(2), desc="Creating plots", leave=True):
        for j in range(5):
            # Load image
            if i == 0:
                image = X[j]
            elif i == 1:
                image = visuals[j]

            # Plot the image in the current subplot
            ax[i, j].imshow(image, cmap='bone')
            ax[i, j].axis('off')

            # Add a label above each image in the bottom row
            if i == 1:
                ax[i, j].set_title(labels[j].title(), fontsize=12, y=1.2)

    # Save and display the subplots
    plt.savefig("./visualization/class_visualization.png")
    plt.show()



    #Previous notes on the backwards pass
    #setting up the calculation in the computation graph
    #output_loss.backwards()
    ##here we are getting the target score
    #output.backward(torch.ones_like(output))
    ##then calculating the gradients with respect to the calculated score tensor


    #previous notes
    #this section of code is no longer needed
    #extract the scores of the output tensor the specified class variable
    #score_tensor = output[:,target_y]

    ##start of class
    #its really all about this image, which I am going to work with to maximize the score.

    #the img that I have here is just randomized noise
    #so I need to see how I can maximize the score to a specific class.

    #print("y_class: ", target_y)

    #write the code to check the score of the image for the target class
    #entering code section


    #""""""
    #I first need to pass the input image through the network
    #not sure why I'm doing torch.nograd
    #This how nn's work, we take an image and we are passing it throught he network
    #then we get  how the neurons were activated
    #with torch.no_grad

    #after with torch.no_grad(): I need to get the output of the network
    #then I guess we need to extract the score from the final layer of the network.
    #get the index of the model's class label
    #index = mod
    #index_of_class = class_l
    #activation_score = output[0,]

    #now that I've passed the image through the netwrok I need to see the activation score, for the target class
    #activation score
