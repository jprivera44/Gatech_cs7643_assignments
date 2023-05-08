import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from helpers.data_utils import *
from helpers.image_utils import *
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

class GradCAM():
    
    def __init__(self, X):
        self.X = X


    def gradcam(self):

        # Define a preprocessing function to resize the image and normalize its pixels
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Check for GPU support
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT').to(device)

        ##############################################################################
        # TODO: Define a hook function to get the feature maps from the last         #
        # convolutional layer. Then register the hook to get the feature maps        #
        ##############################################################################
        
        ################## Part 1: Print and Initialize
        print("Model is ",model)
        feature_maps = {}
        


        ################## Part 2: Define the forward hook function
        #2 experiments
        #the first just having it call the hook function by itself
        #the second it just  calling the get activation funcion with the hook function embedded within
        def hook_forward(model,input,output):
            #printing to see if it was called
            print("calling forward hook function")
            #saving items into a dictionary
            feature_maps['forward'] = output.detach()

        
        ################## Part 3: Define the backward hook function
        def hook_backward(model,input,output):
            #printing to see when it is called
            print("calling backward hook function")
            #get the gradients of the target class with respect to the feature maps
            grads = output[0]

            #verify that class idx is the one I want

            #set the gradients of all classes to 0, except the target class
            grads[torch.arange(grads.size(0)) != class_idx] = 0

            #saving the gradient to the dictionary
            feature_maps['backward'] = grads
            #print("hook backwards test",input[0])
            print("shape of input is",input[0].shape)
            
            #print("Printing out the hook here", input[0][0, :1, :1])




        ################## Part 4:Register hooks and define list
        #Do I need to keep the () when I call
        #Forward:Calling the hook function on the last layer of the network
        model.features[-1].expand3x3.register_forward_hook(hook_forward)
        #Backward:registering for the backwards call
        model.features[-1].expand3x3.register_backward_hook(hook_backward)
        #define the list for the gradcams
        gradcams = []


        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        print("starting for loop")
        for x in self.X:
            # Load an input image and perform pre-processing
            image = Image.fromarray(x)
            x = preprocess(image).unsqueeze(0)

            # Make a forward pass through the model
            logits = model(x)

            # Get the class with the highest probability
            class_idx = torch.argmax(logits)

            #SECTION2
            ###############################################################################
            # TODO: To generate a Grad-CAM heatmap, first compute the gradients of the    #
            # output class with respect to the feature maps. Then, calculate the weights  #
            # of the feature maps. Using these weights, compute the Grad-CAM heatmap.     #
            # Use the cv2 for resizing the heatmap if necessary.                          #
            ###############################################################################
            print("########################## NEW IMAGE####################")
            #model.zero_grad()
            #FORWARD LOOP OR FUNCTION
            #print("Insdie for loop:",feature_maps)

            #let's get the score of the class that has the highest
            score_of_class = logits[0,class_idx]

            #I need to use the score of the class, and the 
            #how do I take the 

            score_of_class.backward(retain_graph=True)

            print("Shape of the backwards items",feature_maps['backward'].shape)
            
            #for name, param in model.named_parameters():
             #   print("Printing the name and the param grad",name,param.grad)


            value = feature_maps['forward']
            #print("shape of value is",value[:,:-1])
            # Compute the mean along the channel dimension
            weights = torch.mean(value, dim=(2, 3))

            # Apply a spatial softmax function to obtain the weights
            #weights = F.softmax(mean, dim=1)
            heatmap = torch.sum(value * weights.unsqueeze(2).unsqueeze(3), dim=1).squeeze(0)

            heatmap = cv2.resize(heatmap.numpy(), (x.shape[3], x.shape[2]))



            #weights = torch.mean(value,dim =1)
            #print("weight shapes are",weights.shape)
            #print("Weight averages are",weights[0])
            #print(" ")
            #print(" ")


            #setting up the dummy heatmap
            #heatmap = np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

            # store gradcams
            gradcams.append([x.squeeze(0).permute(1,2,0).numpy(), heatmap])

        return gradcams


if __name__ == '__main__':

    # Retrieve images
    X, y, labels, class_names = load_images(num=5)
    gc = GradCAM(X)
    gradcams = gc.gradcam()
    # Create a figure and a subplot with 2 rows and 4 columns
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)

    # Loop over the subplots and plot an image in each one
    for i in tqdm(range(2), desc="Creating plots", leave=True):
        for j in tqdm(range(5), desc="Processing image", leave=True):
            # Load image
            if i == 0:
                item = gradcams[j]
                image = item[0].clip(0,1)
                ax[i, j].imshow(image, alpha=.87, vmin=.5)
                ax[i, j].axis('off')
            elif i == 1:
                item = gradcams[j]
                image = item[0].clip(0,1)
                overlay = item[1]

                # Plot the image in the current subplot
                ax[i, j].imshow(image, alpha=1, vmin=100.5, cmap='twilight_shifted')
                ax[i, j].imshow(overlay, cmap='viridis', alpha=0.779)
                ax[i, j].axis('off')

            # Add a label above each image in the bottom row
            if i == 1:
                ax[i, j].set_title(labels[j].title(), fontsize=12, y=1.2)

    # Save and display the subplots
    plt.savefig("./visualization/gradcam_visualization.png")
    plt.show()



    """
                print(" ")
            print("Starting loop")
            conv_output_sp = None
            for pos , module in model.features._modules.items():
                #print("Module is ",module)
                #print("Pos is ",pos)
                #if isinstance(module, torch.nn.Conv2d):
                #if module == model.features[10].expand3x3:
                if int(pos) == 11:
                    #in the example they had x here it is logits
                    logits.register_hook(save_grad)
                    conv_output_sp = logits

                    print("module position",pos)
                    print("Module",module.expand3x3)
    
    """