import torch.nn as nn 
import torch 
from math import sqrt 
import torch.nn.functional as F 


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################
        # TODO: Design your own network, define layers here.                          #
        # Here We provide a sample of two-layer fc network from HW4 Part3.           #
        # Your solution, however, should contain convolutional layers.               #
        # Refer to PyTorch documentations of torch.nn to pick your layers.           #
        # (https://pytorch.org/docs/stable/nn.html)                                  #
        # Some common choices: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout   #
        # If you have many layers, use nn.Sequential() to simplify your code         #
        ##############################################################################
        # from 28x28 input image to hidden layer of size 256
        # self.fc1 = nn.Linear(28*28, 8) 
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 16,padding = 2, kernel_size = (5,5),stride = (2,2))
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16,out_channels = 64, padding = 2,kernel_size = (5,5),stride = (2,2))
        self.conv3 = nn.Conv2d(in_channels = 64,out_channels = 8, padding = 2,kernel_size = (5,5),stride = (2,2))
        self.fc_1 = nn.Linear(in_features = 1760, out_features =800) 
        self.fc_2 = nn.Linear(in_features=800, out_features=200)
        self.fc_3 = nn.Linear(in_features=200, out_features=4)
        
        self.init_weights()
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]

        nn.init.normal_(self.fc_1.weight, 0.0, sqrt(1/self.fc_1.weight.size(1)))
        nn.init.constant_(self.fc_1.bias, 0.0)
        ##

    def forward(self, x):
        ##############################################################################
        # TODO: Design your own network, implement forward pass here                 # 
        ##############################################################################
        
        N, C, H, W = x.shape

        ## TODO: forward pass
        z= self.pool(F.relu(self.conv1(x)))
        z= self.pool(F.relu(self.conv2(z)))
        z = F.relu(self.conv3(z))

        # z=z.permute(*torch.arange(z.ndim - 1, -1, -1))
        # print("before flatten: ", z.shape)
        z=torch.flatten(z, start_dim=1)

        # print("after resize: ",z.shape) # 32 x 1760
        z = self.fc_1(z)
        z = self.fc_2(z)
        z = self.fc_3(z)
        # print("final shape: ", z.shape)

        return z