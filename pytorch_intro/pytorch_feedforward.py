import torch
from torch import nn

class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # TODO: set up hidden layers and activation functions for use in forward pass
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass using hidden layers & activation functions
        
        return x

# instantiate pytorch model
model = PolynomialRegressionModel()

# initializing sample training data
x_train = torch.rand((30, 1), dtype=torch.float) * 10
y_train = 4.5 * (x_train ** 3) + 3.6 * (x_train ** 2) - 11.7 * x_train + 2.7

# TODO: experiment with hyperparameters on training
num_epochs = 40
learning_rate = 0.001

# TODO: initialize loss function and optimizer

for epoch in range(num_epochs):
    for iter in range(len(x_train)):
        # TODO: forward pass of the network
        
        # TODO: compute the loss
        
        # TODO: reset gradient of parameters in network
        
        # TODO: backwards pass of the network
        
        # TODO: gradient descent step with optimizer
        
        pass
    
    # TODO: print loss at the end of each epoch
    