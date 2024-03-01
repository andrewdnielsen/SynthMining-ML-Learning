import torch
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.weight = nn.Parameter(torch.rand(1, dtype=torch.float32), requires_grad=True)
        self.bias   = nn.Parameter(torch.rand(1, dtype=torch.float32), requires_grad=True)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.weight * x + self.bias
        return x

model = LinearRegressionModel()

learning_rate = 0.001
num_epochs = 100

x_train = torch.rand((30, 1), dtype=torch.float) * 10
y_train = 4.5 * x_train + 2.7

loss_fn = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for iter in range(len(x_train)):
        # forward pass of the network
        y_pred = model(x_train[iter])
        
        # compute the loss
        loss = loss_fn(y_pred, y_train[iter])
        
        # reset gradient of parameters in network
        optim.zero_grad()
        
        # backwards pass
        loss.backward()
        
        # gradient descent step
        optim.step()
    
    slope = float(model.weight[0])
    intercept = float(model.bias[0])
    
    print(f'Epoch {epoch+1}: {slope:.3f}x + {intercept:.3f}')
    