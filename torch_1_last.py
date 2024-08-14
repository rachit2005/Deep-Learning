import torch
import torch.nn.functional as F
from torch import nn

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start , end, step).unsqueeze(dim=1)
print(X.shape)
# created a target tensor 
Y =  weight*X + bias

# spllititng data into training and testing
train_split = int(0.8*len(X)) #getting 80 percent of x data

x_train , y_train = X[:train_split] , Y[:train_split]
x_test , y_test = X[train_split:] , Y[train_split:]

'''using subclass nn.Module to make our Model '''
# Subclass nn.Module to make our model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(1,1)
    
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
         return self.linear_layer(inputs)

# Set the manual seed when creating the model (this isn't always need but is used for demonstrative purposes, try commenting it out and seeing what happens)
torch.manual_seed(42)
model = LinearRegressionModelV2()

# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model.parameters(), # optimize newly created model's parameters
                            lr=0.01)

torch.manual_seed(42)

# Set the number of epochs 
epochs = 1

for epoch in range(epochs):
    ### Training
    model.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model(x_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model(x_test)
    
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")      