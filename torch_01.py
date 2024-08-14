# https://www.learnpytorch.io/01_pytorch_workflow/#2-build-model

import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn # it contains all pytorch building blocks for neural networks

# Data (preparing and loading)

# We'll use linear regression to create the data with known parameters 

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start , end, step)
# creating a target tensor 
Y =  weight*X + bias

# spllititng data into training and testing
train_split = int(0.8*len(X)) #getting 80 percent of x data

x_train , y_train = X[:train_split] , Y[:train_split]
x_test , y_test = X[train_split:] , Y[train_split:]

# plotting the data 
def plot_prediction(train_data = x_train , train_label = y_train , test_data = x_test , test_labels = y_test , predictions = None):
    'plots training data , test data and compares predictions'
    plt.scatter(train_data , train_label , c = "b" , s=4 , marker="*" )
    plt.scatter(test_data , test_labels , c = "g" , s=4 , marker="^" )

    if predictions is not None:
        plt.scatter(test_data , predictions , c="r" , marker="D")
    
    plt.show()

# *************************************************Building the model*******************************************************************

class LinearRegressionModel(nn.Module): # almost everything inherits from pytorch nn
    # nn.Module is the base class for all neural networks modeules
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1 , requires_grad=True , dtype=torch.float32)) #A kind of Tensor that is to be considered a module parameter.
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True , dtype= torch.float32))

# all nn.Module subclasses requires you to overwrite forward() , this method defines what happens in forward computation
    def forward(self , x:torch.tensor) -> torch.Tensor:
        return self.weight*x + self.bias
    
# create a random seed
# torch.manual_seed(42)  

linear_model = LinearRegressionModel()

# checking the model 
# print(linear_model)
# # print(list(linear_model.parameters())) # returns the parameter
# print(linear_model.state_dict()) # list named parameter with thier values


#******************** prediction with model --> torch.inference_mode() is used when using a model for inference (making predictions).
with torch.inference_mode():
    y_pred = linear_model(x_test)

# we can also do this -->
# y_pred = linear_model(x_test) #'but above code is faster because --> '
'''
torch.inference_mode() turns off a bunch of things (like gradient tracking, which is necessary for training but not for inference(making prediction)) 
to make forward-passes (data going through the forward() method) faster.'''

# print(y_pred)




'''
Loss function: Used when we refer to the error for a single training example.
Cost function: Used to refer to an average of the loss functions over an entire training data.
so loss function may also be called cost function or criterion in different areas'''

'''
optimiser: --> takes into account the loss of a model and adjusts the model's parameter to imporve model and reduce cost function. '''

#setup a loss function
loss_function = nn.L1Loss()

# setup an optimizer
optimizer = torch.optim.SGD(params=linear_model.parameters() , lr= 0.1) 
#lr = learning rate --> defines how big/small the optimizer changes the parameter (a small lr results in a small change  , a large lr results in large changes)



'''Training'''
# Set the number of epochs (how many times the model will pass over the training data)
torch.manual_seed(42)
epochs= 100

epoch_count = []
loss_values = []
test_loss_values = []

#1)--> loop through the data
for epoch in range(epochs):
    # set the model to training model 
    linear_model.train()

    # 2)--> forward pass
    y_pred = linear_model(x_train)

    # 3) --> calculate loss function
    loss = loss_function(y_pred , y_train)

    # 4)--> optimizer zero grad
    optimizer.zero_grad()

    # 5) --> perform backpropagation on the loss wrt the parameters of the model
    loss.backward()

    # 6)--> step the optimizer (perform gradient descent)
    optimizer.step() #by default how the optimizer changes will acculumate through the loop so ... we have to zero them above in step 3 for the next iteration of the loop

    '''****************************************************Testing********************************************************************************''' 
    linear_model.eval() #Set the module in evaluation mode.--> turns off gradient tracking,etc
    with torch.inference_mode():
        # 1)--> forward pass 
        test_pred = linear_model(x_test)

        # 2)--> calculate the loss 
        test_loss = loss_function(test_pred , y_test)

    if epoch%10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        # print(f"Epoch {epoch} || loss {loss} || test loss {test_loss}")

# plot_prediction(predictions=test_pred)

'''converting tensor to numpy array and then plotting it''' 

# plt.plot(epoch_count , np.array(torch.tensor(loss_values).cpu().numpy()) , label = "Train Loss")
# plt.plot(epoch_count , np.array(torch.tensor(test_loss_values).cpu().numpy()) , label = "Test Loss")
# plt.title('Training and test loss curve')
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.show()


'''Saving a model in PyTorch'''

# from pathlib import Path

# # create the model folder
# model_path = Path('models')
# model_path.mkdir(parents=True , exist_ok=True)

# # create model save path 
# model_name = 'linear_regression.pth'
# model_save_path = model_path/model_name

# # saving model state dict 
# print('saving')
# torch.save(obj=linear_model.state_dict() , f=model_save_path)

'''Loading a PyTorch model'''

# # Instantiate a new instance of our model (this will be instantiated with random weights)
# loaded_model_0 = LinearRegressionModel()

# # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
# loaded_model_0.load_state_dict(torch.load(f=model_save_path))

'''Tessting the our loaded model'''

# loaded_model_0.eval()
# with torch.inference_mode():
#     y_load_pred = loaded_model_0(x_test)

# print(test_pred == y_load_pred)
