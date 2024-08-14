# https://www.learnpytorch.io/02_pytorch_classification/#0-architecture-of-a-classification-neural-network 

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

x, y = make_circles(1000 , noise=0.03 , random_state=42)  

# make dataframe of circle data
circles = pd.DataFrame({'x1' : x[:,0] , 'x2': x[:,1] , 'label': y})

''' visualise the data'''
plt.scatter(x=x[:,0] , y=x[:,1] )
# plt.show()

'''check input and output shapes'''
# we x and y as array but we want it in tensor

# Turn data into tensors
x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# splitting data into test and train
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2 , random_state=42)

'''building a model '''
from torch import nn

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0()

'we can replicate above model using nn.Sequential() --> for simple model only'
model = nn.Sequential(
    nn.Linear(in_features=2 , out_features=5),
    nn.Linear(in_features=5 , out_features=1)
)

# make predictions
with torch.inference_mode():
    untrained_pred = model(x_test)

"setup loss function and optimizer"
# regression --> MAE or MSE (mean absolute error or mean squared error)
# classification --> binary cross entropy or categorical cross entropy

loss_fn = nn.BCEWithLogitsLoss() #binary cross entropy with logits--> for binary classification "as we have now" --> sigmoid activation function built-in
# loss_fn = nn.BCELoss() #binary cross entropy --> for binary classification "as we have now"
# loss_fn = nn.CrossEntropyLoss() #categorical cross entropy --> for multi class classification

optimizer = torch.optim.SGD(params=model.parameters() , lr = 0.01)


"calculate accuracy"
def accuracy(y_true , y_pred):
    correct = torch.eq(y_true , y_pred).sum().item()
    acc = (correct/len(y_pred) ) *100
    return acc

" raw prediction/outputs of model is known as logits"
# y_logits = model(x_test)[:5]
# print(y_logits)

'Use sigmoid on model logits'
# y_pred_probs = torch.sigmoid(y_logits)
# print(y_pred_probs)


'''
If y_pred_probs >= 0.5, y=1 (class 1)
If y_pred_probs < 0.5, y=0 (class 0)
'''

# Find the predicted labels (round the prediction probabilities)
# y_preds = torch.round(y_pred_probs)

# In full
# y_pred_labels = torch.round(torch.sigmoid(model_0(x_test)[:5]))

# Check for equality
# print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

'''********************** Training and Testing model loop ********************************'''
torch.manual_seed(42)
torch.manual_seed(42)

# Set the number of epochs
epochs = 100

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(x_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls (i.e 1,0 in this case)
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 

    acc = accuracy(y_true=y_train, y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(x_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# make prediction and evaluate the model 

import requests
from pathlib import Path

if Path("helper_function.py").is_file():
    print("already exist")
else:
    print('downloading')
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")

    with open('helper_function.py' , 'wb') as f:
        f.write(request.content)

from helper_function import plot_decision_boundary , plot_predictions

plt.subplot(1,2,1)
plt.title("train")
plot_decision_boundary(model , x_train , y_train)

plt.title("test")
plt.subplot(1,2,2)
plot_decision_boundary(model , x_test , y_test)

