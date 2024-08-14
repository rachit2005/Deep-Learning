import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

x, y = make_circles(1000 , noise=0.03 , random_state=42)  

# make dataframe of circle data
circles = pd.DataFrame({'x1' : x[:,0] , 'x2': x[:,1] , 'label': y})


# Turn data into tensors
x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# splitting data into test and train
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2 , random_state=42)


'''***********************Improving the model********************************'''

class CircleModelv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2 , out_features=10)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=10 , out_features=10)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(in_features=10 , out_features=10)
# PyTorch has a bunch of ready-made non-linear activation functions that do similiar but different things.
# One of the most common and best performing is ReLU (rectified linear-unit, torch.nn.ReLU()).
        self.relu_3 = nn.ReLU()
        self.layer_4 = nn.Linear(in_features=10 , out_features=1)
        self.relu_4 = nn.ReLU()

    def forward(self,x):
        z = self.layer_1(x)
        z = self.relu_1(z)
        z = self.layer_2(z)
        z = self.relu_2(z)
        z = self.layer_3(z)
        z = self.relu_3(z)
        z = self.layer_4(z)

        return self.relu_4(z)

def accuracy(y_true , y_pred):
    correct = torch.eq(y_true , y_pred).sum().item()
    acc = (correct/len(y_pred) ) *100
    return acc

model_1 = CircleModelv1()

loss_fn = nn.BCEWithLogitsLoss()
optimizer_2 = torch.optim.SGD(params=model_1.parameters() , lr = 0.1)

torch.manual_seed(42)

epochs = 2000

for epoch in range(epochs):
    model_1.train()

    y_logits_2 = model_1(x_train).squeeze()
    y_pred_2 = torch.round(torch.sigmoid(y_logits_2)) # logits --> probabilities --> predict values

    loss_2 = loss_fn(y_logits_2 , y_train)
    acc_2 = accuracy(y_train , y_pred_2)

    optimizer_2.zero_grad()

    loss_2.backward()

    optimizer_2.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits_2 = model_1(x_test).squeeze()
        test_pred_2 = torch.round(torch.sigmoid(test_logits_2))

        test_loss_2 = loss_fn(test_logits_2 , y_test)
        test_acc_2 = accuracy(y_test , test_pred_2)

    if epoch%100 ==0:
        print(f"Epoch: {epoch} | Loss: {loss_2:.5f}, Accuracy: {acc_2:.2f}% | Test loss: {test_loss_2:.5f}, Test acc: {test_acc_2:.2f}%")

# Plot decision boundaries for training and test sets
from helper_function import plot_decision_boundary
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_0, x_train, y_train) # model_0 = no non-linearity

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, x_test, y_test) # model_3 = has non-linearity
plt.show()