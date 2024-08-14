import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# give us data in the numpy array format 
x_blobs , y_blobs = make_blobs(1000 , n_features=NUM_FEATURES , #x features
                           centers=NUM_CLASSES , # y features 
                           cluster_std=1.5 ,  #give the clusters a little shake up (try changing this to 1.0, the default)
                           random_state=RANDOM_SEED)

# converting data to tensor 
x_blobs = torch.from_numpy(x_blobs).type(torch.float32)
y_blobs = torch.from_numpy(y_blobs).type(torch.LongTensor)

x_train , x_test , y_train , y_test = train_test_split(x_blobs , y_blobs , test_size=0.2, random_state=42)

from torch import nn
class BlobModel(nn.Module):
    def __init__(self , in_feat , out_feat , hidden_unit = 8):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(in_features= in_feat , out_features=hidden_unit),
            # nn.ReLU(),
            nn.Linear(in_features= hidden_unit , out_features=hidden_unit),
            # nn.ReLU(),
            nn.Linear(in_features= hidden_unit , out_features=out_feat)
        )

    def forward(self , x):
        return self.layer_stack(x)

model = BlobModel(NUM_FEATURES , NUM_CLASSES , 8)

loss_function = nn.CrossEntropyLoss()  # use for multi classification model
optimizer = torch.optim.SGD(params=model.parameters() , lr=0.01)

def accuracy(y_true , y_pred):
    correct = torch.eq(y_true , y_pred).sum().item()
    acc = (correct/len(y_pred) ) *100
    return acc

"******************************** Training And Testing Loops ****************************"
# Fit the model
torch.manual_seed(42)

# Set number of epochs
epochs = 100


for epoch in range(epochs):
    ### Training
    model.train()

    # 1. Forward pass
    y_logits = model(x_train) # model outputs raw logits 
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels
    # print(y_logits)
    # 2. Calculate loss and accuracy
    loss = loss_function(y_logits, y_train) 
    acc = accuracy(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model(x_test)
      test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
      # 2. Calculate test loss and accuracy
      test_loss = loss_function(test_logits, y_test)
      test_acc = accuracy(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


'''*****************************   Plotting the classificationg graph  *************************************'''

from helper_function import plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, x_test, y_test)

plt.show()