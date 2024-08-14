import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

INPUT_SIZE = 28
sequnece_length = 28
NUM_CLASSES = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 1
num_layer = 2
hidden_size = 256

train_dataset = datasets.MNIST(root="root" , train=True , transform=transforms.ToTensor() , download=True)
test_dataset = datasets.MNIST(root="root" , train=False , transform=transforms.ToTensor() , download=True)

train_loader = DataLoader(train_dataset , BATCH_SIZE , shuffle=True)
test_loader = DataLoader(test_dataset , BATCH_SIZE , shuffle=False)

# image , label = train_dataset[0]

class RNN(nn.Module):
    def __init__(self, input_size , hidden_size , num_layer , output_size):
        # input --> [1,28,28]
        super(RNN , self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layer
        # for batch first --> (batch_size x no_of_seq x no_of_features)
        # self.rnn = nn.RNN(input_size=input_size , hidden_size=hidden_size , num_layers=num_layer, batch_first=True)
        # self.gru = nn.GRU(input_size=input_size , hidden_size=hidden_size , num_layers=num_layer, batch_first=True)

        self.lstm = nn.LSTM(input_size , hidden_size , num_layer , batch_first=True)

        # after rnn , output --> [1,28,256] , hidden state output --> [2,1,256] as we have passed it in forward function
        self.fc = nn.Linear(hidden_size , output_size)
        # output --> [1 , 10]

    def forward(self , x):
        # print(x.shape) --> [1,28,28]
        h0 = torch.zeros(self.num_layers, x.size(0) , self.hidden_size)
        # since the lstm has cell state which is not in the case of rnn and gru 
        c0 = torch.zeros(self.num_layers , x.size(0) , self.hidden_size)

        # print(h0.shape) --> [2,1,256]
        # out , h1 = self.rnn(x , h0)
        # out , h1 = self.gru(x , h0)

        out , h1 = self.lstm(x , (h0 , c0))
        # out = out.reshape(out.shape[0] , -1) # --> [batch_size , 28*28]
        out = self.fc(out[:,-1, :])
        return out


model = RNN(INPUT_SIZE , hidden_size , num_layer , NUM_CLASSES)
# y = model(image)
# print(y.shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters() , lr= LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch , (image , label) in enumerate(train_loader):
        # print(image.shape)
        image = image.squeeze(1)
        # print(f"after reshaping {image.shape}")

        out = model(image)
        loss = loss_fn(out , label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch + 1} ............")

def accuracy(loader , model):
    if loader.dataset.train:
        print('checking acc on trainig data')
    else:
        print('checking on test data')

    num_correct , num_sample = 0 , 0
    with torch.inference_mode():
        for x , y in loader:
            x = x.squeeze(1)
            scores = model(x)
            _ , pred = scores.max(1)
            num_correct +=(pred == y).sum()
            num_sample += pred.size(0)

        print(f"got {num_correct}/{num_sample} with acc {(float(num_correct) / float(num_sample))*100:.2f}")

accuracy(train_loader , model)
accuracy(test_loader, model)
        