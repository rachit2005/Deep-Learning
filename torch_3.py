# https://www.learnpytorch.io/03_pytorch_computer_vision/#where-does-computer-vision-get-used  

import torch
from torch import nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision # --> contains datasets , model architecture and image transformation often used in computer vision 
from torchvision import datasets # --> example of computer visionn datasets
from torchvision import transforms # --> images need to be transformed (turned into a number) before being used in models , common image transformation are found in here
from torchvision.transforms import ToTensor #--> convert a PIL image or numpy array to tensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer 

'''Tensor image are expected to be of shape (C, H, W), where C is the number of channels, and H and W refer to height and width.'''

# fashionMNIST dataset for computer vision (importing from torch vision) is a large dataset of handdwritten digits used for training image processing system 

# we are getting data in dataset format 
train_data = datasets.FashionMNIST(
    root='data' ,# where to download data to
    train=True, # do we want the training dataset? --> true
    download= True, # do we want to download? --> true 
    transform= ToTensor(), # how do we want to transform the data --> converting it to tensor
    target_transform=None # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
classes= train_data.classes # gives classes 
class_to_idx = train_data.class_to_idx # gives a dictionary of classes with its coresponding index
data = train_data.data #gives data 

# check the shape
image , label = train_data[0]
# print(f"Images shape: {image.shape} --> (color , height , width) || labels: {labels}")
'''for black andd white images  --> color_channel = 1 ,  while for colored images --> color_channel = 3'''

'''************************* Visualisation of our data ****************************'''
# plot a single image 

# plt.imshow(image.squeeze() , cmap="gray")
# plt.title(classes[label])
# print(image.shape)
# print(image.squeeze().shape) --> Returns a tensor with all specified dimensions of input of size 1 removed.
# plt.show()
 
# plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
row , col = 4,4
for i in range(1 , row*col + 1):
    # randomly generating an index to view the images 
    random_idx = torch.randint(0,len(train_data) , size=[1]).item()
    # print(random_idx)
    img , lab = train_data[random_idx]
    fig.add_subplot(row, col , i)
    plt.imshow(img.squeeze() , cmap = "gray")
    plt.title(classes[lab])
    plt.axis(False)

# plt.show()

"***************************   Prepare Dataloader  **************************************"
from torch.utils.data import DataLoader
# dataloader turns our datasets into a python iterables 
# moreover we want to turn our data into minibatches for more computational efficient 

BATCH_SIZE = 32
# turn dataset into iterables(batches)
train_dataloader = DataLoader(
    dataset=train_data , # train data has 60000 images 
    batch_size= BATCH_SIZE, # how many sample in a batch to load
    shuffle=True 
)

test_dataloader = DataLoader(
    dataset = test_data,
    shuffle=False, # it just for evaluation not for training the model
    batch_size=BATCH_SIZE
)

'''
next() --> Return the next item from the iterator,
iter() --> convert an iterable object into an iterator. '''
train_fea_batch , train_lab_batch = next(iter(train_dataloader)) #now this is input tensor data

'''To know about torch.randint()--> https://pytorch.org/docs/stable/generated/torch.randint.html'''




'''*******************************    Build a baseline model   ***************************************'''

# # create a flatten layer
# flatten_model = nn.Flatten()

# # get a single sample
# x = train_fea_batch[0] #its size is [1,28,28]

# # flatten the sample
# output = flatten_model(x)  # it condense the information for our model
# # print(output.shape) 

class FashionModelV0(nn.Module):
    def __init__(self , in_shape: int , hidden_unit: int , output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(), # condense information  --> transform multi-dimensional input data into a one-dimensional array (recommended to check by printing as above we did)
            # this is often required when transitioning from cnn to fully connected neural networks
            nn.Linear(in_features= in_shape ,# --> size of each input sample
                    out_features=hidden_unit # --> size of each output sample
                    ),
            nn.Linear(in_features=hidden_unit , out_features=output_shape)
        )

    def forward(self,x):
        return self.layer_stack(x)
    
# torch.manual_seed(42)
# setup model with input parameter
model_0 = FashionModelV0(28*28 , 10 , len(classes))

# setup loss function, optimizer and accuracy
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters() , lr = 0.01)
from helper_function import accuracy_fn

'''Creating a function to time our experiments'''

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time. and returns total time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

'''********************     Creating a training loop and training with the batch data      ******************************'''
'''
for training -->
1. loop through epochs
2. loop through training batch  , performs training steps , calculate training loss
3. loop through testing batch , perform testing steps , calculate testing loss
'''
# created a function to train other model too 

def train_step(model: torch.nn.Module, train_data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    "train model and return train loss"
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_data_loader):
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_data_loader.dataset)} samples")

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(train_data_loader)
    train_acc /= len(train_data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    return train_loss

def test_step(model: torch.nn.Module, test_data_loader: torch.utils.data.DataLoader,  loss_fn: torch.nn.Module,  accuracy_fn,):
    "test model and returns test loss and test accuracy"

    test_loss, test_acc = 0, 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        # we are not enumerating through the test dataloader 
        for X, y in test_data_loader:
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(test_data_loader)
        test_acc /= len(test_data_loader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return test_loss , test_acc

from tqdm.auto import tqdm
epochs = 3

# for model_0 -->

# torch.manual_seed(45)
# train_time_start = timer()

# for epoch in tqdm(range(epochs)):
#     print(f"epoch : {epoch} \n ------------")
#     # looping through the training batches 
#     train_loss = train_step(model_0 , train_dataloader , loss_fn , optimizer)
#     # testing loop
#     test_loss , test_acc = test_step(model_0 , test_dataloader , loss_fn , accuracy_fn)
#     print(f"\n Train loss {train_loss:.4f} || test loss {test_loss:.2f} || test acc {test_acc:.4f}")


# # calculating total time to train and evaluate our model 
# train_time_end = timer()
# total_train_time = print_train_time(train_time_start , train_time_end , "cpu")


'''Make Predictions And Get Results'''
def eval_model(model: nn.Module , data_loader : torch.utils.data.DataLoader , loss_function: nn.Module , accuracy):
    '''returns a dictionary containing the results of model predicting on data_loader'''
    loss , acc = 0,0
    model.eval()
    with torch.inference_mode():
        for x , y in data_loader:
            y_pred = model(x)
            loss += loss_function(y_pred , y)
            acc += accuracy(y_pred.argmax(dim=1) , y)

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model name" : model.__class__.__name__, # only works when model was created with a class
            "accuracy": acc,
            "loss" : loss,}

results = eval_model(model_0 , test_dataloader , loss_fn , accuracy_fn)
# print(results)


'''Creating a better model with non linearity'''
class FashionModelv1(nn.Module):
    # nn.ReLU --> turns negetive value to 0 and leave positive as it it 
    def __init__(self , in_shape:int , hidden_units:int , out_shape : int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #flatten inputs into a single vector
            nn.Linear(in_features=in_shape , out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units , out_shape),
            nn.ReLU()
        )

    def forward(self , x:torch.Tensor):
        return self.layer_stack(x)

model_1 = FashionModelv1(28*28 , 10 , len(classes))
optimizer_1 = torch.optim.SGD(params=model_1.parameters() , lr= 0.01)

# for model_1 --> 

torch.manual_seed(42)
epochs = 3
start = timer()

# for epoch in tqdm(range(epochs)):
#     print(f"epochs \n ------------------------")
#     train_loss = train_step(model_1 , train_dataloader , loss_fn , optimizer_1)
#     test_loss , test_acc = test_step(model_1 , test_dataloader , loss_fn , accuracy_fn)

#     print(f"train loss {train_loss} || test loss {test_loss} || test accuracy {test_acc}")

# end = timer()
# results_1 = eval_model(model_1 , test_dataloader , loss_fn , accuracy_fn)
# total_time = print_train_time(start , end , "cpu")

'''*****************   Building Convolutional Neural Networks --> (to understand much better and everything abt CNN --> https://poloclub.github.io/cnn-explainer/)   *******************************'''
# Input layer -> [Convolutional layer -> activation layer -> pooling layer(downsampling layer)] -> Output layer

class FashionmodelV2(nn.Module):
    '''Model architecture that replicates the TinyVGG'''

    def __init__(self, input : int , hidden_units:int , out_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Conv3D --> used with 3d images (nn.Conv.3d() expects a 5d tensor [batch number , color channels ,number of frames, height  , width] as input)
            # Conv2d --> used with 2d images (nn.Conv.2d() expects a 4d tensor [batch number , color channels , height  , width] as input)
            # Conv1d --> often used to analyse text or temporal signal (nn. Conv1d expects a 3-dimensional input in the shape [batch_size, channels, seq_len])
            nn.Conv2d(in_channels=input , 
                      out_channels=hidden_units, 
                      kernel_size=3, #--> a number that specifies the height and width of a square convolution window
                      padding=1,
                      stride=1),# defines how many pixel to filter to skip when they move across the image from left to right and top to bottom
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3, 
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            # read doc for maxpooling in cnn --> https://deepai.org/machine-learning-glossary-and-terms/max-pooling#:~:text=Max%20pooling%20is%20performed%20on,maximum%20value%20within%20the%20window.
            nn.MaxPool2d(kernel_size=2 #--> the size of the window to take a max over
            )
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3 ,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels = hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # both cnn blocks will going to learn features that best represents our data and extract them and then the output layer is going to take them and then classify them into our classes 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # To find the in_feature in nn.Linear --> watch https://www.youtube.com/watch?v=V_xro1bcAuA&t=66171s time: 18:00 till the end till u find do not skip this part
            nn.Linear(in_features=hidden_units*7*7 , out_features=out_shape)            
        )

    def forward(self , x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        classified_info = self.classifier(x)

        return classified_info

torch.manual_seed(42)
model_cnn = FashionmodelV2(input= 1, # since we have color channel = 1, 
                           hidden_units=10,
                           out_shape=len(classes)
                           )

y_cnn = model_cnn(image.unsqueeze(0)) #--> as input requires 4D tensor

'''************************       Training and Testing Model            ******************************'''
torch.manual_seed(42)
epochs = 3
start_model = timer()
optimizer_cnn = torch.optim.SGD(params=model_cnn.parameters() , lr = 0.01)

for epoch in tqdm(range(epochs)):
    train_loss_cnn = train_step(model_cnn , train_dataloader , loss_fn , optimizer_cnn)
    test_loss_cnn , test_acc_cnn = test_step(model_cnn , test_dataloader , loss_fn , accuracy_fn)

    print(f"train loss {train_loss_cnn} || test loss {test_loss_cnn} || test acc {test_acc_cnn}")

model_stop = timer() # 137.898 seconds

time_taken = print_train_time(start_model , model_stop , "cpu")
results_cnn = eval_model(model_cnn , test_dataloader , loss_fn , accuracy_fn)

''' ***************   Random Prediction       *************************'''

def make_prediction(model: nn.Module , data: list ):
    pred_probs = []
    model.eval()

    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample , dim=0)

            # forward pass 
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze() , dim=0)
            pred_probs.append(pred_prob)

    return torch.stack(pred_probs)

import random
test_sample = []
test_label = []

for sample , label in random.sample(list(test_data) , k=9):
    test_sample.append(sample)
    test_label.append(label)

# plt.imshow(test_sample[0].squeeze() , cmap="gray")
# plt.show()

pred_probs = make_prediction(model_cnn , test_sample)

# convert prediction probabilities to labels 
pred_classes = pred_probs.argmax(dim=1)

nrows = 3
ncol = 3
for i , sample in enumerate(test_sample):
    plt.subplot(nrows , ncol , i+1)

    plt.imshow(sample.squeeze() , cmap='gray')
    pred_label = classes[pred_classes[i]]
    truth_label = classes[test_label[i]]

    title_txt = f"pred {pred_label} || truth: {truth_label}"

    if pred_label == truth_label:
        plt.title(title_txt , fontsize = 10 , c="g")
    else:
        plt.title(title_txt , fontsize = 10 , c="r")
plt.axis(False)
# plt.show()

'''Confusion Matrix'''
# Import tqdm for progress bar
from tqdm.auto import tqdm

# 1. Make predictions with trained model
y_preds = []
model_cnn.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Do the forward pass
    y_logit = model_cnn(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)


from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(classes), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=classes, # turn the row and column labels into class names
    figsize=(10, 7)
)
plt.show()