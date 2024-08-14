# https://www.learnpytorch.io/04_pytorch_custom_datasets/#what-is-a-custom-dataset

import torch
from torch import nn
import torch.utils
import torch.utils.data
import torchvision

'''How to get our own data into pytorch --> by using custom datasets'''

# 1. Getting Data --> data is subset of Food101 dataseet from pytorch
import requests
import zipfile
from pathlib import Path

# setup path to a data file 
data_path = Path("data/")
image_path = data_path/"pizza_steak_sushi"
# if image folder doesnt exist , download it and prepare it ...
if image_path.is_dir():
    # print(f"{image_path} dir already exist ....")
    pass
else:
    print("downloading.......")
    image_path.mkdir(parents=True , exist_ok=True)
    # downlaoding the zip file 
    with open(data_path/ 'pizza_steak_sushi' , 'wb') as f:
        req = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(req.content)

    # unzip file data 
    with zipfile.ZipFile(data_path/'pizza_steak_sushi.zip' , "r") as zip_ref:
        print('unzipping data')
        zip_ref.extractall(image_path)


# 2. data preparation and exploration 
import os
def walk_through_dir(dir_path):
    "walks through dir path returning its contents"
    for dirpath , dirnames , filenames in os.walk(dir_path):
        print(f"dirpath -- {dirpath}")
        print(f"dir names -- {len(dirnames)}")
        print(f"filenames -- {len(filenames)}")

# walk_through_dir(image_path)

# setup train and testing paths
train_dir = image_path/"train"
test_dir = image_path/"test"

# 2.1 visualising images 
import random 
from PIL import Image

# get all image path 
image_path_list = list(image_path.glob("*/*/*.jpg"))
# pick a random image 
random_image_path = random.choice(image_path_list)
# get image class(the name of the directory of the image stored in)
image_class = random_image_path.parent.stem
img = Image.open(random_image_path)
'''
print(f"image path--> {random_image_path}")
print(f"image belongs from--> {image_class}")'''

# 2.2 visualisation of image with matplotlib
import numpy as np
import matplotlib.pyplot as plt

# turn the image into the array
img_array = np.asarray(img)
# plot the image with matplotlib
# plt.imshow(img_array)
# plt.title(f"class --> {image_class} || shape --> {img_array.shape} --> (height , width , color channel)")
# plt.axis(False)

# 3. transforming data to the tensor 
from torch.utils.data import DataLoader
from torchvision import datasets , transforms

# 3.1 Transforming data with torchvision.transforms --> https://pytorch.org/vision/stable/transforms.html
data_transform = transforms.Compose([
    # resize our images to (64,64)
    transforms.Resize(size=(64,64)),
    # flip image randomly on the horizontal
    #try with different data argumentaton
    # turn the image into a torch tensor 
    transforms.ToTensor(), # --> onverts a PIL Image or numpy.ndarray (H x W x C)  to a torch tensor of shape (C x H x W)
])


def plot_transformed_images(image_paths , transformer , n = 3):
    "selects random image and transforms them then plots them"
    random_image_paths = random.sample(image_paths , k = n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig , ax = plt.subplots(nrows=1 , ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"orginal\nsize: {f.size}")
            ax[0].axis(False)

            # transform and plot image 
            transformed_image = transformer(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].axis(False)
            ax[1].set_title(f"transformed\nsize: {transformed_image.size}")

            fig.suptitle(f"class: {image_path.parent.stem}" , fontsize = 16)

# plot_transformed_images(image_path_list , data_transform , 3)
# plt.axis(False)
# plt.show()

# 4. 
# option -1 -->a) loading image data using `torchvision.datasets.ImageFolder`--> converts whole directory of images into the tensor format
train_data = datasets.ImageFolder(root=train_dir, # directory which contains the image
                                  transform=data_transform, # -->transform (callable, optional): A function/transform that takes in a PIL image
                                  target_transform=None ,# a transform for the label/targets
                                  )

test_data = datasets.ImageFolder(root=test_dir,  transform=data_transform,  target_transform= None)

class_name = train_data.classes # all classes of images as in folder
class_dict = train_data.class_to_idx # dictionary format of classes
image_tensor , label_no = train_data[0] 
# print(train_data[0]) # --> gives images tensor(of shape [3,64,64]) and label of it.

# Rearange the order of dimension
image_permute = image_tensor.permute(1,2,0)
# plt.imshow(image_permute )
# plt.show()

# b) --> turn image tensor to dataloaders(as done in prev classes)
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data , batch_size=32 , shuffle=True) # adds a batch numberr (converts 3D tensor to 4D tensor for nn.Conv3D)
test_dataloader = DataLoader(dataset=test_data , batch_size=32 , shuffle=False)  # --> [32 , 3 , 64,64]
img , label = next(iter(train_dataloader))
# print(img.shape) # --> shows the [batch_no , color_channel , height , width]


# option 2 --> Loading Image Data with a `Custom Dataset`ie creating our own dataset
import os
import pathlib
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple , Dict , List

# creating a helper function to get class names in a list and in dictionary format
def find_classes(dir: str) -> Tuple[List[str] , Dict[str, int]]:
    "finds the class folder names in a target directory"
    class_name_found = sorted(entry.name for entry in list(os.scandir(dir)) if entry.is_dir()) # -->  os.scandir is going to iterate over the class names in folder and list function is going to list them and sorted fn going to sort them in acen order

    if not class_name_found:
        raise FileNotFoundError('nahi milla mujhe bhaiya')
    # create a dictionary of index labels
    class_to_idx = {class_label : i for i , class_label in enumerate(class_name_found)}

    return (class_name_found , class_to_idx)

# Create a custom Dataset to replicate ImageFolder
class ImageFolderCustom(Dataset):
    def __init__(self , targ_dir:str ,transform = None ):
        self.paths= list(pathlib.Path(targ_dir).glob("*/*.jpg")) # listing every image path in targ_directory
        # setup transform
        self.transform = transform
        self.class_name , self.class_to_idx = find_classes(targ_dir)

    # creating a function to load images 
    def load_image(self , index:int) -> Image.Image:
        "opens an image via a path and returns it"
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self)-> int:
        "returns total number of samples"
        return(len(self.paths))

    def __getitem__(self, index:int) -> Tuple[torch.Tensor , int]:
        "returns a sample of data in --> (data: torch.Tensor , label:int)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # --> expects path in format: data_folder/class_name/images.jpg
        class_idx = self.class_to_idx[class_name]

    # transform if necceray
        if self.transform:
            return self.transform(img) , class_idx
        else:
            return img , class_idx

train_custom_dataset = ImageFolderCustom(train_dir , data_transform)
test_custom_dataset = ImageFolderCustom(test_dir , data_transform)

# print(f"class name {custom_dataloader.class_name} || img shape {custom_dataloader[0].shape} ")
# print(custom_dataloader.class_to_idx)
# print(custom_dataloader.class_name)
# print(custom_dataloader.__getitem__(0)[0].shape)

# displaying a raondom images 
def display_images( dataset: torch.utils.data.Dataset, display_shape:bool = True , class_name:List[str] = None , no_img :int = 10 ):
    "display images"
    if no_img > 10:
        no_img = 10
        display_shape = False
        print(f"no_img is too high to load resetting it to 10")

    rand_image_samples = random.sample(range(len(dataset)) , k=no_img)

    for i , targ_idx in enumerate(rand_image_samples):
        targ_image , targ_label = dataset[targ_idx]
        targ_image_adjust = targ_image.permute(1,2,0) #[C,H,W] --> [H,W,C]

        plt.subplot(1,no_img,i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")

        if class_name:
            title = f"Class Name {class_name[targ_label]}"
            if display_shape:
                title = title + f"\n shape {targ_image_adjust.shape}"
        plt.title(title)
# display_images(train_data , True , class_name , 4)

# now turning dataset into daat loader
train_dataloader_customs = DataLoader(dataset = train_custom_dataset , shuffle=True , batch_size=32)
test_dataloader_customs = DataLoader(dataset = test_custom_dataset , shuffle=False , batch_size=32)

# print(train_custom_dataset.class_name == train_data.classes)
# print(train_custom_dataset.class_to_idx == train_data.class_to_idx)

'''Model )--> TinyVGG'''
class TinyVGGModle_0(nn.Module):
    def __init__(self, input:int , output:int , hidden_units:int):
        super().__init__()
        
        self.Layer_stack1 = nn.Sequential(
            nn.Conv2d(in_channels=input , out_channels=hidden_units , kernel_size=3 , stride=1 , padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units , out_channels=hidden_units, kernel_size=3 , stride=1 , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2)
        )

        self.Layer_stack2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units , out_channels=hidden_units , kernel_size=3 , stride=1 , padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units , out_channels=hidden_units, kernel_size=3 , stride=1 , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16 , out_features = output)
        )

    def forward(self, x)-> torch.Tensor:
        return self.classifier(self.Layer_stack2(self.Layer_stack1(x)))

model_0 = TinyVGGModle_0(3 , len(train_data.classes) , 10)

image_batch , labell_batch = next(iter(train_dataloader))
# print(image_batch.shape)
# print(labell_batch)

y_logits = model_0(image_batch)
print(y_logits.shape)
'''Training and testing the model'''

from helper_function import accuracy_fn

def train_model(model: nn.Module , train_dataloader: torch.utils.data.DataLoader , loss_fn: nn.Module , optimizer: torch.optim.Optimizer , accuracy_fn):
    "train model and return its train loss and train accuracy respectively"
    model.train()
    train_loss , train_acc = 0,0

    for batch , (image , label) in enumerate(train_dataloader):
        train_logits = model(image)
        loss = loss_fn(train_logits , label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_prediction = torch.argmax(torch.softmax(train_logits, dim=1 ) , dim=1)
        train_acc += accuracy_fn(label , train_prediction)

    train_loss /=len(train_dataloader)
    train_acc /=len(train_dataloader)

    return train_loss , train_acc

def test_model(model: nn.Module, test_dataloader: torch.utils.data.DataLoader , loss_fn:nn.Module , accracy_fn):
    "test the model and returns its test loss and test accuracy respectively"
    model.eval()
    test_loss , test_acc = 0,0
    with torch.inference_mode():
        for batch , (image , label )in enumerate(test_dataloader):
            test_logits = model(image)
            test_loss += loss_fn(test_logits , label).item()
            test_prediction = torch.argmax(torch.softmax(test_logits , dim=1) , dim=1)
            test_acc += accracy_fn(label , test_prediction)

    return test_loss , test_acc

from tqdm.auto import tqdm

def train_and_test_model(model: nn.Module , train_dataloader: torch.utils.data.DataLoader , test_dataloader: torch.utils.data.DataLoader , loss_fn: nn.Module , optimizer: torch.optim.Optimizer , accuracy_fn , times_to_train:int):
    epochs = times_to_train
    results = {
        "train_loss" : [],
        "train_acc": [],
        "test_loss" : [],
        "test_acc" : []
    }

    for epoch in tqdm(range(epochs)):
        train_loss , train_acc = train_model(model=model , train_dataloader=train_dataloader , loss_fn=loss_fn , optimizer=optimizer , accuracy_fn=accuracy_fn)
        test_loss , test_acc = test_model(model=model , test_dataloader=test_dataloader , loss_fn=loss_fn , accracy_fn=accuracy_fn)

        print(f"epoch : {epoch} ............")
        print(f"train loss : {train_loss} ")
        print(f"train accuracy : {train_acc} ")
        print(f"test loss : {test_loss} ")
        print(f"test accuracy : {test_acc} ")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

loss_fn = nn.modules.CrossEntropyLoss()
optimizer_0 = torch.optim.Adam(params=model_0.parameters() , lr=0.01)

from timeit import Timer
start = Timer()
# training_results = train_and_test_model(model_0 , train_dataloader ,test_dataloader , loss_fn , optimizer_0 , accuracy_fn , 3 )
end = Timer()

'''Plot loss curve --> a great way of tracking model's progress over time'''

# plot_loss_curves(results=training_results)

'''Model 1 --> TinyVGG with data argumentation'''
# building a transformer 
train_argumentation_transformer = transforms.Compose(
   [ transforms.Resize(size=(64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()]
)

test_argumentation_transformer = transforms.Compose(
    [transforms.Resize(size=(64,64)),
    transforms.ToTensor()]
)

# building the datasets 
train_data_arg = datasets.ImageFolder(root=train_dir , transform=train_argumentation_transformer , target_transform=None)
test_data_arg = datasets.ImageFolder(root=test_dir , transform=test_argumentation_transformer , target_transform=None)

# building the dataloader 
train_dataloader_arg = DataLoader(dataset=train_data_arg , shuffle=True , batch_size=32)
test_dataloader_arg = DataLoader(dataset=test_data_arg , shuffle=False , batch_size=32)

model_1 = TinyVGGModle_0(3 , len(train_data_arg.classes) , 10)
optimizer = torch.optim.Adam(params=model_1.parameters() , lr = 0.01)

# training and test our model now with data argumentation 

training_results_with_arg = train_and_test_model(model=model_1 , train_dataloader=train_dataloader_arg , test_dataloader=test_dataloader_arg , loss_fn=loss_fn , optimizer=optimizer , accuracy_fn=accuracy_fn , times_to_train=3)

# print(training_results_with_arg)
# plot_loss_curves(results=training_results_with_arg)
# plt.show()
'''
import pandas as pd
model_0_df = pd.DataFrame(training_results)
model_1_df = pd.DataFrame(training_results_with_arg)

# Setup a plot 
plt.figure(figsize=(15, 10))

# Get number of epochs
epochs = range(len(model_0_df))

# Plot train loss
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot test loss
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot train accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

# Plot test accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()
'''

custom_image_path = data_path/"pizza.jpeg"
if not custom_image_path.is_file():
    with open(custom_image_path , "wb") as f:
        reqs = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        f.write(reqs.content)
else:
    pass

custom_image_uint8 = torchvision.io.read_image(str(custom_image_path)) # --> converts image into a tensor 
custom_image_uint8_type = custom_image_uint8.type(torch.float32)/255 # --> converting image into a float32 type and tensor values to between 0 and 1 by dividing
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
])

custom_image_transformed = custom_image_transform(custom_image_uint8_type)

# making prediction 
model_1.eval()
with torch.inference_mode():
    pred = model_1(custom_image_transformed.unsqueeze(dim=0))

pred_label = torch.argmax(torch.softmax(pred , dim=1) , dim=1)
print(class_name[pred_label])

def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image)
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

pred_and_plot_image(model_1 , custom_image_path , class_name , custom_image_transform)
plt.show()


