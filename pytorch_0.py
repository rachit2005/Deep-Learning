# read this --> https://www.learnpytorch.io/00_pytorch_fundamentals/

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creating tensors
# 1) --> scalar --> scalar has zero dimension
scaler = torch.tensor(7) 
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.

# print(scaler)
# print(scaler.ndimension()) #shows dimension of the tensor
# print(scaler.item()) #give only 0th dimension tensor as integer

# 2) --> vector -->vector has 1 dimention

vector = torch.tensor([7,7]) 
# print(vector)
# print(vector.ndimension())
# print(vector.shape) #shows the shape/rectangle size of the tensor



# 3) --> matrix --> matrix has 2 dimension

matrix =torch.tensor([[7,8 , 3] , 
                      [3,4,4],   #creating a matrix of 2x3
                      [4,5,7]])

# you can slice/get item using index adressing as well
# print(matrix)
# print(matrix[0]) # gives [7,8,3]
# # dimension is determined by the number sq brakets inside it
# print(matrix.ndimension())  # as it has two so its dimension is 2
# print(matrix.shape) #shows shape as [columns , row]

# 4) tensor --> 
tensor = torch.tensor([[[2,3] , 
                        [3,4] , #its dimension is 3 
                        [4,5]]])
# print(tensor)
# print(tensor.ndimension())
# print(tensor.shape)


advanced_tensor = torch.tensor([[[[2,4] ,
                                  [6,7],
                                  [4,5]]]]) #dimension is 4
# print(advanced_tensor)
# print(advanced_tensor.ndimension())
# print(advanced_tensor.shape) #shape is [1,1,3,2]

# why use random tensor --> Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers...

# Create a random tensor of size (rows , columns)
random_tensor = torch.rand(size=(3, 4))
# print(random_tensor, random_tensor.dtype)

# Create a random tensor of size (dimension/no of matrices , rows , columns)
random_image_size_tensor = torch.rand(size=(4, 3, 4)) #for imagess --> heights, width , color channels(rgb)
# print(random_image_size_tensor, random_image_size_tensor.ndim , random_image_size_tensor.shape)

# Create a tensor of all zeros
zeros = torch.zeros(size=(3,3, 4))
# print(zeros, zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(2,3,4))
# print(ones)

# create a range of tensor 
range_tensor = torch.arange(0 , 10) 
print(f"Creating a range of tensor --> {range_tensor}")

# creating tensor like 
range_tensor_like = torch.zeros_like(range_tensor) #Returns a tensor filled with the scalar value 0, with the same size as input 'tensor'
print(f"Creating a Tensor using torch.zero_like(input = tensor) --> {range_tensor_like}")

'''****************************************** Getting information from tensors ************************************'''
info_tensor = torch.rand(3,3)
# print(info_tensor.shape)
# print(info_tensor.dtype)
# print(info_tensor.device) # gives the devisce in which the tensor is stored

'''**********************************    TENSOR OPERATIONS   **********************************************************'''

tensor_op = torch.tensor([[1,2,3] , [4,5,6]])
# print(tensor_op + 10) #simply adds the 10 in tensor of any dimension 

# print(tensor_op * 10) #simply multiplies the 10 in tensor of any dimension 
# print(torch.multiply(tensor_op , 10)) #simply multiplies the 10 in tensor of any dimension 
# both above code is same 

# you can also reassigns it 

tensor_op = tensor_op + 10
# print(tensor_op)


#********************************************* MATRIX MULTIPLICATION *********************************************************************
# element-by-element wise multiply
tensor1 = torch.tensor([1,2,3])
tensor2 = torch.tensor([4,5,6])
mul_tensor = torch.multiply(tensor1 , tensor2)
# print(mul_tensor ,"same result", tensor2*tensor1)
# print(mul_tensor)

# matrix multiplication
# print(torch.matmul(tensor1 , tensor2))

# using for loop --> not recommended as this is slower
# value = 0
# for i in range(len(tensor1)):
#     value += tensor1[i] * tensor2[i]

# print(value)

# taking transpose of matrix 
matrix2 = torch.tensor([[1,2] , [3,4] , [6,7]])
# print(matrix2)
# print(matrix2.T) #transpose of matrix2


# Finding the min, max, mean, sum, etc
# Create a tensor

x = torch.arange(0, 100, 10) #tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# print(f"Minimum: {x.min()}")
# print(f"Maximum: {x.max()}")
# # print(f"Mean: {x.mean()}") # this will error
# print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
# print(f"Sum: {x.sum()}")

# print(f"max at: {x.argmax()}")
# print(f"min at : {x.argmin()}")

# reshaping stacking squeezing and unsqueezing
y = torch.arange(1,9) #number of elements is 8
# print(y , y.shape)

# add an extra dimension
y_reshaped = y.reshape(1,8) #added a single dimension
y_reshaped2 = y.reshape(2,4) # you can add number of dimension as (row , columns) , if rows*columns = total number of elements in a tensor(2*4 = 8)
# print(y_reshaped)
# print(y_reshaped2)

z = y_reshaped.view(2,4)
# print(z)
# shows all rows and changes 3rd column to 5
# changing z changes y_reshaped
z[:,3] = 5
# print(z[:,:4])

# stacking tensors on top of each other
# x = tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# try changing dim to dim=1 and see what happens
x_stack = torch.stack([x,x,x,x]) 
x_stack2 = torch.vstack([x,x,x,x]) 
x_stackh = torch.hstack([x,x,x,x]) 
x_stackc = torch.column_stack([x,x,x,x]) 
# print(x_stack)
# print(x_stack2)
# print(x_stackh)
# print(x_stackc)

# torch.squeeze() --> removes all single dimensions from a target tensor 
# print(y_reshaped)
# print(y_reshaped.squeeze())
# print(y_reshaped.unsqueeze(dim=0).shape) # Add an extra dimension with unsqueeze on the dim = index of shape list

# torch.permute()--> rearranges the dimensions of a target tensor in a specified order
x_orginal = torch.rand(size=(4,4,3)) # height , width , color --> (0,1,2)
x_permute = x_orginal.permute(2,0,1) # rearranges as (color , height , weight)
# print(x_orginal)
# print(x_permute)

x2 = torch.arange(1,10).reshape(1,3,3)
# print(x2)
# print(x2[0,0,:2])

# NumPy array to tensor
array = np.arange(1.0, 8.0)
tensor_numpy = torch.from_numpy(array).reshape(7,1)
# print(array)
# print(tensor_numpy)

# Tensor to NumPy array
numpy_tensor = tensor_numpy.numpy() # will be dtype=float32 unless changed || converts tensor to numpy array of same shape
print(numpy_tensor)
print( tensor_numpy)