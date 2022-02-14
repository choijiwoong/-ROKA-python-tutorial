import torch
import numpy as np

#[1. Inistialization of tensor]
data=[[1,2],[3,4]]
x_data=torch.tensor(data)#Make tensor directly by data

np_array=np.array(data)#Make tensor by nparray
x_np=torch.from_numpy(np_array)

x_ones=torch.ones_like(x_data)#x_data의 속성을 유지하는 tensor
print(f"Ones Tensor: \n {x_ones} \n")
x_rand=torch.rand_like(x_data, dtype=torch.float)#x_data의 속성을 덮어씌우는 tensor
print(f"Random Tensor: \n{x_rand} \n")

shape=(2,3,)#[2][3]. shape is tuple that points dimemsion of tensor.
rand_tensor=torch.rand(shape)#rand값float으로
ones_tensor=torch.ones(shape)#1로
zeros_tensor=torch.zeros(shape)#0으로
print(f"Random Tensor: \n {rand_tensor}\n")
print(f"Ones Tensor: \n{ones_tensor} \n")
print(f"Zeros Tensor: \n{zeros_tensor}")

#[2. Attrubution of tensor]
tensor=torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#[3. Operation of tensor]
if torch.cuda.is_available():
    tensor=tensor.to('cuda')#can GPU, change GPU

tensor=torch.ones(4,4)
print('First row: ', tensor[0])#normal slicing as Numpy
print('First column: ', tensor[:,0])
print('Last column: ',tensor[..., -1])
tensor[:,1]=0
print(tensor)

t1=torch.cat([tensor, tensor, tensor], dim=1)#concat tensor
print(t1)

y1=tensor @ tensor.T#arthmetic operations(matrix multiplication). same result.
y2=tensor.matmul(tensor.T)
y3=torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y1,"\n", y2,"\n", y3)

z1=tensor*tensor#arthmetic operations(element-wise product). same result.
z2=tensor.mul(tensor)
z3=torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

