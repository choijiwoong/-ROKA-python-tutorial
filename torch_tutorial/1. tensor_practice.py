import torch
import numpy as np

#Make tensor(5명 학급 2개의 성적)
 #list data
score_list=[[80,90,39,85,29],[10,9,100,60,88]]
score_tensor=torch.tensor(score_list)
 #nparray data
score_nparray=np.array(score_list)#make numpy array by list data
score_tensor=torch.from_numpy(score_nparray)

#Make tensor by like- method; keep property only
 #torch.ones_like
print('\n\nMake tensor by like-method')
data_ones=torch.ones_like(score_tensor)#make ones tensor that's property depends score_tensor
print(f"Ones_tensor like score_tensor: {data_ones}\n")
 #torch.rand_like
data_rand=torch.rand_like(score_tensor, dtype=torch.float)#잘 모르겠는데 명시적으로 score_tensor의 속성을 말해주는건가 암시적변환막으려고..? 일단 중요해보이진 않고 assert처럼 test추가느낌..
print(f"Random Tensor: {data_rand}\n")

#Make tensor with shape; information of dimention
print('\n\nMake tensor by shape')
shape=(3,2,5)#3학년 2반 5명정보까지 수용가능한 shape
rand_tensor=torch.rand(shape)#shape정보대로 tensor생성
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(rand_tensor,'\n',ones_tensor,'\n',zeros_tensor)
 #example
print("score_tensor+randvalue=",score_tensor+(rand_tensor*10-rand_tensor*10%1))#add score_tensor to rand value(0~9)
 #concat tensor
concat_tensor=torch.cat([rand_tensor, ones_tensor, zeros_tensor], dim=1)#shape(3,2,5)중에 index1 dimention, 2따리(반) 기준 concat
print("concat rand_tensor, ones_tensor, zeros_tensor: ", concat_tensor)

#Attrubution of tensor
print('\n\nGet attribution of score_tensor')
print(f"Shape of score_tensor: {score_tensor.shape}")#get dimention information by .shape
print(f"Date type of score_tensor: {score_tensor.dtype}")#get data type by .dtype
if torch.cuda.is_available():#check gpu
    score_tensor=score_tensor.to('cuda')
print(f"Device tensor is stored on: {score_tensor.device}\n")#get location of that tensor (cuda or cpu) by .device

#Operation of tensor
print('first row of score_tensor: ', score_tensor[0])
print('first cloumn of score_tensor: ', score_tensor[:,0])
print('Last column of score_tensor: ', score_tensor[:,-1])

score_tensor[:,2]=50#modifing specific column
print('Modified specific column score_tensor: ',score_tensor)

#arthmetic operations
 #matrix_multiplication
score2_tensor=score_tensor*score_tensor#way1
score2_tensor=score_tensor.matmul(score_tensor.T)#.T make tensor
tensor_storage=torch.rand_like(score_tensor, dtype=float)#Make tensor that's saving same property with score_tensor
torch.mul(score_tensor, score_tensor, out=tensor_storage)#multiple to tensor_storage
print("score_tensor * score_tensor= ", tensor_storage)

score_sum=score_tensor.sum()
score_sum_item=score_sum.item()#convert tensor to item(int, float, ...etc)
print(f"sum of score is {score_sum_item} and it's data type is {type(score_sum_item)}\n")

#convert with numpy array
score_nparray=score_tensor.numpy()
print(score_nparray)

print(torch.from_numpy(score_nparray).add(1))
