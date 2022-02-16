import torch

#basic of tensor
z=torch.zeros(5,3)#shape(5,3)
print(z)
print(z.dtype)#default is float32

i=torch.ones((5,3), dtype=torch.int16)#explicit datatype
print(i)#if we change dtype from default value, tensor report it's type


#random tensor with manual_seed
torch.manual_seed(1729)
r1=torch.rand(2,2)
print('A random tensor: ')
print(r1)

r2=torch.rand(2,2)
print('\nA differenct random tensor: ')
print(r2)

torch.manual_seed(1729)#same seed with r1's. 종종 결과를 재현하기 위해 특정시드를 사용하여 같은 결과를 도출시키기도한다.
r3=torch.rand(2,2)
print('\nShould match r1: ')
print(r3)


#operations to tensor
ones=torch.ones(2,3)
print(ones)
twos=torch.ones(2,3)*2
print(twos)
threes=ones+twos
print(threes)
print(threes.shape)

r1=torch.rand(2,3)
r2=torch.rand(3,2)
try:
    r3=r1+r2#runtime error!_differnet shape!!!
except RuntimeError as e:
    print("Error occur! trace: ",e)

#sample of small operation on tensor
r=(torch.rand(2,2)-0.5)*2#[0,+1)] [-0.5, +0.5)] [-1,+1]
print('A random matrix, r:')
print(r)

print('\nAbsolute value of r:')
print(torch.abs(r))#[0,1]

print('\nInverse sine of r:')
print(torch.asin(r))#can calculate because r's range is [-1,+1]

print('\nDeterminate of r:')#특별한 계산식에 따라 행렬의 원소들을 대입하여 얻은 결과값
print(torch.det(r))
print('\nSingular value decomposition of r:')#직교하는 벡터 집합에 대하여, 선형변화후에 크기가 변할지라도 여전히 직교하는 집합들
print(torch.svd(r))#이를 이용하여 임의의 행렬을 정보량에 따라 여러 layer로 쪼개어 생각할 수 있다. A=U sigma V transposition 이미지 압축&복원등에 사용

print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))
