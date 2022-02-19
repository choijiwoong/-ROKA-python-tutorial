import torch

#Autograd in learning
BATCH_SIZE=16
DIM_IN=1000
HIDDEN_SIZE=100
DIM_OUT=10

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1=torch.nn.Linear(1000,100)
        self.relu=torch.nn.ReLU()
        self.layer2=torch.nn.Linear(100,10)#no gradient yet!

    def forward(self, x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        return x
    
some_input=torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)#[16, 1000]
ideal_output=torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)#[16,10]

model=TinyModel()

print("Before backward_weight : ",model.layer2.weight[0][0:10])#sample slice_Show weight of layer2
print("Before backward_grad: ", model.layer2.weight.grad, end='\n\n')#None_no calculated gradient yet_Show gradient of layer2
#쉽게말해 모델들을 만들면서 gradient를 따로 정의 안하니 당연히 grad함수가 없는데
optimizer=torch.optim.SGD(model.parameters(), lr=0.001)#make optimizer by using model's parameters
prediction=model(some_input)
loss=(ideal_output-prediction).pow(2).sum()#we can use Euclidean distance as simple loss function
print("result of loss function: ",loss, end='\n\n')
#간단하게 유클리드 두점사이공식으로 결과랑 prediction차 제곱을 loss function으로 만들고
loss.backward()
print("After backward_weight: ",model.layer2.weight[0][0:10])#Show weight of layer2 after backwarding loss
print("After backward_grad: ", model.layer2.weight.grad[0][0:10], end='\n\n')#gradient is calculated now, but weight is not changed yer because we don't execute optimizer yet
#backward하니 당연히 gradient가 계산되고
optimizer.step()#update weights_Backpropagation
print("After backpropagation_weight: ", model.layer2.weight[0][0:10])#weight is updated!
print("After backpropagation_grad: ", model.layer2.weight.grad[0][0:10],end='\n\n')
#그 결과로 backpropagation하니 weight가 updata된 것을 확인할 수 있음. 각 단계별 차이를 가시화한 건 좋은데 너무 난잡해서 정리해둠. 총총
#아래는 정말로 친절하게(덕분에 햇갈림) optimizer.step()없으면 value of update가 누적된다는 것을 보려주기 위해 backward만 한 뒤 보여준거고 거기에 zero_grad역활까지 가시화한거.
for i in range(0,5):
    prediction=model(some_input)
    loss=(ideal_output - prediction).pow(2).sum()#use simple loss function too
    loss.backward()

print("Before zero_grad()_grad: ",model.layer2.weight.grad[0][0:10])
optimizer.zero_grad()
print("After zero_grad()_grad: ",model.layer2.weight.grad[0][0:10], end='\n\n')
