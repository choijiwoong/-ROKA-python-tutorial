"""
 Hyperparameter는 모델 최적화 과정을 제어하는 매개변수이다. 이들은 모델 학습과 convergence rate에 영향을 미칠 수 있으며, 주로 아래의 하이퍼 파라미터를 정의한다.
epoch(데이터셋 반복횟수), batch_size(매개변수 갱신 전 신경망을 통해 전파된 데이터 샘플의 수),
learning_rate(각 batch | epoch에서 모델의 매개변수를 조절하는 비율_작으면 학습속도 느리고, 크면 예측할 수 없는 동작 가능성 발생가능성 상승)
learning_rate=1e-3
batch_size=64
epochs=5
 Optimization Loop는 hyperparameter설정 뒤 가능하며, 하나의 epoch는 train lop(dataset을 iterate하고 최적의 매개변수로 수렴), validataion/test loop(test dataset iterate)로 구성된다.
trainning loop에서는 계산한 예측과 정답(label)을 비교하여 loss를 도출하는 lossfunction을 통해 degree of dissimilarity를 측정한다.
일반적인 loppfunction으로는 regression task에 사용되는 nn.MSELoss(Mean Square Error), classification에 사용되는 nn.NLLoss(Negative Log Likelihood), nn.LogSoftmax와 nn.NLLLoss를 합친 nn.CrossEntropyLoss가 있다.
loss_fn=nn.CrossEntropyLoss()
 Optimizer는 모델의 Error를 줄이기 위한 모델 매개변수 조정 과정을 정의하며(SGD_Stochastic Gradient Descent, ADAM, RMSProp) 모든 logic(최적화절차)는 optimizer객체에 encapsulate된다.
학습단계(loop)의 최적화는 세단계로, optimizer.zero_grad()로 모델 매개변수의 변화도를 재설정하며(변화도가 가산되기에 중복계산을 막기위해 반복마다 0설정),
loss.backwards()로 예측손실을 역전파하며, optimizer.step()으로 이 변화도로 매개변수를 조정하는 3단계로 구성된다.
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data=datasets.FashionMNIST(#Get dataset for trainning
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)
test_data=datasets.FashionMNIST(#Get dataset for test
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)
train_dataloader=DataLoader(training_data, batch_size=64)#make loader object each
test_dataloader=DataLoader(test_data, batch_size=64)

class NeralNetwork(nn.Module):#Make network by inheritating nn.Module
    def __init__(self):
        super(NeuralNetwork, self).__init__()#super constructor call
        self.flatten=nn.Flatten()#set flatten funtor
        self.linear_relu_stack=nn.Sequential(#set sequential
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x):#define work in forwarding
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits#[-inf, +inf] will be passed to softmax, lossfuncion(both function is Cross_entropyLoss)
model=NeuralNetwork()#make model object

def train_loop(dataloaer, model, loss_fn, optimizer):#define loop for trainning
    size=len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):#each element of dataloader. (p.s enumerate makes tuple of index, element
        pred=model(X)#pass X to model, get predict
        loss=loss_fn(pred,y)#get loss

        optimizer.zero_grad()#initialization for consistent backwarding
        loss.backward()#backpropagate loss
        optimizer.step()#modifing parameter by loss

        if batch%100==0:#print state of processing per 100
            loss, current=loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
def test_loop(dataloader, model, loss_fn):#no optimizer because test doesn't use backpropagation
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, current=0, 0

    with torch.no_grad():#protect useless calculation&optimization for doing gradient because it's test
        for X, y in dataloader:
            pred=model(X)#get predict
            test_loss+=loss_fn(pred, y).item()#add loss value
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()#잘모름
    test_loss/=num_batches
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
loss_fn=nn.CrossEntroypyLoss()#make loss_fn
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)#make optimizer

epochs=10
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
