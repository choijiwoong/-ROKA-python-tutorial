#torch.nn에서 신경망 구성요소들을 제공하며, pytorch의 모든 모듈들은 nn.Module의 subclass들이다.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()#call basic constructor(super is using based's functions without method override)
        self.flatten=nn.Flatten()#make one dimention
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28, 512),#input_dim, output_dim
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    def forward(self, x):
        x=self.flatten(x)#make flatten
        logits=self.linear_relu_stack(x)#throw(pass) to network
        return logits#about percent like sigmoid. sigmoid makes percent, logit get that percent and make real number
model=NeuralNetwork().to(device)#make instance
print(model)#check model's structure

X=torch.rand(1, 28, 28, device=device)#make random tensor
print("data for test: ",X)
logits=model(X)#get logits after passing model. (Do not call model.forward() directly!) get 10-dimention tensor that has raw predicated value orderby each class
pred_probab=nn.Softmax(dim=1)(logits)#get predicated percent by passing raw predicated percent to softmax
print("pred_probab: ",pred_probab)
y_pred=pred_probab.argmax(1)#softmax결과 중 가장 높은 확률을 가진 index반환
print(f"Predicated class: {y_pred}")
#softmax는 tensor확률[-inf+inf], 을 numeric확률로 변경시켜준다. 그 값을 기반으로 one-hot vector표현이 가능하다. 이 원핫벡터와 실제 raw 예측값과 오차를 비교하여 backpropagation이 가능하다.

"""
마지막 레이어에 activation function이 없다면 그 값이 [-inf, +inf]인데, sigmoid function을 통해 [0,1]y값으로 매칭하여 확률의 형태로 만드는 것이다.
softmax에 대해 알기위해서는 sigmoid, logit, softmax의 상관관계를 알아야한다. 우선 확률 값을 [0,1]범주외에도 성공확률/실패확률로 나타낼 수 있고 이를 Odds라고 한다.
이 odd값에 log를 취해 logit이라는 개념을 만들었는데, 이는 함수의 증감형태를 유지, 극점의 위치를 유지, 곱|나눗셈을 linear combination으로 나타낼 수 있는 듯 여러 장점이 있다.
즉, logit함수는 [0,1] 확률값을 다시 [-inf,+inf]로 변경하는 역활로 logit와 sigmoid(logistic)은 역함수 관계이다.
 softmax함수는 이 [-inf,+inf]를 [0,1]로 만드는 sigmoid의 일반형으로 신경망 연산을 하며 사용된 logit값들을 확률로 변경해주는 것이다.

"""

#layer
print('\n\n')
input_image=torch.rand(3,28,28)
print(input_image.size())

#flatten
flatten=nn.Flatten()#make flatten instance
flat_image=flatten(input_image)#flatten image
print(flat_image.size())

#linear
layer1=nn.Linear(in_features=28*28, out_features=20)#Linear apply linear transformation to input by using weight & bias
hidden1=layer1(flat_image)#flat_image's shape (3, 28*28). do Linear so shape(3,20)
print(hidden1.size())

#relu
print(f"Before ReLU: {hidden1}\n\n")
hidden1=nn.ReLU()(hidden1)#ReLU maintain positive, make negative to zero
print(f"After ReLU: {hidden1}")

#sequential
seq_modules=nn.Sequential(
    flatten,#nn.Flatten
    layer1,#Linear(28*28,20)
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image=torch.rand(3,28,28)
logits=seq_modules(input_image)
print('\nlogits value after passing seq_modules: ',logits)

#softmax
softmax=nn.Softmax(dim=1)#2번째 차원에 softmax를 적용한다.
pred_probab=softmax(logits)
print('\nvalue of pred_probab after softmax: ',pred_probab)

#model parameter
print("\nModel structure: ", model, '\n\n')
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")
"""
nn.Module을 상속하여 model을 만들었기에 모델 객체 내부의 모든 필드들이 자동으로 track되어, parameters, names_parameters메소드로 모든 매개변수에 접근할 수 있다.
각 매개변수들을 순회하며 매개변수의 크기와 값을 출력한다. 이 때 sequence_module로 들어간 각 module들에 전달된 매개변수의 값을 알 수 있기에 backpropagation에 활용될 듯 하다.
무튼 중점은, 모든 module들의 매개변수 값을 실시간으로 확인할 수 있다는 정도. size도 마찬가지.
"""
