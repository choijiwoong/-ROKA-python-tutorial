"""
LeNet-5의 간단한 버전은, 컨볼루션 레이어C1은 학습한 기능을 검색하여 맵을 출력한다. 이 맵은 활성화 맵S2에서 다운샘플링하며 또다른 컨볼루션 레이어
C3이 이 활성화 맵을 스캔하고, S4은 다운샘플링된 기능조합의 공간적 위치를 설명하는 활성화 맵을 표시한다. 마지막으로 F5, F6 및 OUTPUT끝의
완전 연결 계층이 최종 활성화 맵을 가져와 10자리를 나타내는 10개의 bin중 하나로 분류하는 분류기이다. 뭔소리일까 해보며 이해해보자
 컨볼루션 레이어는 입력 이미지는 특정 Filter(kernal)을 이용하여 이미지의 특징들을 추출 후, FeatureMap으로 생성한다. 쉽게 내 방식대로 말하면
이미지 정보들을 Filter단위로 나누어 합성곱결과로 FeatureMap을 만든다. 이때 입력 이미지가 작아지는 것을 막기 위해 가장자리를 특정 값으로 채우는
Padding기법을 사용한다. 필터에 적용하는 활성 함수로는 STEP, Sigmoid, TanH, Relu등이 있다.이 다음에 주로 Polling layer을
이용하는데, 범위 내의 픽셀 중 대표값을 추출하는 방식으로 특징을 추출하며 Max Pooling, Average Pooling, Min Polling 세가지 방식으로 대표값 추출이 가능하다.
고로 컨볼루션 레이어는 패딩을 통해 범위 내 대표값을 추출하기 때문에 이미지의 크기가 작아지지만 형태분석이 수월해지는 장점이 있다.
 이를 통해 만들어지는 이미지분류 인공신경망을 Fully Connected Layer(FCL)이라고 하며, FCL이 픽셀간의 연관성을 유지못한다는 단점을 보완,
컨볼루션 레이어와 풀링 레이어를 FCL앞에 구성한 CNN을 개발한다.
 CNN의 기본 구성은 3개의 컨볼루션 레이어와 2개의 풀링 레이어, 1개의 FCL로 구성된다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F#for activation function

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #Kernal; 1 input image channel(black & white), 6 output channels, 3x3 square convolution
        self.conv1=nn.Conv2d(1, 6, 3)#input1, output6, size3 _2D convolution return tensor
        self.conv2=nn.Conv2d(6, 16, 3)#input6, output16, size3
        #affine operation; y=Wx+b
        self.fc1=nn.Linear(16*6*6, 120)#16_output of conv2, 6*6_dimension of image
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)#classify one of 10bins
        
    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))#Max pooling over (2,2) window with conv1*****
        x=F.max_pool2d(F.relu(self.conv2(x)), 2)#Max pooling over (2) window with conv2
        x=x.view(-1, self.num_flat_features(x))#change shape of tensor to (auto, self.num_flat_features(x))#ㄹㅇ 전체 크기
        x=F.relu(self.fc1(x))#use relu as activate function
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features#get num of flat features in many dimention of tensor

net=LeNet()
print(net)

input=torch.rand(1,1,32,32)#dummy input
print('\nImage batch shape:')
print(input.shape)

output=net(input)#(not yet learned!)
print('\nRaw output')
print(output)
print(output.shape)
