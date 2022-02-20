"""
torch.nn.Module, torch.nn.Parameter등 pytorch에서 딥러닝 네트워크를 구축하는 도구들을 지원한다.
Linear은 분류기 모델에 자주 사용되며,
 컨볼루션 레이어는 높은 수준의 공간 상관관계가 있는 데이터를 처리하기위해 구축되어 컴퓨터 비전에 일반적으로 사용된다.
NLP또한 단어의 즉각적인 컨텍스트가 문장의 의미(상관관계)에 영향을 미칠 수 있기에 사용된다.
 RNN은 자연어 문장 등의 순차 데이터에 사용되며, 일종의 메모리 역활을 하는 hidden state(자신에게 보내는 값)를 유지하므로서 이를 수행한다.
RNN의 내부 구조, 변형(LSTM, GRU)는 복잡하지만 아래의 예시는 LSTM기반 품사 tagger이다. RNN은 정보와 정보사용시점거리가 멀 경우 역전파 gradient가 줄어드는 문제
vanishing gradient problem이 있는데, 이를 극복하기위해 RNN의 hidden state에 cell-state를 추가한 것니 LSTM이다.
 Transformers는 BERT(언어모델로 입력15%를 랜덤masking후 예측하게하는모델)을 이용하여 NLP최신의 다목적 네트워크이다.
 
"""
import torch

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1=torch.nn.Linear(100,200)
        self.activation=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(200,10)
        self.softmax=torch.nn.Softmax()

    def forward(self, x):
        x=self.linear1(x)
        x=self.activation(x)
        x=self.linear2(x)
        x=self.softmax(x)
        return x

tinymodel=TinyModel()

print('The model: ')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params: ')
for param in tinymodel.parameters():#access to parameters of tinymodel
    print(param)

print('\n\nLayer params: ')
for param in tinymodel.linear2.parameters():#access to parameters of linear2 of tinymodel
    print(param)

#[Common Layer Types]
 #linear_classifier model
lin=torch.nn.Linear(3,2)
x=torch.rand(1,3)
print('\n\n[Common Layer Types_Linear]\nInput:')
print(x)

print('\nWeight ans Bias parameters: ')
for param in lin.parameters():#parameter's autograd & tensor's autograd are different.
    print(param)#we can get output vector by multipling weight matrix and adding bias

 #convolutional layers
print('\n\n[Common Layer Types_Convolution]')
import torch.nn.functional as F

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1=torch.nn.Conv2d(1,6,5)#number of input channel(color_1~3), number of output functions, size of window or kernal
        self.conv2=torch.nn.Conv2d(6,16,3)

        self.fc1=torch.nn.Linear(16*6*6, 120)
        self.fc2=torch.nn.Linear(120,84)
        self.fc3=torch.nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))#conv's output is activated map. conv1_6(feature)*28*28(weight, height_5픽셀창은 28개 유효위치)
        #2x2즉 4개 셀의 최대값을 pooling한다. 그 결과 6x14x14의 저해상도 활성화 맵이 제공된다.
        x=F.max_pool2d(F.relu(self.conv2(x)), 2)#conv2는 16x12x12를 출력하고 pooling에 의해 16x6x6으로 축소된다.
        x=x.view(-1, self.num_flat_features(x))#flatten_다음 레이어에서 사용될 수 있도록(선형) 16*6*6요소 벡터로 모양을 바꾼다.
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self, x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features

 #convolutional layers
print('\n\n[Common Layer Types_Recurrent]\n')
class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):#size of embedding for input word, memory size of LSTM, number of voca(one-hot), tag of output set
        super(LSTMTagger, self).__init__()
        self.hidden_dim=hidden_dim
        self.word_embeddings=torch.nn.Embedding(vocab_size, embedding_dim)

        self.lstm=torch.nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag=torch.nn.Linear(hidden_dim, tagset_size)#hidden state space to tag space

    def forward(self, sentence):
        embeds=self.word_embeddings(sentence)#noe-hot vector의 인덱스로 표현되는 단어가 있는 문장이 input. 이를 차원공간으로 매핑.
        lstm_out, _=self.lstm(embeds.view(len(sentence), 1, -1))#
        tag_space=self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores=F.log_softmax(tag_space, dim=1)
        return tag_scores


#[Other Layers and Functions]
 #data manipulation layers
print('\n\n[Other Layers and Functions]\n')
my_tensor=torch.rand(1,6,6)
print("before maxpooling:", my_tensor)

maxpool_layer=torch.nn.MaxPool2d(3)#2차원 데이터를 3개씩
print("after maxpooling:", maxpool_layer(my_tensor),end='\n\n')

 #normalization layers_한계층의 출력을 다른 계층에 공급하기 전에 중앙에 다시 놓고 정규화하기에 더 높은 학습률을 사용할 수 있다.
my_tensor=torch.rand(1,4,4)*20+5
print("my_tensor: ", my_tensor)
print("my_tensor.mean(): ", my_tensor.mean())#15주위

norm_layer=torch.nn.BatchNorm1d(4)
normed_tensor=norm_layer(my_tensor)
print("normed_tensor:", normed_tensor)
print("normed_tensor.mean():",normed_tensor.mean(),end='\n\n')#0주위

 #dropout layer_더 적은 데이터로 추론을 실행하도록 만든다.
my_tensor=torch.rand(1,4,4)
print("before dropout: ",my_tensor)
dropout=torch.nn.Dropout(p=0.4)
print("dropout 1:", dropout(my_tensor))
print("dropout 2:", dropout(my_tensor))#more smaller
