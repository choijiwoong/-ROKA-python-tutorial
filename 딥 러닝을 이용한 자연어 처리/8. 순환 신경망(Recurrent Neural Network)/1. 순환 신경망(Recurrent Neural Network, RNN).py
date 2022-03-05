"""입력과 출력을 시퀀스 단위로 처리하는 시퀀스 모델이다.(시퀀스기에 Recurrent) RNN은 가장 기본적인 인공 신경망 시퀀스 모델로 다른 모델로는 LSTM, GRU등이 있다.
(재귀 신경망, 순환신경망과는 전혀다른 개념이다)

    1. 순환 신경망(Recurrent Neural Network, RNN)
FFNN이 아닌 신경망 중 RNN이 그 중 하나로, 활성화 함수 결과를 출력층 방향으로도 보내며, 은닉층 노드의 다음 계산의 입력으로 보내는 특징이 있다.
RNN에서는 은닉층에서 활성화함수로 결과를 내보내는 노드를 Cell이라고 하며, 이전의 값을 기억하는 메모리 역활을 하기에 memory cell, RNN cell이라고 부른다.
메모리 셀은 각각의 시점(time step)에서 자신에게 보내는 값을 은닉 상태(hidden state)라고 하며, RNN은 FFNN에서의 표현과는 조금 달리 입력벡터, 출력벡터라는 표현을 사용한다.
 RNN은 입력과 출력에 길이에 따라 one-to-many, many-to-one, many-to-many구조의 설계가 가능하며, 각각 하나의 이미지 입력에 대해 사진의 제목(단어들의 나열)을 출력하는
Image Captioning, 단어 시퀀스에 대해 하나의 출력을 하는 spam detection과 sentiment classification, 챗봇이나 번역기, 개체명 인식이나 품사 태깅에 사용된다.
식으로 표현하면 은닉층은 h=tanh(Wh+Wx+b), 출력층은 f(Wh+b)꼴이며, 이 식에서 각각의 가중치값은 하나의 층에서 모든 시점에서 값을 동일하게 공유하지만
은닉층이 2개 이상이라면 각 은닉층에서의 가중치는 서로 다르다. 출력층의 활성화 함수는 Classification에 따라 선택하면 된다."""

    #2. 케라스(Keras)로 RNN 구현하기
from tensorflow.keras.layers import SimpleRNN
#model.add(SimpleRNN(hidden_units))#RNN층을 추가

#model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))#hidden_units은 은닉상태의 크기를 정의하며, 메모리셀의 output_dim과 동일하다. RNN의 capacity를 조정하며 보통 128,256,512,1024크기를 가진다.
#model.add(SimpleRNN(hidden_units, input_length=M, input_dim=N))#another annotation
"""
RNN층은 (batch_size, timesteps, input_dim)크기의 3D 텐서를 입력으로 받는다. 이는 사용자의 설정에 따라 두 가지 종류의 출력을 내보내는데
메모리 셀의 최종 시점의 은닉 상태만을 리턴하고 싶다면 (batch_size, output_dim)크기의 2D 텐서를 리턴하고, 메모리 셀의 각 시점(timestep)의
은닉 상태값들을 모아서 전체 시퀀스를 리턴하고 싶다면 (batch_size, timesteps, output_dim)크기의 3D 텐서를 리턴한다. 이는 RNN층인자 return_sequences=True로 조정한다.
즉, 차이점을 다시 말하면 메모리 셀이 모든 시점(지금까지의 모든 time step)에 대한 은닉 상태값을 출력하고, 별도기재하지 않으면 마지막 시점의 은닉상태값만을 출력한다.
 마지막 은닉상태로는 many-to-one을, 모든 은닉상태로는 추가적인 RNN은닉층이나 many-to-many를 풀 수 있다.
LSTM이나 GRU도 model.add()를 통하여 추가하는 코드는 SimpleRNN과 같은 형태를 가진다.(내부 메커니즘은 다르지만)"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN

model=Sequential()
model.add(SimpleRNN(3, input_shape=(2,10)))#model.add(SimpleRNN(3, input_length=2, input_dim=10))_3: 히든 유닛의 값(output_dim?)
model.summary()#출력값이 (batch_size, output_dim) 2D Tensor일 때, output_dim은 hidden_units의 값인 3이다. batch_size는 현 단계에서 모르기에 (None, 3)이 된다.
#Output Shape: (None, 3)
model=Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10)))#batch_size를 기재하면 (8,3)이 된다.
model.summary()
#Output Shape: (8, 3)
model=Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10), return_sequences=True))#return_sequences를 True로 하여 (batch_size, timesteps, output_dim) 3D Tensor를 출력하게한다.
model.summary()
#Output Shape(8, 2, 3) 사실 이 Output Shape, Input Shape부분 잘 이해 안감. 첫번째 인자는 hidden_units로, 은닉 상태의 크기이며 다음 시점의 메모리 셀과 출력층으로 보내는 값의 크기(output_dim과도 동일하다)
#입출력 텐서는 batch_size, timesteps, output_dim으로 구성되며, 첫번째 모델의 경우 아는게 hidden_units밖에 없기에 (None, 3)
#두번째는 batch_size를 input_shape에 같이 전달해줬기에 (8,3), 세번째는 인자를 통해 timesteps별 hidden state를 출력하기 위해 (8,2,3). 즉 8은 batch_size, 3개의 출력값을 2개의 timesteps별로 담은 데이터 반환.

    #3. 파이썬으로 RNN 구현하기_실제 3D Tensor를 입력받는 keras와 다르게, (timesteps, input_dim)의 2D Tensor를 입력받는다고 가정(batch개념 제외)
#timesteps는 자연어처리에서 보통 문장의 길이이며, input_dim은 단어 벡터의 차원이다.
import numpy as np

timesteps=10
input_dim=4
hidden_units=8

inputs=np.random.random((timesteps, input_dim))
hidden_state_t=np.zeros((hidden_units,))
print('\n\n초기 은닉 상태: ', hidden_state_t)

wx=np.random.random((hidden_units, input_dim))#입력에 대한 가중치
wh=np.random.random((hidden_units, hidden_units))#은닉 상태에 대한 가중치
b=np.random.random((hidden_units,))#편향값
print('가중치 wx의 크기(shape): ', np.shape(wx))
print('가중치 wh의 크기(shape): ', np.shape(wh))
print('편향의 크기(shape): ', np.shape(b))

total_hidden_states=[]

for input_t in inputs:
    output_t=np.tanh(np.dot(wx, input_t)+np.dot(wh, hidden_state_t)+b)#output 계산

    total_hidden_states.append(list(output_t))#hidden_state에 추가하여 timesteps별 hidden_states저장토록
    hidden_state_t=output_t#hidden_state update for next loop
    
total_hidden_states=np.stack(total_hidden_states, axis=0)#깔끔출력
print('모든 시점의 은닉 상태: ')
print(total_hidden_states)

    #4. 깊은 순환 신경망(Deep Recurrent Neural Network)_2개이상의 은닉층을 가지는 RNN(x->cell_1->cell_2->y)
model=Sequential()
model.add(SimpleRNN(hidden_units, input_length=10, input_dim=5, return_sequences=True))
model.add(SimpleRNN(hidden_units, return_sequences=True))

    #5. 양방향 순환 신경망(Bidirectional Recurrent Neural Network)
"""시점 T에서의 출력값을 예측할 때, 이전 시점의 입력뿐만 아니라 이후 시점의 입력또한 예측에 기여할 수 있다.(역방향 hidden state전파)
예로 단어빈칸의 왼쪽과 오른쪽을 모두 고려하여 정답을 결정하는 language modeling이 있다. 고로 현재 시점의 예측을 더욱 정확하게 도와준다.
 기본적으로 두 개의 메모리 셀을 사용하여 하나는 Forward States를 다른건 Backward States를 전달받아 현재의 hidden state를 계산한다."""
from tensorflow.keras.layers import Bidirectional

timesteps=10
input_dim=5

model=Sequential()
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True), input_shape=(timesteps, input_dim)))
#Bidirectionalize할 SimpleRNN을 정의하고, input_shape를 정의한다..? 그냥 보다 정확한 Bidirectionalization을 위해서 input_shape가 필요해서 따로 빼둔거같음.

#Bidirectional Recurrent Neural Network에 은닉층 추가 버전(deep BRNN)
model=Sequential()
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True), input_shape(timesteps, input_dim)))
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True)))
#이는 태깅 작업에 유용하다
