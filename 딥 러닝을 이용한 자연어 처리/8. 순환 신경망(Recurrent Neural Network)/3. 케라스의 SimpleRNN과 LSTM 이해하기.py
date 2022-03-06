    #1. 임의의 입력 생성하기
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional

train_X = [[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]
print("shape of train_X: ", np.shape(train_X))#(4,5) 단어 벡터의 차원은 5이고, 문장의 길이가 4인 경우를 가정한 입력.
#4번의 timesteps이 존재하며, 각 시점마다 5차원의 단어벡터가 입력된다. RNN은 3D Tensor입력을 받기에 batch_size를 1로 추가해주자.

train_X=[[[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]]
train_X=np.array(train_X, dtype=np.float32)
print("shape of train_X after making to 3D Tensor(by batch_size=1): ", train_X.shape,'\n')#(batch_size=1, timesteps=4, input_dim=5) 샘플이 1개이므로 batch_size=1

    #2. SimpleRNN 이해하기: return_sequences, return_state인자는 default로 False이다. 출력값보다 shape를 주목하자.
rnn=SimpleRNN(3)#rnn=SimpleRNN(3, return_sequences=False, return_state=False) 은닉 상태의 크기를 3으로 지정.
hidden_state=rnn(train_X)
print('hidden state: ',hidden_state, 'shape: ',hidden_state.shape)#마지막 은닉 상태만 출력. (1,3)

rnn=SimpleRNN(3, return_sequences=True)#모든 timesteps의 hidden_states를 출력하게끔
hidden_states=rnn(train_X)
print('hidden states: ',hidden_states, 'shape: ',hidden_states.shape, '\n')#(1,4,3)

#return_state와 return_sequences인자의 변화에 따른 리턴값의 변화를 심심하니까 살펴볼건데, return_state가 True일 경우 return_sequences여부와 상관없이 마지막 은닉 상태를 출력한다.
#return_state=True로 하면서 return_sequences=True로 하면 두개의 출력, 모든 h_state와 l_h_state을 출력한다.
rnn=SimpleRNN(3, return_sequences=True, return_state=True)
hidden_states, last_state=rnn(train_X)
print('hidden states(both True): ', hidden_states, hidden_states.shape)
print('last hidden state(both True): ', last_state, last_state.shape, '\n\n')#마지막 벡터값(last hidden_state)의 값이 일치하는 것을 당연히 확인할 수 있다.

#만약 return_sequences=False, return_state=True의 경우 당연하게도 모두 last_hidden_state를 출력한다.
rnn=SimpleRNN(4, return_sequences=False, return_state=True)
hidden_state, last_state=rnn(train_X)
print('hidden state: ',hidden_state, 'shape: ', hidden_state.shape)
print('last hidden state: ', last_state, 'shape: ', last_state.shape, '\n')


    #3. LSTM 이해하기: 대부분 SimpleRNN보다 LSTM이나 GRU를 사용한다.
lstm=LSTM(3, return_sequences=True, return_state=True)
hidden_states, last_hidden_state, last_cell_state=lstm(train_X)#cell_state도 반환. (파이썬은 대단하고만.. API개발자가 대단한건가.. 역시 언어의 정답은 파이썬..)
print('hidden states: ',hidden_states, 'shape: ', hidden_states.shape)
print('last hidden state: ', last_hidden_state, 'shape: ', last_hidden_state.shape)
print('last cell state: ', last_cell_state, 'shape: ', last_cell_state.shape, '\n')#LSTM의 Last Cell State를 리턴하며, 이는 다음 memory cell에 삭제게이트에 의해 사용된다.

    #4. Bidirectional(LSTM) 이해하기: return_sequences에 대해 hidden_state의 변화비교를 위해 hidden_state를 동일하게 사용하자.
k_init=tf.keras.initializers.Constant(value=0.1)
b_init=tf.keras.initializers.Constant(value=0)
r_init=tf.keras.initializers.Constant(value=0.1)

#내가 기억하기로 Bidirectional을 이용하기 위해 input_dim과 output_dim을(shape) 2nd argument로 입력했던거같은데.. 찾아보니 input_shape입력하는데.. Bidirectional에 대한 추가공부 필요할듯
bilstm=Bidirectional(LSTM(3, return_sequences=False, return_state=True, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c=bilstm(train_X)#hidden_state, cell_state

print('hidden state: {}, shape: {}'.format(hidden_states, hidden_states.shape))#(1,6)_정방향 LSTM의 last_hidden_state와 역방향 LSTM의 첫번째 시점의 hidden_state가 연결됀 채 반환된다.
print('forward state: {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state: {}, shape: {}'.format(backward_h, backward_h.shape), '\n')
#정방향 LSTM의 마지막 시점의 은닉 상태값 : [0.6303139 0.6303139 0.6303139], 역방향 LSTM의 첫번째 시점의 은닉 상태값 : [0.70387346 0.70387346 0.70387346]

#return_sequences=True로 한다면
bilstm=Bidirectional(LSTM(3, return_sequences=True, return_state=True, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
hidden_states, forward_h, forward_c, backward_h, backward_c=bilstm(train_X)

print('hidden states: {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('forward state: {}, shape: {}'.format(forward_h, forward_h.shape))
print('backward state: {}, shape: {}'.format(backward_h, backward_h.shape))
#***return_sequences=False, 정방향 LSTM의 마지막 시점의 은닉 상태와 역방향LSTM의 초기 시점의 은닉상태이 연결되어 반환. 이라고하는데
#그림으론 잘 모르겠고, 정방향 기준 마지막의 backward_state, forward_state가 연결되어 표시된다는거 같음. return_state의 대상 cell이 가진 두개의 state니까.
#반대로 위의 예시처럼 return_sequences를 True로 한다면, 정방향 기준 마지막의 forward_state, 역방향기준 마지막 forward_state가 연결되어 나온다는거 같음
#즉, 그냥 일관성이 없이 나오니까 유의하라는거 같음

