    #1. Character-Level Neural Maching Translation
 #Training
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense#C-Level은 Embedding_layer 사용 안함!
from tensorflow.keras.models import Model
import numpy as np

encoder_inputs=Input(shape=(None, src_vocab_size))#char vocab의 크기로 sourse문장을 의미한다.(tar_vocab은 target 문장. 디코더 인풋용)
encoder_lstm=LSTM(units=256, return_state=True)#Encoder의 last hidden_state만 필요하기에 return_sequences는 활성화시키지 않는다.

encoder_outputs, state_h, state_c=encoder_lstm(encoder_inputs)#state만 Decoder의 입력으로 사용할 예정이다.
encoder_states=[state_h, state_c]#이 Encoder last timestep의 states가 Decoder의 입력으로 들어갈 Context Vector이다.


decoder_inputs=Input(shape=(None, tar_vocab_size))
decoder_lstm=LSTM(units=256, return_seqeunces=True, return_state=True)#Decoder에서 각 단어별(문자별)예측을 할 것이기에 return_sequences를 활성화한다.

decoder_outputs, _, _=decoder_lstm(decoder_inputs, initial_state=encoder_states)#위에서 정의한 Decoder용 LSTM의 initial state로 Context Vector(Encoder's last state)를 넣는다.

decoder_softmax_layer=Dense(tar_vocab_size, activation='softmax')#target vocab_size출력뉴런 softmax Dense층을 만든뒤
decoder_outputs=decoder_softmax_layer(decoder_outputs)#decoder_outputs를 통과시켜 decoder_outputs을 얻는다.


model=Model([encoder_input, decoder_inputs], decoder_outputs)#각각의 input들은 Functional API순서에 따라 계산되어 decoder_outputs로 연결된다. 위에 Input을 2개 정의했다.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


 #Working
encoder_model=Model(inputs=encoder_inputs, outputs=encoder_states)#인코더를 정의하는데, decoder_inputs을 사용하지 않는다.

decoder_state_input_h=Input(shape=(256,))#Decoder에서 이전 시점의 상태를 다음 시점에서 가용하게 하기위한 버퍼역활 텐서
decoder_state_input_c=Input(shape=(256,))
decoder_states_inputs=[decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c=decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)#이전 시점의 상태를 initial_state로 설정한다. 입력은 Decoder이기에 그대로 tar_size를 받는 Input을 재사용한다. 별도로 지정한 input아니다.
#Training과 다르게 state_h과 state_c를 저정한다.

decoder_states=[state_h, state_c]
decoder_outputs=decoder_softmax_layer(decoder_outputs)
decoder_model=Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs]+decoder_states)#디코더에 입력되는 decoder_inputs과 이전시점의 decoder states를 같이 input한다. 마찬가지로 디코더의 출력으로 outputs와 state를 출력한다.
#(여긴 모델 컴파일 안하고 사용하노..?)

index_to_src=dict((i, char) for char, i in src_to_index.items())
index_to_tar=dict((i, char) for char, i in tar_to_index.items())
def decode_sequence(input_seq):
    states_value=encoder_model.predict(input_seq)#input값에서 encoder의 state가 담긴 Context Vector

    target_seq=np.zeros((1,1,tar_vocab_size))
    target_seq[0,0,tar_to_index['\t']]=1.#<SOS> one-hot 생성 for woking decoder. decoder의 입력은 항상 <SOS>로.

    stop_condition=False
    decoded_sentence=""

    while not stop_condition:
        output_tokens, state_h, state_c=decoder_model.predict([target_seq]+states_value)#states_value(context vector)를 initial_state로 사용할 예정.

        sampled_token_index=np.argmax(output_tokens[0, -1, :])
        sampled_char=index_to_tar[sampled_token_index]#예측결과를 문자로 변환

        decoded_sentence+=sampled_char#result에 append

        if (sampled_char=='\n' or len(decoded_sentence>max_tar_len):#종료조건 확인
            stop_condition=True

        target_seq=np.zeros((1,1,tar_vocab_size))
        target_seq[0,0,sampled_token_index]=1.#decoder의 입력값으로 사용된 예측된 문자를 one-hot벡터로.

        state_value=[h,c]#맨 처음에 encoder의 state_h, state_c를 받아 사용했는데, 이제부턴 방금구한 decoder의 state_h와 state_c를 사용한다.
    return decoded_sentence    
