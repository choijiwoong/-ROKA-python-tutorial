 #Training
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model

embedding_dim=64
hidden_units=64

encoder_inputs=Input(shape=(None,))
encoder_embedded=Embedding(src_vocab_size, embedding_dim)(encoder_inputs)#Word-Level과 달리 Embedding 수행
encoder_masked=Masking(mask_value=0.0)(encoder_embedded)#그냥 정확도를 위해 padding값 제거
encoder_lstm=LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c=encoder_lstm(encoder_masked)
encoder_states=[state_h, state_c]#Context Vector


decoder_inputs=Input(shape=(None,))
decoder_embedding_layer=Embedding(tar_vocab_size, hidden_units)
decoder_embedded=decoder_embedding_layer(decoder_inputs)#to hidden_units개수로 Embedding
decoder_masked=Masking(mask_value=0.0)(decoder_embedded)

decoder_lstm=LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _=decoder_lstm(decoder_masked, initial_state=encoder_states)#Encoder의 Context Vector를 Decoder의 initial state로 사용(마찬가지로 훈련중에는 decoder의 state_h와 state_c사용X)

decoder_dense=Dense(tar_vocab_size, activation='softmax')
decoder_ouputs=decoder_dense(decoder_outputs)#Multi-classification을 위해 softmax층을 지나 다시 tar_vocab_size로


model=Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

 #Working
encoder_model=Model(encoder_inputs, encoder_states)#인코더 모델은 그대로 만들어서 사용!

decoder_state_input_h=Input(shape=(hidden_units,))
decoder_state_input_c=Input(shape=(hidden_units,))
decoder_state_inputs=[decoder_state_input_h, decoder_state_input_c]#이전 시점의 상태를 담을 버퍼텐서

decoder_embedded2=decoder_embedding_layer(decoder_inputs)#훈련시 사용한거 재활용하여 임베딩

decoder_outputs2, state_h2, state_c2=decoder_lstm(decoder_embedded2, initial_state=decoder_states_inputs)#이전 시점의 상태를 현 시점의 initial state로
decoder_states=[state_h2, state_c2]#output state를 저장 for 다음 시점의 입력으로 사용.

decoder_outputs2=decoder_dense(decoder_outputs2)

decoder_model=Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs2]+decoder_state2)#입력단어와 이전 states를 합하여 입력으로, 출력단어와 states를 출력으로.


def decode_sequence(input_seq):
    states_value=encoder_model.predict(input_seq)

    target_seq=np.zeros((1,1))
    target_seq[0,0]=tar_to_index['<sos>']#decoder의 입력의 시작은 <sos>

    stop_condition=False
    decoded_sentence=''

    while not stop_condition:
        output_tokens, state_h, state_c=decoder_model.predict([target_seq]+states_value)#Model의 input부 형식과 일치! []안은 단어, +는 상태!

        sampled_token_index=np.argmax(output_tokens[0,-1:])
        sampled_char=index_to_tar[sampled_token_index]#예측된어 단어로 변환.

        decoded_sentence+=' '+sampled_char#현새 시점 예측 단어를 예측 문장에 추가.

        if (sampled_char=='<eos>' or len(decoded_sentence)>50):#종료조건 체크
            stop_condition=True

        target_seq=np.zeros((1,1))
        target_seq[0,0]=sampled_token_index#예측된 단어를 다음 입력으로 사용하기 위함.

        states_value=[state_h, state_c]#state 갱신 for 다음 디코더 입력의 initial_state로 사용하기 위함.

    return decoded_sentence
