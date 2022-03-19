""" 이전의 seq2seq모델은 Encoder의 출력인 context vector에 모든 정보를 압축하여 Decoder에 전달하다보니 정보 손실이 발생하며, RNN의 고전적인 문제인
Vanishing gradient가 존재하기에 입력 문장이 길수록 번역 품질이 떨어진다는 두가지 문제가 있다. 이를 보정하기위해 Attention기법이 사용된다.

 1. 어텐션(Attention)의 아이디어
디코더에서 출력단어를 예측하는 매 timestep별로 인코더의 전체 입력 문장을 다시한번 참고한다는 것이다.
이때 모든 문장을 균일하게 보는 것이 아닌 decoder의 해당 timestep에서 연관되는 입력 단어의 부분을 좀 더 집중(attention)하여 보는 것이다.

 2. 어텐션 함수(Attention Function)
Attention(Q, K, V)=Attention Value로 표현하며, 주어진 Query에 대한 모든 Key의 유사도를 구하여, 이에 매핑된 Value를 반영한다.
Query는 특정 timestep에서의 Decoder cell's hidden_state를, Keys는 all-timesteps에서 Encoder cell's hidden states를, Values는 all-timesteps에서 Encoder sell's hidden_states를 의미한다.

 3. 닷-프로덕트 어텐션(Dot-Product Attention)
가장 수식적으로 이해하기 쉬운 어텐션으로, 인코더의 모든 정보를 소프트맥스를 통해 도움이 되는 정도를 수치화하여 이를 하나의 정보로 담아 디코더로 전송한다.
 Attention mechanism에서 Decoder의 입력으로 이전 time-step의 hidden_state, cell_state외에도 attention_value를 입력받기에 우선 Attention Score을 계산해야한다.
이 Attention Score는 인코더의 모든 hidden-states가 디코더의 현 시점의 hidden_state와 얼마나 유사한지를 나타내는 값이다. 이를 구하기 위해서
decoder의 current hidden_state를 transpose하고, encoder의 각 hidden_state와 dot product를 수행하여 각 Scalar를 얻는다.
 그리고 Softmax를 적용하여 Attention Distribution을 얻고, 이 각각의 값을 Attention Weight라고 칭한다.
 그 뒤 인코더의 각 시점의 Attention_Weight와 hidden_state를 가중합하여 Attention Value를 구한다. (즉, 기존의 hidden_state와 attention weight를 곱하여 나온 값을 모두 더한다.)
이를 Attention Value라고 하며, 이를 Context Vector라고도 부른다. 이것은 seq2seq의 Encoder의 마지막 은닉상태를 지칭하는 context vector와는 다른 것이다.
 그렇게 구한 Attention Value와 Decoder의 현재 timestep의 hidden_state를 concatenate하여 하나의 벡터로 만든다. 이것을 다음 time_step input으로 넣어 예측을 향상시킨다.
 이 하나의 벡터를 출력층으로 보내기 전에 tanh를 거쳐 신경망 연산을 한번 더 추가한다. 논문에서 그랬기에 이유는 모른다. 최종적으로 이 값을 다음 timestep의 입력으로 사용한다.

 4. 다양한 종류의 어텐션(Attention)
웬만한 어텐션들의 차이는 (Encoder각 timestep의 hidden_state와 decoder의 current_timestep's hidden_state)중간수식의 차이로, 닷-프로덕트 어텐션은 중간수식이 내적(dot-product)였기 때문이고,
이 중간수식에 따라 여러 종류(dot, scaled dot, general, concat, location-base)의 어텐션 스코어 함수가 존재한다.
 dot-product 어텐션의 경우 사람의 이름을 따 Luong Attention이라고 부르기도 한다. concat도 사람이름을 따 바다나우(Bahdanau) 어텐션으로 부르기도 한다.
과거의 어텐션은 seq2seq의 성능을 보완하기 위해 사용되었지만, 현재는 거의 대체되어가고 있다."""
