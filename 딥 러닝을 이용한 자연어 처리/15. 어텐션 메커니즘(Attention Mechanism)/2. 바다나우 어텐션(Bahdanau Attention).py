""" 복습겸 루옹 어텐션(dot-product)를 복습하고, 바다나우 어텐션을 이해해보자.

 1) 바다나우 어텐션 함수(Bahdanau Attention Function)
Attention(Q, K, V)=Attention Value로, Query: t-1 decoder's hidden_state, Keys: Encoder's hidden_states, Values: Encoder's hidden_states이다.
dot-product Attention function과 달리 Query의 timestep이 t가 아닌 t-1임에 주목하자.

 2) 바다나우 어텐션(Bahdanau Attention)
우선 당연히 Attention Score가 먼저며, 루옹 어텐션과 달리 t-1시점의 은닉상태를 사용하며,
학습가능한 가중치 행렬을 Encoder의 h_stat에, t-1 Decoder의 h_stat에 곱한 값을 더한 뒤 tanh을 지나게 한다. 그 뒤 또 학습가능한 행렬을 곱하여
Encoder의 h_stat들의 유사도가 기록된 Attention Score벡터를 얻는다.
 여기에 softmax하여 Attention Distribution을 구하고, 이 각각의 값을 기존과 마찬가지로 Attention Weight라고 한다.
이 각 Encoder의 Attention Weight와 decoder t-1의 hidden_state를 가중합(곱하고 더해버렷)하여 Attention value를 구한다.
이를 Context Vector라고 부른다(인코더의 문맥을 포함하고 있다.)
 기존의 루웅 어텐션에서는 이 Context Vector를 Decoder의 현재 timestep의 hidden_state를 concate하고 tanh를 거쳐 다름 timestep의 입력으로 사용했지만,
바다나우 어텐션에서는 context vector와 decoder의 hidden_state가 아닌 아예 입력인 단어의 임베딩 벡터에 concatenate하여 현재 시점의 새로운 입력으로 사용한다."""
