"""가장 단순한 형태의 RNN을 Vanilla RNN이라고 한다.

    1.바닐라 RNN의 한계
비교적 짧은 Sequence에만 효과를 보이며, time step이 길어질 수록 정보가 뒤로 충분히 전달되지 못하여 시점이 충분히 길 경우 전체 정보의 영향력이 의미가 없을 수 있다.
이 때 중요한 정보가 앞쪽에 위치한다면 큰 문제가 되는데, 이를 장기 의존성 문제(The problem of Long-Term Dependencies)라고 한다.

    2. 바닐라 RNN 내부 알아보기: h=tanh(Wx+Wh+b)
    3. LSTM(Long Short-Term Memory)
LSTM은 메모리 셀에 입력 데이트, 망각(삭제) 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억해야할 것들을 정한다.
고로 hidden state와 같이 cell state를 사용하며, 공통적으로 시그모이드 함수를 사용하며, 그 값으로 게이트를 조절한다.
 입력 게이트는 현재 정보를 기억하기 위한 것으로, σ(W1xX+W2xH+b), tanh(W1xX+W2xH+b)가 각각 i, g로 표시되어 입력게이트로 입력된다.(H는 hidden_state이전의)
 삭제 게이트는 기억을 삭제하기 위한 게이크로 σ(Wx+WH+b)가 f로 표현되어 삭제게이트로 입력된다. sigmoid를 통해 도출된 값은 정보의 양을 의미하며
0에 가까울 수록 정보의 손실이 많음을 의미한다.(이전 cel_state정보)
 셀 상태는 입력게이트에서 구한 i와 g에 대해 entrywise product(원소별 곱)를 진핸하는데, C(cell_state)=f∘C+i∘g로 셀상태를 구할 수 있으며 이 값은 다음 t+1시점의 LSTM셀로 넘겨진다.
즉, 삭제 게이트를 통과한 값(sigmoid)으로 이전 cell_state의 반영도를 결정하는 것으로, 삭제 게이트를 통과한 값(f)가 0이라면 이전 cell_state는
영향을 끼치지 못하고 입력 게이트의 값(g∘i)만이 영향을 끼치며 이는 삭제 게이트가 완전히 닫히고 입력 게이트가 완전히 열렸다고 표현한다.
반대로 입력 게이트를 통과한 값(g∘i)가 0이라면 현재 LSTM cell의 cell_state는 이전의 cell_state에만 의존하며 이를 입력 게이트가 완전히 닫히고
삭제 게이트만을 연 상태를 의미한다. 결론적으로 삭제게이트는 t-1의 cell_state반영도, 입력 게이트는 t의 입력반영도를 결정한다.
 출력 게이트는 o=σ(Wx+Wh+b)가 tanh를 통과한 현재의 cell_state가 입력으로 사용되며, 현재 t시점의 hidden_state를 결정한다.
셀상태의 값은 tanh를 지나 (-1,+1)이 되고, 위의 출력 게이트값과 연산되며 값이 걸러져 hidden_state가 된다. 이는 출력층으로도 향한다.

와 이해는 되는데 뭔가 간지난다. 기억이 관건일듯"""

    #[GRU]
"""GRU(Gated Recurrent Unit)은 LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 update하는 계산을 줄였다(최적화)
GRU의 성능은 LSTM과 유사하지만, LSTM의 구조를 단순화시켰다.

    1. GRU(Gated Recurrent Unit)
성능은 비슷하나, 데이터의 양이 적을땐 매개변수값이 적은 GRU가 조금 빠르고 데이터 양이 많으면 LSTM이 더 좋다.
케라스에서 SimpleRNN, LSTM과 같이 GRU를 사용할 수 있다."""
model.add(GRU(hidden_state, input_shape=(timesteps, input_dim)))
