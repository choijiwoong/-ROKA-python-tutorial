""" 이는 seq2seq구조인 인코더&디코더 구조이지만 RNN없이 Attention만으로 구현한 모델로, 보다 우수한 성능을 보여준다.

    [2. 트랜스포머의 주요 하이퍼파라미터]
d_model=512 : 인코더와 디코더에서 정해진 입력과 출력의 크기이다.
num_layers=6: 인코더와 디코더가 총 몇층으로 구성되었는지
num_heads=8 : 어텐션을 분할해서 수행 후 병합하는데, 병렬의 개수를 의미한다.
d_ff        : 내부의 FFN의 은닉층 크기를 의미한다.(입출력층 크기는 d_model이다)

    [3. 트랜스포머(Transformer)]
RNN없지만 Encoder-Decoder구조를 사용하며, t개의 timestep이 아닌 N개씩 Encoders&Decoders가 존재하는 구조이다.
마찬가지로 Decoder는 <sos>를 받아 종료심볼 <eos>가 출력될때 까지 연산을 진행한다. 이전에 각 단어의 Embedding Vector를 받는 것과 달리,
트랜스포머의 인코더와 디코더는 embedding vector에서 조정된 값을 입력받는다.

    [4. 포지셔널 인코딩(Positional Encoding)]
RNN은 자연어처리에서 순차적인 입력특성으로 각 단어의 positional information을 가진다는 장점이 있었는데, RNN을 사용하지 않는 트랜스포머에서는
단어의 위치 정보를 단어 Embedding Vector에 더하여 모델의 입력으로 사용하는데, 이를 Positional Encoding이라고 한다.
Positional Encoding은 말 그대로 입력 문장에서 어느 위치에 속하는지 를 나타내는 벡터로 I am a student의 a의 경우 [0,0,1,0]꼴이다.
 구체적으로는 sin꼴의 PE(pos, 2i)함수와 cos꼴의 PE(pos, 2i+1)함수를 사용하는데, 이 값을 임베딩 벡터에 더해 positional information을 전달한다.
PE함수의 (pos, i)꼴은 행렬 index를 의미하는데, 트랜스포머에서 각 단어 embedding vector에 positional information을 더할 때, 한번에 행렬로 묶어 더하기 때문이다.
고로 2차원 matrix가 각각 만들어지는데, 입력문장의 임베딩 벡터의 위치를 (pos, i)꼴로 나타내는 것이다."""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()#기존의 init
        self.pos_encoding=self.positional_encoding(position, d_model)#positional_encoding실행

    def get_angles(self, position, i, d_model):
        angles=1/tf.pow(10000, (2*(i//2))/tf.cast(d_model, tf.float32))
        return position*angles

    def positional_encoding(self, position, d_model):
        angle_rads=self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],#새로운 차원을 가장 낮은 방향에 추가하다는거(10,20,tf.newaxis)->(10,20,1)
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],#반대
            d_model=d_model)
        
        sines=tf.math.sin(angle_rads[:, 0::2])#짝수 인덱스에 sin적용_PE(pos, 2i)
        cosines=tf.math.cos(angle_rads[:, 1::2])#1부터 step2_PE(pos,2i+1)

        angle_rads=np.zeros(angle_rads.shape)#position, i와 sine, cosine을 기반으로 positional_encoding
        angle_rads[:, 0::2]=sines
        angle_rads[:, 1::2]=cosines
        pos_encoding=tf.constant(angle_rads)
        pos_encoding=pos_encoding[tf.newaxis, ...]

        print('(PositionalEncodind)', pos_encoding.shape)#(1, 50, 128)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs+self.pos_encoding[:, :tf.shape(inputs)[1], :]#기존의 embedding+positional_information
#test
sample_pos_encoding=PositionalEncoding(50, 128)#입력문장단어 50개, 각 단어 128차원 임베딩 벡터 가정.

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0,128))
plt.ylabel('Position')
plt.colorbar()
plt.show()

""" [5. 어텐션(Attention)]
트랜스포머에서는 인코더에서 Encoder Self-Attention, 인코더-디코더에서 Masked Decoder Self-Attention, Encoder-Decoder Attention 해서 총 세가지의 어텐션이 사용되는데,
셀프어텐션이란 Query, Key, Value가 동일한 경우(출처가 같다. not same shape)를 의미한다. (고로 Query에 Decoder vector, Key&Value에 Encoder vector를 사용하는 ENcoder-Decoder Attention에는 self-가 빠져있다)

    [6. 인코더(Encoder)]
하나의 인코더 층은 크게 2개의 sublayer로 나뉘는데, Multi-head(병렬) Self-Attention과 Position-wise FFNN이다.

    [7. 인코더의 셀프 어텐션]
 1) 셀프 어텐션의 의미와 이점
어텐션 함수는 주어진 Query에 대한 모든 Key와의 유사도를 각각 구해 이를 가중치로 Value에 반영한 뒤 가중합하여 리턴한다.
자기 자신에게 어텐션을 수행하는 self-attention의 경우 Q,K,V가 입력 문장의 모든 단어벡터들을 동일하게 의미한다.
 셀프 어텐션은 문장에서 대명사가 있을 경우 그 대명사가 어느 단어와 유사도가 높은지를 알아내는 효과를 얻을 수 있다.

 2) Q,K,V 벡터 얻기
셀프 어텐션을 입력 문장의 단어 벡터에 수행하기 이전에, 인코더 초기입력인 d_model차원을 가지는 단어 벡터들을 사용하여
Q,K,V 벡터를 먼저 얻는다. 이는 d_model차원 단어벡터보다 더 작은 차원을 가진다.(논문에서 512차원(d_model) 단어벡터를 64차원(d_model/num_heads=8)으로 변환)
 기존의 단어 벡터에서 가중치 행렬(d_model x (d_model/num_heads))을 곱하여 Q,K,V벡터로 변환하는데, 이들은 훈련 과정에서 학습된다.

 3) 스케일드 닷-프로덕트 어텐션(Scaled dot-product Attention)
트랜스포머에는 내적만을 사용하는 어텐션함수가 아닌 특정값(n)으로 나눈 score(q,k)=q dot k/sqrt(n)을 사용한다.
이를 기존의 dor-product attention에서 값을 스케일링한다하여 Scaled dot-product Attention이라고 한다.
 논문에서는 이 스케일링하는 값으로 dk(d_model/num_heads)에 루트를 씌운 값을 사용한다. 이는 위에서 Q,K,V벡터얻을때 64였기에 루트를씌워 8을 사용한다.
이렇게 나온 값에 softmax를 사용하여 마찬가지로 Attention Distribution을 구하고, 각 V벡터와 가중합하여 Atention Value를 구하여 단어 I에 대한 Context Vector로서 나머지 Q, K에서도 동일한 과정을 거친다.

 4) 행렬 연산으로 일괄처리하기
위의 Scaled dot-product Attention을 각 단어별로 따로 구하기보다 Matrix Operation으로 일괄계산을 하면 된다.
문장 행렬에 가중치 행렬을 곱하여 Q, K, V행렬을 구한다. 이 Q,K행렬을 내적한 뒤 element-wise로 sqrt(dk)를 나눠주면 Attention Score를 가지는 Matrix가 되고,
이에 softmax를 사용하고 V Matrix를 곱하여 Attention Value Matrix를 구한다. 위의 과정을 Attention(Q, K, V)=softmax( QK^T / sqrt(dk) ) V로 정리가 가능하다.
 이들의 크기는, 입력문장길이: seq_len, 문장행렬 크기(seq_len, d_model)
Q벡터와 K벡터 차원을 dk, V벡터 차원을 dv라고 표기하면, Q,K행렬은 (seq_len, dk), V행렬은 (seq_len, dv)이면 가중치 행렬을 크기추정하여
Q,K의 W는 (d_model, dk), V의 W는 (d_model, dv)크기를 가진다.
 논문에서는 dk와 dv의 차원을 d_model/num_heads로 설정하였기에 결과적으로 Attention(Q,K,V)를 통해 나오는 Attention Value Matrix는 (seq_len, dv)가 된다.

 5) 스케일드 닷-프로덕트 어텐션 구현하기"""
def scaled_dot_product_attention(query, key, value, mask):
    #query.shape(batch_size, num_heads, query의 문장길이, d_model/num_heads)
    #key.shape(batch_size, num_heads, key의 문장길이, d_model/num_heads)
    #value.shape(batch_size, num_heads, value의 문장길이, d_model/num_heads)
    #padding_mask.shape(batch_size, 1, 1, key의 문장길이)
    matmul_qk=tf.matmul(query, key, transpose_b=True)#Attention Score matrix

    depth=tf.cast(tf.shape(key)[-1], tf.float32)
    logits=matmul_qk/tf.math.sqrt(depth)#scaling

    if mask is not None:#Attention Score Matrix의 마스킹 위치에 매우 작은 음수값을 넣어 softmax를 지난뒤 0이 되게 한다.??
        logits+=(mask*-1e9)

    attention_weights=tf.nn.softmax(logits, axis=-1)#(batch_size, num_heads, query문장길이, key문장길이)

    output=tf.matmul(attention_weights, value)#(batch_size, num_heads, query문장길이, d_model/num_heads)

    return output, attention_weights
#test_임의의 Query, Key, Value인 Q, K, V 행렬 생성. Query의 [0,10,0]은 Key의 두번째값과 일치하게했다. 분포를 확인해보자.
np.set_printoptions(suppress=True)
temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

temp_out, temp_attn=scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print("(scaled_dot_product_attention)Attention_Distributions: ", temp_attn)#Attention_Distributions:  tf.Tensor([[0. 1. 0. 0.]], shape=(1, 4), dtype=float32)
print('(scaled_dot_product_attention)Attention_Value: ', temp_out)#Attention_Value:  tf.Tensor([[10.  0.]], shape=(1, 2), dtype=float32)
#Query는 4개 Key중 두번째와 일치하기에 Attention Distribution이 [0,1,0,0]을 갖는다. 고로 Value의 두번째 값인 [10,0]이 출력된다.

#test2_Query의 값만 바꿔보자.
temp_q=tf.constant([[0,0,10]], dtype=tf.float32)
temp_out, temp_attn=scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print('(scaled_dot_product_attention)Attention_Distributions2:', temp_attn)#Attention_Distributions2: tf.Tensor([[0.  0.  0.5 0.5]], shape=(1, 4), dtype=float32)
print('(scaled_dot_product_attention)Attention_Value2: ', temp_out)#Attention_Value2:  tf.Tensor([[550.    5.5]], shape=(1, 2), dtype=float32)
#Query값이 Key의 세번째와 네번째모두 유사하다는 의미이며, Value값은 Value의 세번째와 네번째에 0.5씩 곱하여 합한것이다.

#test3_3개의 Query를 입력으로 사용해보자.
temp_q=tf.constant([[0,0,10], [0,10,0], [10,10,0]], dtype=tf.float32)#3,3
temp_out, temp_attn=scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print('(scaled_dot_product_attention)Attention_Distributions3:', temp_attn)#각 원소별로 잘 나온다. 결과를 행렬로 반환.
print('(scaled_dot_product_attention)Attention_Value3: ', temp_out)
"""
 6) 멀티 헤드 어텐션(Multi-head Attention)
앞서 어텐션에서 d_model차원을 가진 단어 벡터를 num_heads로 나눈 차원을 가지는 Q,K,V벡터로 바꾸고 어텐션을 수행하였다.
트랜스포머 연구진이 num_heads로 나누어 여러번의 어텐션을 병렬로 처리하는 것이 더 효과적이라고 판단했기 때문이다.
고로 d_model/num_heads차원을 가지는 Q, K, V에 대하여 num_heads개의 병렬 어텐션을 수행한다. 이때 각각의 어텐션 값 행렬을 Attention Head라고 부르며,
이에 곱해지는 가중치 행렬 W^Q, W^K, W^V는 8개의 Attention Head마다전부 다르다.
 의미는 다른 시각의 정보들을 수집하겠다는 것으로, 어텐션별로 단어의 연관도를 다른 시각에서 보기에 말 그대로 여러 관점의 해석을 반영할 수 있다는 것이다.
병렬 어텐션 수행 후 모든 Attention heads를 concatenate(seq_len, d_model=d^v x num_heads)하여 (seq_len, d_model)이 된다.
이 행렬에 또다른 가중치 행렬W^O를 곱하는데, 이것이 멀티-헤드 어텐션의 최종 결과물이다. 이는 인코더의 입력인 문장행렬(seq_len, d_model)과 크기가 동일하다.
그 뒤 Positional Wise FFNN을 지나며 크기가 계속 유지되어 다음 인코더에서 다시 입력으로 사용된다. (트랜스포머는 병렬 Encoders-Decoders형태)

 7) 멀티 헤드 어텐션(Multi-head Attention) 구현하기
크게 Q,K,V행렬을 위한 가중치 행렬WQ,WK,WV와 attention heads를 concatenate한뒤 곱하는 WO행렬이 있다.
이는 Dense layer을 통해 구현하며, Multi-head Attention은 크게 다섯가지 파트로 구성된다.
WQ,WK,WV를 나타내는 d_model크기 Dense Layer->num_heads만큼 나누기->scaled dot-product attention->concatenate->WO를 나타내는 Dense Layer"""
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads=num_heads
        self.d_model=d_model

        assert d_model%self.num_heads==0

        self.depth=d_model//self.num_heads#논문기준 64

        self.query_dense=tf.keras.layers.Dense(units=d_model)#WQ
        self.key_dense=tf.keras.layers.Dense(units=d_model)#WK
        self.value_dense=tf.keras.layers.Dense(units=d_model)#WV
        self.dense=tf.keras.layers.Dense(units=d_model)#WO

    def split_heads(self, inputs, batch_size):#Q,K,V를 num_heads개수만큼 split
        inputs=tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0,2,1,3])

    def call(self, inputs):#inputs을 dictionary형태로 받는다.
        query, key, value, mask=inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size=tf.shape(query)[0]

        query=self.query_dense(query)#WQ, WK, WV Dense 지나기
        key=self.key_dense(key)
        value=self.value_dense(value)

        query=self.split_heads(query, batch_size)#for making multi-head
        key=self.split_heads(key, batch_size)
        value=self.split_heads(value, batch_size)

        scaled_attention, _=scaled_dot_product_attention(query, key, value, mask)#나눈 Q,K,V에 대해 Scaled Dot-Product Attention수행. output, attention_weights반환
        scaled_attention=tf.transpose(scaled_attention, perm=[0,2,1,3])#transpose할 위치 perm으로 지정.
        #(batch_size, num_heads, query문장길이, d_model/num_heads)->(batch_size, query문장길이, num_heads, d_model/num_heads)

        concat_attention=tf.reshape(scaled_attention, (batch_size, -1, self.d_model))#헤드 연결(num_heads를 concate). reshape로 concate하는건 잘 모르긴 해 허헣 찾아보니 남은 차원으로 넣는다네!
        #(batch_size, query문장길이, d_model)

        outputs=self.dense(concat_attention)#WO
        #(batch_size, query문장길이, d_model)

        return outputs
"""
 8) 패딩 마스크(Padding Mask)
scaled_dot_product_attention()의 logits+=(mask*-1e9)의미를 알아보자.
입력 문장에 <PAD>토큰이 있을 경우 어텐션연산에서 제외하기가 어려운데 포함하자니 실질적인 의미를 가진 단어가 아니기에 이에대해 유사도계산을 하지 않게 Masking해야한다.
즉, 값을 가리기위해 <PAD>열 전체에 매우 작은 음수값을 넣어 마스킹을 하는 연산인 것이다. 그렇다면 softmax를 거치며 해당 위치의 값은 0에 수렴하게되어 단어간 유사도에서 <PAD>를 반영하지 않게할 수 있다.
 패딩 마스크의 구현은 입력된 정수 시퀀스에서 패딩 토큰의 인덱스인지 아닌지를 판별하는 함수를 구현하면 된다."""
def create_padding_mask(x):
    mask=tf.cast(tf.math.equal(x,0), tf.float32)#0이면 1로, 아니면 0으로. 
    return mask[:, tf.newaxis, tf.newaxis, :]#(batch_size, 1, 1, key의 문장길이)
#test
print("(create_padding_mask)",create_padding_mask(tf.constant([[1,21,777,0,0]])))
"""
다음으로 알아볼 것은 인코더가 2개의 sublayer로 나뉘어지는데 Multi-heads Self-Attention외에 다른 하나인 Position-wise FFNN이다.

    [8. 포지션-와이즈 피드 포워드 신경망(Position-wise FFNN)]
포지션 와이즈 FFNN의 수식은 FFNN(x)=MAX(0, xW1+b1)W2+b2로 표현되며, MAX는 ReLU를 의미한다.
x는 Multi-head Self-Attention의 결과인 (seq_len, d_model)이며, W1은 (d_model, dff)의 크기, W2는 (dff, d_model)의 크기를 가진다. dff는 논문에서 2048의 크기를 가진다.
이 매개변수들은 하나의 인코더 층 내에서 각 문장, 단어들마다 동일하게 사용되지만 인코더 층마다 다른 값을 가진다. 구현은 아래와 같다."""
#outputs=tf.keras.layers.Dense(units=dff, activation='relu')(attention)#size: d_model->dff
#outputs=tf.keras.layers.Dense(units=d_model)(outputs)#size: dff->d_model
"""
    [9. 잔차 연결(Residual connection)과 층 정규화(Layer Normalization)]
구조도 상 Multi-head Self-Attention뒤와 FFNN뒤에 사용되는 Add&Norm이 기법이 추가적으로 사용되는데, 이를 잔차 연결 & 층 정규화라고 한다.
 1) 잔차 연결(Residual connection)
식으로 표현하면 H(x)=x+F(x)로, F()는 서브층(Attention or FFNN)에 해당한다. 즉 그 연산결과에 자기자신을 한번 더 더하는 즉, 서브층의 입력과 출력을 더하는 행위(차원이같기에 가능)이다.
sub층이라는 것을 반영하여 식으로 표현하면 H(x)=x+Multi-head Attention(x) 혹은 H(x)=x+Sublayer(x)로 표현한다.

 2) 층 정규화(Layer Normalization)
잔차 연결의 입력을 x, 층정규화까지 마친후의 결과행렬을 LN으로 하면, Add&Norm을 수식으로 LN=LayerNorm(x+Sublayer(x))로 표현이 가능하다.
층 정규화는 텐서의 마지막 차원에 대하여 평균과 분산을 구하고, 어떤 수식으로 정규화하여 학습을 돕는 과정이다. 이 마지막 차원은 트랜스포머에서 d_model차원을 의미한다.
 lni=LayerNorm(xi)즉 정규화 마친 벡터를 lni라고 부르며, 우선 마지막층 방향으로 각각 평균과 분산을 구하고 정규화를 시작한다.
먼저 평균과 분산을 통해 다음과 같이 정규화한다. x^(hat) i,k=(xik-평균)/sqrt(분산+e) e(입실론)은 분모0을 방지하는 것으로, 그냥 평균과 분산으로 정규화한다고만 알면 된다.
그 뒤 감마r(초기값=1)와 베타b(초기값=0)라는 벡터를 준비한 뒤, 다음의 수식을 연산하면 최종적인 lni계산이 완료된다.
lni=rx^(hat)i+b=LayerNorm(xi) _이는 keras에서 LayerNormalization()으로 제공된다.

    [10. 인코더 구현하기]
인코더 입력문장의 패딩이 있을 수 있기에 패딩 마스크를 사용하며, 총 2개의 서브층(Multi-head Self-Attention, position-wise FFNN)이 사용되며 서브층 이후에 Dropout과 잔차연결, 층 정규화를 수행한다."""
def encoder_layer(dff, d_model, num_heads, dropout, name='encoder_layer'):
    inputs=tf.keras.Input(shape=(None, d_model), name='inputs')

    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')

    attention=MultiHeadAttention(
        d_model, num_heads, name='attention')({#call()
            'query': inputs, 'key': inputs, 'value': inputs,#Q=K=V
            'mask': padding_mask
            })
    attention=tf.keras.layers.Dropout(rate=dropout)(attention)#Dropout
    attention=tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs+attention)#Residual connection & Layer Normalization

    outputs=tf.keras.layers.Dense(units=dff, activation='relu')(attention)#position-wise FFNN
    outputs=tf.keras.layers.Dense(units=d_model)(outputs)#shape유지.

    outputs=tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs=tf.keras.layers.LayerNormalization(epsilon=2e-6)(attention+outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
"""이것이 하나의 인코더 층이다. 트랜스포머는 num_layers개수만큼 인코더 층을 사용한다.

    [11. 인코더 쌓기]
인코더 층을 num_layers만큼 쌓고, 마지막 인코더층에서는 (seq_len, d_model)의 행렬을 디코더로 보내주며 인코딩 연산을 마무리지어야한다.
인코더 층을 num_layers개만큼 쌓는 코드는 아래와 같다."""
def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='encoder'):
    inputs=tf.keras.Input(shape=(None,), name='inputs')

    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')#padding을 제외시킬 목적으로 mask

    #포지셔널 인코딩+드롭아웃
    embeddings=tf.keras.layers.Embedding(vocab_size, d_model)(inputs)#단어(문장) 임베딩
    embeddings*=tf.math.sqrt(tf.cast(d_model,tf.float32))#d_model..? scaled연산 이전에 미리 곱해두는건가_A.아님! 아래 참고.
    embeddings=PositionalEncoding(vocab_size, d_model)(embeddings)#PE함수 이용, word embedding vector에 positional information추가.
    outputs=tf.keras.layers.Dropout(rate=dropout)(embeddings)

    #인코더를 num_layers개 쌓는다.
    for i in range(num_layers):
        outputs=encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name='encoder_layer_{}'.format(i))([outputs, padding_mask])
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
"""
    [12. 인코더에서 디코더로(From Encoder To Decoder)]
인코더는 총 num_layers만큼의 층 연산을 순차적으로 한 뒤 마지막 층의 인코더 출력을 디코더에게 전달한다.
디코더는 똑같이 num_layers만큼의 연산을 진행한다.

    [13. 디코더의 첫번째 서브층: 셀프 어텐션과 룩-어헤드 마스크]
디코더도 인코더와 동일하게 임베딩과 포지셔널 인코딩을 거쳐 문장 행렬이 입력되는데, seq2seq와 같이 Teacher Forcing을 이용하여 훈련하기에 <sos> ~문장 행렬을 한번에 입력받는다.
이때 문제는 기존 seq2seq는 이전 timestep의 입력단어만 참고했지만, 이번엔 통째로 넣다보니 다음 시점의 값을 컷닝하여 학습할 수 있다.
고로 현재 시점보다 미래의 단어를 참고하지 못하게 룩-어헤드 마스크(look-ahead mask)를 도입한다.
 이는 디코더의 첫번째 서브층에서 이루어지며, 인코더의 Multi-head Self-Attention을 동일하게 수핸하는데, Attention Score Matrix에서 마스킹을 적용하는 것이 다르다.
이때 마스크 전달은 scaled_dot_product_attention에서 이전에 패딩 마스크를 전달했던 거에 그대로 look-ahead mask를 전달하면 된다.
 햇갈리기에 중간정리하면, 트랜스포머는 총 세가지 어텐션이 존재하며 모두 Multi-head Attention이다. 이들에게 전달되는 마스크는 아래와 같다.
Encoder's Self-Attention: Padding_mask, Decoder's Masked Self-Attention(first sublayer): look-ahead_mask, Decoder's Encoder-Decoder Attention(second sublayer): padding_mask
이때 유의할 점은 look-ahead mask를 사용하는 Masked Self-Attention에서도 padding-mask가 필요하기에 이들을 합쳐서 하나의 마스크를 전달한다."""
def create_look_ahead_mask(x):#Decoder의 first sublayer의 미래 토큰을 Mask하기 위함(feat. padding)
    seq_len=tf.shape(x)[1]
    look_ahead_mask=1-tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)#마스킹하고픈데에 1, 마스킹하지않는곳에 0. Lower trangular part를 나타낸다. 아래쪽 삼각형으로 해서 100, 110, 111이런느낌으로 미래토큰 막음
    padding_mask=create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)
#test
print(create_look_ahead_mask(tf.constant([[1,2,0,4,5]])))#삼각형모양+패딩마스크의 3번째열
"""
    [14. 디코더의 두번째 서브층: 인코더-디코더 어텐션]
인코더-디코더 어텐션은 Query가 디코더행렬, Key&Value가 인코더행렬이기에 셀프 어텐션이 아니기에 이전 어텐션들과는 다르다.
그 트랜스포머 그림 잘보면 디코더 2번째 서브층 화살표 3개중 2개는 인코더에서 온다. 이게 그 의미이다.
 어텐션 스코어 행렬은 Q와 K^T를 곱하여 구하며, 나머지 Multi-head Attention은 다른 어텐션과 동일하다.

     [15. 디코더 구현하기]"""
def decoder_layer(dff, d_model, num_heads, dropout, name='decoder_layer'):
    inputs=tf.keras.Input(shape=(None, d_model), name='inputs')
    enc_outputs=tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    look_ahead_mask=tf.keras.Input(shape=(1,None,None), name='look_ahead_mask')
    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')

    #Multi-head Attention(First sublayer/Masked Self-Attention)
    attention1=MultiHeadAttention(
        d_model, num_heads, name='attention_1')(inputs={
            'query': inputs, 'key': inputs, 'value': inputs,#Self-Attention
            'mask': look_ahead_mask
            })
    attention1=tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1+inputs)#Residual Connection & Layer Normalization

    #Multi-head Attention(Second sublayer/ Decoder-Encoder Attention)
    attention2=MultiHeadAttention(
        d_model, num_heads, name='attention_2')(inputs={
            'query': attention1, 'key': enc_outputs, 'value': enc_outputs,#Q!=K=V. non-self attention
            'mask': padding_mask
            })
    attention2=tf.keras.layers.Dropout(rate=dropout)(attention2)#Dropout
    attention2=tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2+attention1)#Residual Connection & Layer Normalization

    #Position-wise FFNN(Third sublayer)
    outputs=tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs=tf.keras.layers.Dense(units=d_model)(outputs)
    outputs=tf.keras.layers.Dropout(rate=dropout)(outputs)#Dropout
    outputs=tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs+attention2)#Residual Connection & Layer Normalization

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

#   [16. 디코더 쌓기]
def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='decoder'):
    inputs=tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs=tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    #디코더는 룩어헤드(첫서브층)와 패딩(두서브층)둥 다 사용_둥 좀 귀여운데?
    look_ahead_mask=tf.keras.Input(shape=(1,None,None), name='look_ahead_mask')
    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')

    embeddings=tf.keras.layers.Embedding(vocab_size, d_model)(inputs)#임베딩
    embeddings*=tf.math.sqrt(tf.cast(d_model, tf.float32))#아무리봐도 scaled때문은 아닌데. 아 d_model의 차원을 가지는 단어 벡터들을 만들기 위함이네. 이를 통해 QKV를 얻고. 그리고 num_heads로 나뉘 병렬처리 후 concat
    embeddings=PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs=tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):#디코더 num_layers개 쌓기
        outputs=decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name='decoder_layer_{}'.format(i))(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

#   [17. 트랜스포머 구현하기]_인코더의 출력을 연결하며, 디코더의 끝단에 다중 클래스 분류 문제를 풀 수 있게 vocab_size 출력층을 추가해준다.
def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='transformer'):
    inputs=tf.keras.Input(shape=(None,), name='inputs')#Encoder
    dec_inputs=tf.keras.Input(shape=(None,), name='dec_inputs')#Decoder

    enc_padding_mask=tf.keras.layers.Lambda(create_padding_mask, output_shape=(1,1,None), name='enc_padding_mask')(inputs)#인코더의 패딩마스트
    look_ahead_mask=tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1,None,None), name='look_ahead_mask')(dec_inputs)#디코더의 룩어헤드 마스트(첫번째 서브층 전용)
    dec_padding_mask=tf.keras.layers.Lambda(create_padding_mask, output_shape=(1,1,None), name='dec_padding_mask')(inputs)#디코더의 패딩 마스크(두번째 서브층 전용)

    enc_outputs=encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_padding_mask])
    dec_outputs=decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])#enc_output을 decoder의 input으로 전달.

    outputs=tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)#다음 단어 예측을 위한 출력층

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
"""
    [18. 트랜스포머 하이퍼파라미터 정하기]
단어 집합의 크기를 임의로 9000으로 정하자. 그럼 그로부터 임베딩 테이블(룩업)과 포지셔널 인코딩 행렬의 행의 크기를 결정할 수 있다.
 논문과는 다르게 인코더-디코더 층의 개수를 4개, 인코더와 디코더의 포지션 와이즈 피드 포워드 신경망의은닉층을 512개,
인코더와 디코더의 입출력차원은 128개, 멀티-헤드어텐션에서 병렬적으로 사용할 헤드의 수 4로 설정해보자."""
small_transformer=transformer(
    vocab_size=9000,
    num_layers=4,
    dff=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name='small_transformer')
tf.keras.utils.plot_model(small_transformer, to_file='small_transformer.png', show_shapes=True)

#   [19. 손실 함수 정의하기]_다중 클래스 분류를 풀 예정
MAX_LENGTH=40#그 다음 챕터에서 이값으로 씀. 패딩이거로해서리
def loss_function(y_true, y_pred):
    y_true=tf.reshape(y_true, shape=(-1, MAX_LENGTH-1))#Flatten하여 equal확인

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)#Functinoal API전용으로다가 있던거구나. 왜 loss='sparse~말고 다른거있다 했었네

    mask=tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss=tf.multiply(loss, mask)

    return tf.reduce_mean(loss)
"""
    [20. 학습률]
학습률 스케줄러(Learning rate Scheduler)는 미리 학습 일정을 정해두고 일정에 따라 학습률이 조정되는 방법으로, 트랜스포머의 경우
사용자가 정한 단계까지 학습률을 증가시켰다가 단계에 이르면 학습률을 점차 떨어뜨리는 방식을 사용한다. 이 기준을 warmup_steps라 칭하여
이보다 작으면 선형적으로 증가시키고, 크면 역제곱근에따라 감소시킨다. 식은 코드 혹은 사진의 식을 참고하자"""
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model=d_model
        self.d_model=tf.cast(self.d_model, tf.float32)
        self.warmup_steps=warmup_steps

    def __call__(self, step):#lrate=d^-0.5 x min(arg1, arg2)
        arg1=tf.math.rsqrt(step)#reverse sqrt 1/sqrt()
        arg2=step*(self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1, arg2)
#test_학습률의 변화 시각화
sample_learning_rate=CustomSchedule(d_model=128)

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()
