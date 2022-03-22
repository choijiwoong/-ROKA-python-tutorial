class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):#length of sentence, dimention of embedding vector
        super(PositionalEncoding, self).__init__()
        self.pos_encoding=self.positional_encoding(position, d_model)#member function의 결과를 저장. 얘는 약간 이런식으로 Meta Programming가능할거같은데..

    def get_angles(self, position, i, d_model):#PE함수의 삼각함수 내부에 들어갈 식 리턴
        angles=1/tf.pow(10000, (2*(i//2))/tf.cast(d_model, tf.float32))#이게 positional encoding에 사용되는 PE함수에서 sin, cos의 내부식에 해당.
        return position*angles

    def positional_encoding(self, position, d_model):
        angle_rads=self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],#[0~len(sentence)-1][1]
                                   i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],#[1][0~dim_emb-1]
                                   d_model=d_model)#dim_emb
        #PE functions
        sines=tf.math.sin(angle_rads[:, 0::2])#get_angles 함수를 통해 나온 PE함수의 삼각함수 내부식을 PE(pos, 2i), PE(pos, 2i+1)나누어 저장.
        cosines=tf.math.cos(angle_rads[:, 1::2])

        angle_rads=np.zeros(angle_rads.shape)#각 PE함수의 결과를 담을 변수
        angle_rads[:,0::2]=sines#알맞은 인덱스에 할당해준다
        angle_rads[:,1::2]=cosines
        pos_encoding=tf.constant(angle_rads)
        pos_encoding=pos_encoding[tf.newaxis, ...]#[1][position][d_model]

        print('pos_encoding shape: ', pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):#embedding vector->embedding vector+positional encoded embedding vector
        return inputs+self.pos_encoding[:, :tf.shape(inputs)[1], :]#input(embedding vector)와 positional encoding을 더하여 리턴.
#참고로 plt.pcolormesh(PositionalEncoding(50, 128).pos_encoding.numpy()[0], cmap='RdBu')의 시각화 그래프 의미는, Depth(d_model)가 낮을 때 위치정보가 더 촘촘히 분포한다. 정도 일단 읽을 수 있음.


def scaled_dot_product_attention(query, key, value, mask):#(일반적인 dot-prooduct attention과 유사)q: 모든 시점의 디코더 셀의 은닉 상태들, k: 모든 시점의 인코더 은닉, v: k와 동일
    matmul_qk=tf.matmul(query, key, transpose_b=True)#dot-product(디코더 t은닉, 인코더 all 은닉)

    depth=tf.cast(tf.shape(key)[-1], tf.float32)
    logits=matmul_qk/tf.math.sqrt(depth)#스케일링에 사용할 크기_key dimention의 루트값으로 scaling. Attention Score

    if mask is not None:#현재 마스크가 있다면 반영한다.
        logits+=(mask*-1e9)

    attention_weights=tf.nn.softmax(logits, axis=-1)#attention weights
    output=tf.matmul(attention_weights, value)#attention value(attention weights와 인코더 all은닉 weighted sum)

    return output, attention_weights
    
np.set_printoptions(suppress=True)
temp_k=tf.constant([[10,0,0],
                    [0,10,0],
                    [0,0,10],
                    [0,0,10]], dtype=tf.float32)#(4,3) 인코더의 모든시점 hs
temp_v=tf.constant([[   1,0],
                    [  10,0],
                    [ 100,5],
                    [1000,6]], dtype=tf.float32)#(1,3) 인코더의 모든시점 hs(attention전 + 후니 동일해야할텐데..이렇게까지 자세히 이해하라는 취지가 아니라 대충 설명한거고 사실 달랐던건가.. 여기서 괴리감이 생겼었고만. 근데 이론상으론 key에 대응되는 value라 달라야하긴하는데.. 혹시 k가 position이고 v가 그 값인가)
temp_q=tf.constant([[0,10,0]], dtype=tf.float32)#(1,3) 디코더의 현 시점 hs

temp_out, temp_attn=scaled_dot_product_attention(temp_q, temp_k, temp_v, None)#모든 인코더의 정보가 반영된 디코더의 입력(이론상)
print(temp_attn)#tf.Tensor([[0. 1. 0. 0.]], shape=(1, 4), dtype=float32) Q가 Key어디에 속하는지 softmax, 그 value가져오기
print(temp_out)#tf.Tensor([[10.  0.]], shape=(1, 2), dtype=float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads=num_heads#병렬 작업을 위함
        self.d_model=d_model#전체 사이즈

        assert d_model%self.num_heads==0

        self.depth=d_model//self.num_heads#병렬어텐션 단위

        self.query_dense=tf.keras.layers.Dense(units=d_model)#가중치 행렬을 Dense layer을 사용하여 구현한다.
        self.key_dense=tf.keras.layers.Dense(units=d_model)
        self.value_dense=tf.keras.layer.Dense(units=d_model)

        self.dense=tf.keras.layers.Dense(units=d_model)#마지막에 각각 concat된 병렬어텐션들의 크기를 기존과 맞춰주기 위한 WO(weight output)

     def split_heads(self, inputs, batch_size):#num_heads만큼 q, k, v를 나눈다.(병렬 어텐션의 전처리)
        inputs=tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))#배치 사이즈개수만큼 알아서 reshaping(a, num_heads, depth)->(batch_size, b, num_heads, depth)가 되게 알아서 reshape
        return tf.transpose(inputs, perm=[0,2,1,3])#Transpose^T하는데, 두번째 새번째 shape를 전치. for 전처리 of scaled_dot_product인풋

     def call(self, inputs):
        query, key, value, mask=inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size=tf.shape(query)[0]#query사이즈로 batch_size설정. 즉, encoder의 각 hidden_state개수만큼 병렬하겠다는 거임

        query=self.query_dense(query)#가중치 곱
        key=self.key_dense(key)
        value=self.value_dense(value)

        query=self.split_heads(query, batch_size)#병렬을 위한 나누기
        key=self.split_heads(key, batch_size)
        value=self.split_heads(value, batch_size)

        scaled_attention, _=scaled_dot_product_attention(query, key, value, mask)#기존의 (batch_size, num_heads, query문장길이, d_model/num_heads)에서
        scaled_attention=tf.transpose(scaled_attention, perm=[0,2,1,3])#(batch_size, query의 문장길이, num_heads, d_model/num_heads)로 reshape

        concat_attention=tf.reshape(scaled_attention, (batch_size, -1, self.d_model))#-1를 이용한 auto reshape로 사실상 num_heads병렬작업의 concat(batch_size, query문장길이, d_model)
        outputs=self.dense(concat_attention)#크기맞추기용 WO

        return outputs
    

def create_padding_mask(x):
    mask=tf.cast(tf.math.equal(x,0, tf.float32))#x값중 0(<pad>)인거 1로 변경
    return mask[:, tf.newaxis, tf.newaxis, :]#(batch_size, 1, 1, key의 문장길이)


def encoder_layer(dff, d_model, num_heads ,dropout, name='encoder_layer'):pass#헤헿 오늘은 여기까지.
