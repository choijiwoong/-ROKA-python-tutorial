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


def encoder_layer(dff, d_model, num_heads ,dropout, name='encoder_layer'):#필요한 hyper parameter전달.
    inputs=tf.keras.Input(shape=(None, d_model), name='inputs')

    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')

    attention=MultiHeadAttention(d_model, num_heads, name='attention')({
            'query': inputs, 'key': inputs, 'value': inputs,#Q=K=V 인코더에 사용되는 Multi-head Self Attention이다.
            'mask': padding_mask
        })

    attention=tf.keras.layers.Dropout(rate=dropout)(attention)
    attention=tf.keras.layers.LayerNomalization(epsilon=1e-6)(inputs+attention)#Residual Connection+Layer Normalization

    #Position-wise FFNN
    outputs=tf.keras.layers.Dense(units=dff, activation='relu')(attention)#은닉층의 크기를 dff로 두었고
    outputs=tf.keras.layers.Dense(units=d_model)(outputs)#다시 d_model크기가 나오도록 조정.

    output=tf.keras.layers.Dropout(rate=dropout)(outputs)
    output=tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention+outputs)#Residual Connection+Layer Normalization

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)#이 모델 하나가 하나의 인코더 층으로 사용된다.

def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='encoder'):#구조도의 그림 하나가 인코더 한 층으로 표시되어있지만 실제론 여러 층이다.(추가된 인자는 vocab_size, num_layers)
    inputs=tf.keras.Input(shape=(None,), name='inputs')

    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')

    embeddings=tf.keras.layer.Embedding(vocab_size, d_model)(inputs)#입력 word들에 대한 embedding
    embeddings*=tf.math.sqrt(tf.cast(d_model, tf.float32))#Self-Attention에 사용할 Scaled dot-product Attention에서 scaling을 미리 함. 우리가 사용할 인코더 레이어는 단순한 multi-head attention이기에 scaling해주어 scaled attention을 수행하게 한다.
    embeddings=PositionalEncoding(vocab_size, d_model)(embeddings)#Embedding Vector에 positional encoding을 하여 위치정보를 직접적으로 반영한다.
    outputs=tf.keras.layers.Dropout(rate=dropout)(embeddings)#인코더로 들어갈 준비 완료(word_embedding->positional encoding)

    for i in range(num_layers):#위의 inputs과 padding mask를 전달하여 encoder를 실행하고, 그 결과를 다음 inputs으로 사용하는 Functional API의 진가.
        outputs=encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name='encoder_layer_{}'.format(i))([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)#최종적인 인코더 모델을 반환.

    
def create_look_ahead_mask(x):
    seq_len=tf.shape(x)[1]#여러 입력들 중에서 [0]은 뭐 embedding의 padding값일 거고 [1]이 총 길이겠지 input의. 몇개의 단어인지. 여기서 우린 미래timestep단어를 masking할 것.
    look_ahead_mask=1-tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)#이거 대각선 padding같은건데 seq_len모양 matrix에서 대각선 마스크 만드는거임. 어차피 가려야하는 mask가 하나씩 늘어나니 대각선모양일테니.(a, ab, abc, ...만 참고)
    padding_mask=create_padding_mask(x)#그리고 우리가 인자 mask로 다 전달해야하는데, padding_mask랑 look-ahead마스크랑 어차피 가려질거 따로 전달할 필요 없으니까 둘이 concate하여 한번에 전달.
    return tf.maximum(look_ahead_mask, padding_mask)


def decoder_layer(dff, d_model, num_heads, dropout, name='decoder_layer'):#구조도의 그림 하나가 디코더 한층
    inputs=tf.keras.Input(shape=(None, d_model), name='inputs')
    enc_outputs=tf.keras.Input(shape=(None, d_model), name='encoder_outputs')#인코더 output을 input으로 사용예정.

    look_ahead_mask=tf.keras.Input(shape=(1,None, None), name='look_ahead_mask')#첫번째 Masked-Attention용(첫번째 서브층)
    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')#두번째 Encoder-Decoder Attention용(두번째 서브층)

    attention1=MultiHeadAttention(d_model, num_heads, name='attention_1')({
            'query': inputs, 'key': inputs, 'value': inputs,#Q=K=V. 인코더 어텐션과 다른거 mask가 look-ahead란 것만 다르다.
            'mask': look_ahead_mask
        })
    attention1=tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention+inputs)#Residual Connection & Layer Normalization

    attention2=MultiHeadAttention(d_model, num_heads, name='attention_2')({
            'query': attention1, 'key': enc_outputs, 'value': enc_outputs,#Q!=K=V. 인코더의 input을 사용하는 것은 decoder input을 참고하면서 encoder의 모든 입력흐름을 보기 위함이다.
            'mask': padding_mask#참고로 이따가 Residual Connection에 사용되는 X는 당연히 디코더 attention1을 사용한다. f(x)는 attention2의 출력값을 사용한다. 말하고자하는것은 residual connection에서 x가 encoder_out이 포함된 값이 아니라는 말이다.
        })
    attention2=tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2=tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2+attention1)#Residual Connection: x+f(x)에서 x는 encoder_out을 포함하지 않는 attention1의 값이다.

    outputs=tf.keras.layer.Dense(units=dff, activation='relu')(attention2)#Position-wise FFNN
    outputs=tf.keras.layer.Dense(units=d_model)(outputs)
    outputs=tf.keras.layers.Dropout(rate=dropout)(outputs)#Residual Connection & Layer Normalization
    outputs=tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs+attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)#디코더 레이어 모델 반환. decoder_input값, enc_output값을 주 입력으로 받고, sublayer1, sublayer2에서 사용할 mask를 입력으로 받는다.

def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='decoder'):#구조도 상과 동일하지만 내부 decoder부분은 실제로 여러 층(num_layers)으로 이루어 져있다.(인자는 인코더최종모델과 같이 vocab_size, num_layers가 추가)
    inputs=tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs=tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    look_ahead_mask=tf.keras.Input(shape=(1,None,None), name='look_ahead_mask')
    padding_mask=tf.keras.Input(shape=(1,1,None), name='padding_mask')

    embeddings=tf.keras.layers.Embedding(vocab_size, d_model)(inputs)#word embedding
    embeddings*=tf.math.sqrt(tf.cast(d_model, tf.float32))#scaling
    embeddings=PositionalEncoding(vocab_size, d_model)(embeddings)#Positional encoding
    outputs=tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs=decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name='decoder_layer_{}'.format(i))(
            inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask]#한 층 디코더에 사용되는 word_embedding, encoder's output, masks
        )

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)#디코더의 최종모델 반환.


def transformaer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='transformer'):
    inputs=tf.keras.Input(shape=(None,), name='inputs')
    dec_inputs=tf.keras.Input(shape=(None,), name='dec_inputs')

    enc_padding_mask=tf.keras.layers.Lambda(create_padding_mask, output_shape=(1,1,None), name='enc_padding_mask')(inputs)#Lambda Layer(attention1 mask)
    look_ahead_mask=tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)#attention2 mask)
    dec_padding_mask=tf.keras.layers.Lambda(1, 1, None), name='dec_padding_mask')(inputs)#attention3 mask

    enc_outputs=encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_padding_mask])#입력과 mask을 input으로, 디코더로 전달예정

    dec_outputs=decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])#출력층으로 전달예정

    outputs=tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)#단어예측

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

    
small_transformer=transformer(
    vocab_size=9000,
    num_layers=4,
    dff=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name='small_transformer')
tf.keras.utils.plot_model(small_transformer, to_file='small_transformer.png', show_shapes=True)


def loss_function(y_true, y_pred):
    MAX_LENGTH=40#데이터 전처리에 따라 다름.
    y_true=tf.reshape(y_true, shape=(-1, MAXLENGTH-1))
    
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)#from_logits는 이 값이 분포를 의미하는건지이고, reduction의 기본값은 auto이며, 값들을 그대로 반환하는지, 하나로 모아서 반환해주는지를 의미한다.(각 오차가 shape대로 나옴 len(y_true))
    
    mask=tf.cast(tf.not_equal(y_true, 0), tf.float32)#y_true를 vector로 만들어서..?
    loss=tf.multiply(loss, mask)#대충 y_true와 y_pred loss값을 행렬계산하는거같은데...

    return tf.reduce_mean(loss)#그 뒤 loss값들의 평균을 손실값으로 반환.


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model=d_model
        self.d_model=tf.cast(self.d_model, tf.float32)#d_model 설정
        self.warmup_steps=warmup_steps#warmup_steps설정. 기존 keras LearningRateSchedule에서 d_model, warmup_steps만 사용자가 지정할 수 있게 하려고 상속한거임.

    def __call__(self, step):#트랜스포머에서 사용할 학습률조정 방법
        arg1=tf.math.rsqrt(step)#공식에 따른다
        arg2=step*(self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1, arg2)


#여기부터는 다음 챕터 내용을 담는데, 내용정리이며 실제 작동하지 않은 의사코드들이기때문에(코드는 같지만 검토도 없고, 임포트, 데이터 등 신경안써서 작동안되니까 아래도 전처리 제외하고 요약집 느낌으로 핵심만 담음
 #1. tf.data.Dataset으로 데이터를 배치단위로 불러오기(데이터셋 준비)
BATCH_SIZE=64
BUFFER_SIZE=20000

dataset=tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inptus': answers[:, :-1]#마지막 패딩토큰 제거<EOS>
    },
    {
        'outputs': answers[:, 1:]#맨 처음 토큰 제거<SOS>
    },
))
dataset=dataset.cache()
dataset=dataset.shuffle(BUFFER_SIZE)
dataset=dataset.batch(BATCH_SIZE)
dataset=dataset.prefetch(tf.data.experimental.AUTOTUNE)#학습중일때 데이터로드시간을 줄이기 위해 미리 메모리에 적재

 #2. 트랜스포머 만들기
tf.keras.backend.clear_session()#loop 문을 통해 많은 모델을 생성하는 경우 점점 많은 메모리를 차지하게되는데, 모델들을 전역상태에서 해제하여 메모리를 줄일 수 있게하는데 기여한다.
D_MODEL=256
NUM_LAYERS=2
NUM_HEADS=8
DFF=512
DROPOUT=0.1

model=transformer(vocab_size=VOCAB_SIZE,
                  num_layers=NUM_LAYERS,
                  dff=DFF,
                  d_model=D_MODEL,
                  num_heads=NUM_HEADS,
                  dropout=DROPOUT)


learning_rate=CustomSchedule(D_MODEL)#d_model값을 넣어 LearningRateScheduler생성

optimizer=tf.keras.optimizer.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)#Optimizer의 학습률에 CustomSchduler를 설정한다.

def accuracy(y_true, y_pred):#모델 컴파일 시 사용할 metrics직접 지정.
    y_true=tf.reshape(y_true, shape(-1, MAX_LENGTH-1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)#일반적인 sparse_categorical_accuracy전에 y_true shaping. because of MAX_LENGTH

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
