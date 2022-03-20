""" 트랜스포머는 RNN계열의 seq2seq를 대체하기 위해 등장했기에, RNN이 가능한 분야에는 트랜스포머의 인코더또한 가능하다.
트랜스포머의 인코더를 사용하여 텍스트 분류를 수행해보자."""
 #1. 멀티 헤드 어텐션
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim=embedding_dim
        self.num_heads=num_heads

        assert embedding_dim%self.num_heads==0

        self.projection_dim=embedding_dim//num_heads
        self.query_dense=tf.keras.layers.Dense(embedding_dim)
        self.key_dense=tf.keras.layers.Dense(embedding_dim)
        self.value_dense=tf.keras.layers.Dense(embedding_dim)
        self.dense=tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk=tf.matmul(query, key, transpose_b=True)#Q dot K
        depth=tf.cast(tf.shape(key)[-1], tf.float32)#prepare calc
        logits=matmul_qk/tf.math.sqrt(depth)#sepration for parallel calc
        attention_weights=tf.nn.softmax(logits, axis=-1)#get at_weight
        output=tf.matmul(attention_weights, value)#dot
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x=tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, inputs):
        batch_size=tf.shape(inputs)[0]#X.shape=[batch_size, seq_len, embedding_dim]?

        query=self.query_dense(inputs)
        key=self.key_dense(inputs)
        value=self.value_dense(inputs)

        query=self.split_heads(query, batch_size)
        key=self.split_heads(key, batch_size)
        value=self.split_heads(value, batch_size)

        scaled_attention, _=self.scaled_dot_product_attention(query, key, value)
        scaled_attention=tf.transpose(scaled_attention, perm=[0,2,1,3])

        concat_attention=tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        outputs=self.dense(concat_attention)
        return outputs

 #2. 인코더 설계
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att=MultiHeadAttention(embedding_dim, num_heads)
        self.ffn=tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation='relu'),
             tf.keras.layers.Dense(embedding_dim),]
        )
        self.layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1=tf.keras.layers.Dropout(rate)
        self.dropout2=tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output=self.att(inputs)
        attn_output=self.dropout1(attn_output, training=training)
        out1=self.layernorm1(inputs+attn_output)
        ffn_output=self.ffn(out1)
        ffn_output=self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1+ffn_output)

 #3. 포지션 임베딩_포지셔널 인코딩 대신 위치정보 자체를 학습하게 임베딩 층의 첫번째 인자로 vocab_size가 아닌 max_len of sentence를 넣는다.
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb=tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb=tf.keras.layers.Embedding(max_len, embedding_dim)

    def call(self, x):
        max_len=tf.shape(x)[-1]
        positions=tf.range(start=0, limit=max_len, delta=1)#위치정보 자체를 학습하게끔0~max_len
        positions=self.pos_emb(positions)
        x=self.token_emb(x)#단어 emb값이랑
        return x+positions#위치 emb값 더하여 반환

 #4. 데이터 로드 및 전처리
vocab_size=20000
max_len=200

(X_train, y_train), (X_test, y_test)=tf.keras.datasets.imdb.load_data(num_words=vocab_size)
print('훈련용 리뷰 개수: ', len(X_train))#25000
print('테스트용 리뷰 개수: ', len(X_test))#25000
X_train=tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test=tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

 #5. 트랜스포머를 이용한 IMDB 리뷰 분류
embedding_dim=32
num_heads=2
dff=32

inputs=tf.keras.layers.Input(shape=(max_len,))
embedding_layer=TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim)#포지션 임베딩
x=embedding_layer(inputs)
transformer_block=TransformerBlock(embedding_dim, num_heads, dff)
x=transformer_block(x)
x=tf.keras.layers.GlobalAveragePooling1D()(x)
x=tf.keras.layers.Dropout(0.1)(x)
x=tf.keras.layers.Dense(20, activation='relu')(x)
x=tf.keras.layers.Dropout(0.1)(x)
outputs=tf.keras.layers.Dense(2, activation='softmax')(x)

model=tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X_train, y_train, batch_size=32, epochs=2, validation_data=(X_test, y_test))

print('테스트 정확도: ', model.evaluate(X_test, y_test)[1])
