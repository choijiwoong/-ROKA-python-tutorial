    #[코랩(Colab)에서 TPU 사용하기]
import tensorflow as tf
import os

#TPU 초기화(필요한 설정들)
resolver=tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://'+os.environ['COLAB_TPU_ADDR'])

tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

#TPU Strategy 세팅(여러TPU로 훈련을 나눠하기 위한 분산처리 세팅)
strategy=tf.distribute.TPUStrategy(resolver)

#딥러닝 모델의 컴파일(strategy.scope내에서 모델을 컴파일해야한다)
def create_model():
    return tf.keras.Sequential(
        [tf.keras.layers.Conv2D(256, 3, activation='relu', input_shape=(28,28,1)),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(256, activation='relu'),
         tf.keras.layers.Dense(128, activation='relu'),
         tf.keras.layers.Dense(10)])

with strategy.scope():
    model=create_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
#이제 fit() 실행 시 TPU를 사용하여 학습하게 된다. GPU위에서 실행한다면, 위에 TPU설정부만 지워버리면 된다.


    #[Transformers의 모델 클래스 불러오기]
 #1. many-to-one(텍스트 분류)
from transformers import TFBertForSequenceClassification

model=TFBertForSequenceClassification.from_pretrained('모델 이름', num_labels=분류할 레이블 개수)

 #2. many-to-many(개체명 인식)
from transformers import TFBertForTokenClassification

model=TFBertForTokenClassification.from_pretrained('모델 이름', num_labels=분류할 레이블 개수)

 #3. QA(질의응답)
from transformers import TFBertQuestionAnswering

model=TFBertForQuestionAnswering.from_pretrained('모델 이름')
