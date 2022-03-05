#Keras의 구현 방식으로 Sequential API, Functional API외에도 Subclassing API방식이 존재한다.
    #Subclassing API로 구현한 선형 회귀
import tensorflow as tf

class LinearRegression(tf.keras.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()#tf.keras.Model의 속성들을 가지고 초기화
        self.linear_layer=tf.keras.layers.Dense(1, input_dim=1, activation='linear')#자신의 linear_layer을 Dense layer로

    def call(self, x):#forward연산의 정의
        y_pred=self.linear_layer(x)#call initialized linear_layer
        return y_pred
    
model=LinearRegression()

X=[1,2,3,4,5,6,7,8,9]
y=[11,22,33,44,53,66,77,87,95]

sgd=tf.keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])#tf.keras.Model을 상속받았기에 일반적인 model함수 사용가능
model.fit(X, y, epochs=300)

    """ [Subclassing API를 사용하는 경우]
Sequential API로 구현이 불가하여 Functional API로 구현하려하지만 안되는 경우 서브클래싱 API로 규현하는 경우가 많다.
Functional API는 default로 딥러닝모델을 방향이 있고 순환이 없는 DAG(Directed Acyclic Graph)로 취급하지만, 재귀 네트워크, 트리 RNN처럼 그렇지 않은 경우에 사용한다.
대부분은 Functional API수준에서 구현이 가능하다.

    [세 가지 구현 방식 비교]
Sequential API는 간단하지만 multi-input&out모델, concatenate나 Add연산에는 적합하지 않다.
Functional API는 어려운 모델을 구현하지만 shape를 명시한 Input layer을 모델 앞단에 정의해야만 한다. ex) inputs=Input(shape=(1,))
SubclassingAPI는 non-DAG모델을 구현하지만 OOP에 익숙해야한다."""
