""" [Multi-class Classification]_by softmax not Binary Classification
대표적인 세 개 이상의 선택지 중 하나를 고르는 문제인 다중 클래스 분류로는 붓꽃 품종 예측이 있다.
Multi-class Classfication의 중점은 모든 확률의 총합이 1인 예측값을 얻는 것이다. 이때 softmax function이 사용된다.
 Softmax function은 k차원의 벡터를 입력받아 각 클래스에 대한 확률을 추정한다. 
이때 생각해야하는 점은 소프트 맥스 함수의 입력으로 바꾸는 방법과, 오차 계산방법이다.
 독립변수의 개수대로 출력된 tensor를 종속변수의 개수로 변환하여 softmax에 넣어야하기에 이를 축소시켜야 한다.
고로 축소를 위해서 해당 차원이 나오도록 weight를 곱하여 오차를 최소화하는 가중치로 값을 변경한다.
 두번째로 오차 계산으로는 one-hot vector를 이용하여 softmax를 통과한 예측값(0~1)이 정확하게 실제 one-hot vector값이 되도록
CrossEntropy함수를 사용하여 축소과정에서의 weight, bias를 update한다.

    [원-핫 벡터의 무작위성]
대부분 다중 클래스 분류 문제에서 각 클래스간의 관계가 균등하다면, one-hot vector로 적절히 표현할 수 있다.
일반적인 정수 인코딩보다 one-hot vector가 더 좋은데, Loss function을 구할 경우 정수 인코딩의 경우 동일한 관계의 클래스의 오차를
동일한 손실율로 나타낼 수 없기 때문이다. (2-1)^2!=(3-1)^2 고로 이걸 의도한 상황 즉, 각 클래스가 순서의 의미를 담고있는 경우가 아니라면
one-hot vector를 사용하고 유클리드 거리가 모든 쌍에 동일하다는 것을 원-핫 벡터의 무작위성이라고 한다.(단어의 유사성을 구할 수 없는 단점이기도)

    [비용 함수]
소프트맥스 회귀에서는 비용함수로 크로스 엔트로피 함수를 사용한다. 이는 로지스틱 회귀의 비용함수와 같다."""
    #[아이리스 품종 데이터에 대한 이해]
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/06.%20Machine%20Learning/dataset/Iris.csv", filename="Iris.csv")
data=pd.read_csv('Iris.csv', encoding='latin1')#https://www.kaggle.com/saurabh00007/iriscsv
print('샘플의 개수: ', len(data))
print(data[:5])
print('\n품종 종류: ', data["Species"].unique(), sep='\n')#4개의 feature로 3개의 품종을 예측

#seaborn세팅
sns.set(style='ticks', color_codes=True)
#seaborn의 pairplot은 데이터프레임을 입력으로 받아 데이터프레임의 각 열의 조합에 따라 산점도(scatter plot)을 그린다.
g=sns.pairplot(data, hue='Species', palette='husl')#4개의 특성에 해당하는 모든 쌍의 조합인 16개 경우의 산점도를 그린다.(색상대상, 색상집합)
plt.show()

#각 종과 특성에 대한 연관 관계
sns.barplot(data['Species'], data['SepalWidthCm'], ci=None)#Species들이 SepalWidthCm과의 관계를 barplot으로
plt.show()

#150개 샘플 데이터 중에서 Species열에서 각 품종이 몇 개 있는지
data['Species'].value_counts().plot(kind='bar')#species들의 값들을 count하여 barplot으로
plt.show()

#one-hot encoding을 위한 전처리: 정수 인코딩
data['Species']=data['Species'].replace(['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'], [0,1,2])#각 특성을 정수로 인코딩***
data['Species'].value_counts().plot(kind='bar')#check correct integer encoding
plt.show()

data_X=data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
data_y=data['Species'].values
print("\n상위 5개 feature: ", data_X[:5])
print("상위 5개 label: ", data_y[:5])

#훈련 데이터와 검증데이터의 분리
(X_train, X_test, y_train, y_test)=train_test_split(data_X, data_y, train_size=0.8, random_state=1)#train_test_split by train_size

#y_train과 y_test의 one-hot encoding
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
print("\n원-핫 인코딩된 상위 5개 train label: ", y_train[:5])
print("원-핫 인코딩된 상위 5개 test label: ", y_test[:5])

    #[소프트맥스 회귀]
#input_dim=4, output_dim=3, activation='softmax', loss_function='categorical_crossentropy'_다중클래스분류, optimizer='adam'_일종의 경사하강법
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))#softmax with output_dim=3, input_sim=4
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#categorical_crossentropy, adam, accuracy

#테스트 데이터를 validation_data로 기재하면 train에 사용하진 않지만 각 훈련 횟수마다 테스트 데이터에 대한 정확도를 출력한다. 즉, train epoch마다 validation_data를 test한다는 말(no update to validataion_data!)
history=model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))#trainning. History객체를 반환하며 History.history속성은 학습손실값, 측정항목값, 검증손실값, 검증측정항목값의 기록이다.
#실제 컴파일했을때 오래걸려서 epochs=10으로 함..
epochs=range(1, len(history.history['accuracy'])+1)
plt.plot(epochs, history.history['loss'])#학습손실
plt.plot(epochs, history.history['val_loss'])#검증손실
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#keras에서 테스트 데이터 정확도 측정을 위한 evaluate()로 정확도를 다시 출력할 수 있다.
print('\n테스트 정확도: %.4f'%(model.evaluate(X_test, y_test)[1]))
