#[지도 학습(Supervised Learning)]
#일반적으로 총 4개의 데이터 X_train, y_train, X_test, y_test로 분리한다. 평가에 사용되는 test결과가 모델의 Accuracy가 된다.


#[X와 y 분리하기]
 #zip함수 이용
X, y=zip(['a', 1], ['b', 2], ['c', 3])#zip()은 동일한 개수를 가지는 시퀀스 자료형에서 각 순서에 등장하는 원소들끼리 묶어준다.
print('[X와 y분리하기]\nX 데이터: ', X)
print('y 데이터: ', y)

sequences=[['a', 1], ['b', 2], ['c', 3]]
X, y=zip(*sequences)#위치인자 패킹. 위치인자로 보낸 모든 객체들을 하나의 객체로 관리해준다.
print('X 데이터: ', X)
print('y 데이터: ', y, end='\n\n')

 #데이터프레임 이용
import pandas as pd

values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df=pd.DataFrame(values, columns=columns)
print('df: ', df)

X=df['메일 본문']#데이터프레임은 열의 이름으로 각 열의 접근이 가능하다.
y=df['스팸 메일 유무']

print('X 데이터: ', X.to_list())
print('y 데이터: ', y.to_list(), end='\n\n')

 #Numpy 이용
import numpy as np

np_array=np.arange(0,16).reshape((4,4))
print("전체 데이터: ", np_array)

X=np_array[:, :3]#X가 마지막 열 제외 데이터라 가정, slice
y=np_array[:, 3]#마지막 열이 y라 가정.

print('X 데이터: ', X)
print('y 데이터: ', y, end='\n\n\n')


#[(이미 분리된 X와 y데이터에서)테스트 데이터 분리하기]
 #sklearn.train_test_split 사용 
from sklearn.model_selection import train_test_split#사이킷 런에선 테스트데이터분리를 도와주는 train_test_split지원.

 #train_test_split은 아래와 같이 사용한다.
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1234)#train_size와 test_size는 둘중 하나만 해도 됨.
#X(독립변수데이터), y(종속변수데이터), test_size(테스트데이터개수_비율가능), train_size(학습데이터개수, 비율가능), random_state(난수시드)


X, y= np.arange(10).reshape((5,2)), range(5)
print('[(이미 분리된 X와 y데이터에서)테스트 데이터 분리하기]\nX 전체 데이터: ', X, '\ny 전체 데이터: ', list(y))
#임의의 데이터를 생성했고, 7:3의 비율로 데이터를 분리할 것인데, train_test_split는 기본적으로 데이터의 순서를 섞고 분리한다. 그렇기에 random_state를 사용한다.

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=1234)
print('\n(by using sklearn.train_test_split as 7:3)\nX 훈련 데이터: ', X_train, "\nX 테이스 데이터: ", X_test)
print('Y 훈련 데이터: ', y_train, 'Y 테스트 데이터: ', y_test, end='\n\n\n')

#뭐 당연하겠지만 만약 구현재현을 하고 싶다면 동일한 random_state를 설정하면 된다.

 #수동으로 분리 8:2
X, y=np.arange(0,24).reshape((12,2)), range(12)#임의의 데이터 생성
print('(수동으로 분리)\nX 전체 데이터: ', X, '\ny 전체 데이터: ', list(y), end='\n\n')

num_of_train=int(len(X)*0.8)#자를 개수부터 정하기(배율에 맞추어)
num_of_test=int(len(X)-num_of_train)
print('훈련 데이터의 크기: ', num_of_train, '\n테스트 데이터의 크기: ', num_of_test, end='\n\n')

X_test=X[num_of_train:]#slice로 분리.
y_test=y[num_of_train:]
X_train=X[:num_of_train]
y_train=y[:num_of_train]

print('X 테스트 데이터: ', X_test, '\ny 테스트 데이터: ', list(y_test))
#특징은 sklearn의 train_test_split은 default로 shuffle을 한 반면, 크기를 기준으로 slice하여 나눈 수동버전은 shuffle없이 앞뒤로 나뉜다는 차이가 있다.
