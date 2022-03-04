""" 1. 데이터의 양을 늘린다
데이터의 양이 적으면 특정 패턴이나 노이즈까지 훈련하게 되지만, 많으면 일반적인 패턴을 학습한다. 데이터를 늘리는 방법중 기존의 데이터를
조금씩 변형하고 추가하는 것을 데이터 증식 또는 증강(Data Augmentation)이라고 한다. 이러한 번역을 거친 다음 또 재번역을 하는 Back Translation방법도 존재한다.

    2. 모델의 복잡도(parameter)를 줄인다.
인공신경망의 복잡도는 hidden layer의 수나 매개변수의 수 등으로 결정되기 때문이다. (인공신경망 모델의 parameter수를 capacity라고도 부른다)

    3. 가중치 규제(Regularization) 적용하기
L1 규제(노름): 가중치 w들의 절댓값 합계를 비용함수에 추가한다.
L2 규제(노름): 모든 가중치 w들의 제곱합을 비용함수에 추가한다.
이들의 규제의 강도를 정하는 하이퍼파라미터로서 λ를 사용하며, λ|w|, 1/2 λw^2처럼 사용한다.
λ가 크다면 적합한 매개변수보다도 규제를 위한 항들을 작게 유지하는걸 우선으로 한다는 의미이다.

L2 노름과 L1 노름 전부 비용함수를 최소화하기 위해서 weight이 작아져야하는 특징이 있는데, 이때 가중치가 0이 되면 모델 결과에 아무런 영향을 끼치지 못하게 된다.
다만 L2 규제는 L1규제보다 나은게 제곱을 최소화하므로 완전한 0보다 0에 수렴하는 경향을 가진다. L1 노름은 모델에 영향을 주는 특성을 알아낼 때 유용하며,
대부분의 경우 L2 노름을 권장하며 이를 weight decay라고도 부른다.

    4. 드롭아웃(Dropout)
신경망 학습 시, 특정 뉴런 또는 특정 조합에 너무 의존적이게 되는 것을 방지하기 위해 드롭아웃 비율을 통해 신경망의 일부를 사용하지 않는 방법이다."""
#케라스에서 드롭아웃을 모델에 추가하는 방법
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense

max_words=10000
num_classes=46

model=Sequential()
model.add(Dense(256, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())
