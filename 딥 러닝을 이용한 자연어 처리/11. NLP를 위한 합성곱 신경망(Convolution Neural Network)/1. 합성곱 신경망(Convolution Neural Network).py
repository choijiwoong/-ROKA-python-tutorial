""" 주로 비전분야에서 사용되지만, 1D 합성곱 신경망으로 자연어 처리, 문자 임베딩이 가능하다.
합성곱신경망은 크게 Convolution layer와 Pooling layer로 구성된다.
    1. 합성공 신경망의 대두
공간구조를 기존의 1파원 텐서로 변환하면 spatial structure정보가 유실되어 공간적인 구조정보를 보존하며 학습하는 방법으로 합성곱신경망을 사용한다.

    2. 채널(Channel)
이미지는 (높이, 너비, 채널)의 3차원 텐서로, 채널은 색 성분을 의미한다. 고로 빛의 3원색을 고려하여 이미지 텐서는 (28x28x3)과같은 3D Tensor이며, channel은 depth라고도 부른다.

    3. 합성곱 연산(Convolution operation)
이미지의 특징을 추출하는 것으로, Kernel(or filter)라는 nxm행렬이 이미지를 모두 훑으며(좌상->우하) 겹쳐지는 부분의 이미지와 커널의 원소값을 곱하여 모두 더한 값이다.
일반적으로 kernel의 크기는 3x3혹은 5x5이며, 한번의 연산을 step이라고 한다. 커널을 사용한 합성곱 연산의 결과를 Feature map이라고 하며
step 별 커널의 이동범위를 stride라 부르며 사용자가 지정할 수 있다. 이에 따라 Feature map에 크기가 달라진다. (feature map의 칸수가 step수)

    4. 패딩(Padding)
5x5이미지 conv 3x3커널 with stride=1->3x3 Feature map->입력보다 결과의 크기가 작아진다. 고로 이를 유지하고싶을 때 padding을 사용한다.
3x3 kernel에는 1폭짜리 zero padding, 5x5 kernel에는 2폭짜리 zero padding을 사용하면 크기보존이 가능하다.
ex) 5x5 img->(padding)7x7 img->conv 3x3kernel(stride=1)->5x5 img (패딩먼저하고 conv하여 연산결과를 유지한다는게 핵심인듯)

    5. 합성곱 신경망의 가중치
기존에 input_layer가 은닉층에 weight로 연결된 모양을 생각하면, 이미지에서는 입력 이미지에 곱해지는 kernel이 그 weight역활을 한다.
그러므로서 커널과 맵핑되는 픽셀만을 입력으로 사용하기에 다층 퍼셉트론보다 훨씬 적은 수의 가중치를 사용하여 spatial structure information을 보존한다.
 또한 마찬가지로 가중치 연산 후에 비선형성을 위해 활성화 함수를 지나게 되는데, 렐루나 리키렐루같은 것들이 사용된다.
이러한 conv연산으로 feature map을 얻고 activation function을 지나는 층을 convolution layer라고 한다.
 편향(bias)역시 추가가 가능하며 kernel을 적용한 feature map에 더해진다. 하나의 값만 존재한다.

    6. 특성 맵의 크기 계산 방법
입력의 크기, 커널의 크기, stride로 feature map크기계산이 가능하다.
특성맵 높이=floor((입력높이-커널높이+2패딩폭)/스트라이드)+1)
특성맵 너비=floor((입력너비-커널너비+2패딩폭)/스트라이드)+1)

    7. 다수의 채널을 가질 경우의 합성곱 연산(3차원 텐서의 합성곱 연산)
커널의 채널 수도 입력의 채널수는 같아야 하며, 채널마다 수행한 뒤 결과를 더하여 최종 특성 맵을 얻는다.
채널이 3개라고 가정했을때, 3개의 커널이 사용되는 것이 아닌 3개의 채널을 같이 가진 하나의 커널이 사용된다는 것에 유의해야한다.

    8. 3차원 텐서의 합성곱 연산
다수의 커널을 사용할 경우 사용한 커널 수는 특성맵의 채널수가 된다.
일반화하여 커널의가중치 매개변수의 총 개수(커널의 원소들)를 구할 수 있는데, 하나의 커널의 하나의 채널 크기Ki x Ko에 입력데이터 채널과
커널의 채널이 동일하니 Ki x Ko x Ci, 그리고 그러한 커널이 Co개 있다면 Ki x Ko x Ci x Co이 가중치 매개변수의 총 수가 된다.

 9. 풀링(Pooling)
일반적으로 합성곱 층(합성곱 연산+활성화 함수)이후 풀링층을 추가하여 feature map을 다운샘플링하여 크기를 줄인다. max pooling, average pooling, min pooling등이 있으며,
커널과 스트라이브 개념을 가진다. 다만 합성곱 연산과의 차이는 학습할 가중치가 없으며, 연산후에 채널수가 변하지 않는다는 것이다."""

#2. 자연어 처리를 위한 1D CNN(1D Convolutional Neural Networks)
""" 1. 2D 합성곱(2D Convolutions)
이미지를 커널이 훑으며 겹치는 부분의 값을 서로 곱하여 더한 값을 출력하는 것을 2D 합성곱 연산이라고 한다.

    2. 1D 합성곱(1D Convolutions)
LSTM과 같은 자연어처리를 위해선 embedding layer를 통해 embedding vector로 만들었다. 
기존의 LSTM은 embedding vector들을 하나씩 처리했다면, 1D 합성곱 연산에서는 커널의 높이만으로 해당 커널의 크기라고 간주한다(너비는 문장 임베딩벡터의 차원과 동일하기에)
1D 에다가 너비가 고정되어있기에 위아래로만 움직이며, 이를 마찬가지로 step이라고 한다. 이러한 과정을 반복하면 결과적으로 embedding layer와 비슷한 embedding vector를 얻을 수 있다.
 커널의 크기가 달라진다는 것은(한번의 스텝에 사용되는 단어 개수) Conv에서 kernel은 가중치와 같은 역활을 하기에 학습하게 되는 파라미터의 수가 달라진다는 것이다.
자연어 처리에서느 참고하는 n-gram이 달라진다고도 볼 수 있어 각 연산의 스템에서 참고하는 것을 bigram, trigram이라고 하기도 한다.

    3. 맥스 풀링(Max-pooling)
비전에서의 처리와 마찬가지로 자연어처리 1D CNN을 이용한 embedding vector의 최댓값을 꺼내온다.

    4. 신경망 설계하기
실제 텍스트 분류를 위한 CNN설계 시 이진 분류를 위한 신경망이지만 softmax를 사용하기에 출력층 뉴런의 개수가 2개이며,
여러 크기를 가지는 여러 커널로 conv연산을 한 뒤 모두 풀링을 진행, 얻은 스칼라 값들을 전부 연결(concatenate)하여 뉴런이 2개인 출력층의 입력으로 완전 연결시켜 텍스트 분류를 수행한다.

    5. 케라스(Keras)로 CNN 구현하기"""
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

model=Sequential()
model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))#padding은 valid or same을 사용.
model.add(GlobalMaxPooling1D())
