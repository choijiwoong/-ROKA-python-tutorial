import numpy as np
from common.util import *#im2col->4차원을 2차원으로
from chap5_Backpropagation import *
from collections import OrderedDict

class Convolution:#Conv연산+ReLU+Pooling이 국룰.
    def __init__(self, W, b, stride=1, pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad

    def forward(self, x):
        FN, C, FH, FW=self.W.shape#Filter의 개수, Channel수, 높이, 너비정보
        N, C, H, W=x.shape#데이터 개수, Channel수, 높이, 너비정보 get
        out_h=int(1+(H+2*self.pad-FH)/self.stride)#위 정보를 바탕으로 출력행렬의 높이, 너비 계산(Conv연산. pad와 stride, channel고려)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)

        col=im2col(x, FH, FW, self.stride, self.pad)#im2col로 병렬계산을 위한 입력 데이터 전처리(합성곱 연산에 필요한 정보 필터크기, 스트라이드, 패딩크기 등을 넘겨준다)
        col_W=self.W.reshape(FN, -1).T#im2col로 만들어진 2차원 데이터 병렬 연산을 위한 가중치 reshape(Channel)
        out=np.dot(col, col_W)+self.b#matmul

        out=out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)#Conv출력층의 형상으로 reshaping

        return out

class Pooling:#CNN의 출력의 특징을 추출(max구현)
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

    def forward(self, x):
        N, C, H, W=x.shape#입력 데이터 모냥 가져오기
        out_h=int(1+(H-self.pool_h)/self.stride)#풀링을 하면 특정 영역의 Max, Avg, Min값을 가지게 되니 출력층이 감소한다.
        out_w=int(1+(W-self.pool_w)/self.stride)#출력층의 크기 계산(stride고려, 일반적으로 Pooling의 window는 stride와 같다)

        col=im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)#병렬계산 전처리
        col=col.reshape(-1, self.pool_h*self.pool_w)#window별로 im2col을 통해 한줄씩 matrix모양으로 정렬되어있다.

        out=np.max(col, axis=1)#행별로 Max값을 저장하고

        out=out.reshape(M, out_h, out_w, C).transpose(0, 3, 1, 2)#출력의 모양으로 변환한다.

        return out

class SimpleConvNet:#입력 데이터의 차원, {필터수, 필터크기, 스트라이드, 패딩}, 은닉층뉴런수, 출력층뉴런수, 초기값 std(데이터, conv연산, Dense, Dense정보)
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        #인자정리
        filter_num=conv_param['filter_num']
        filter_size=conv_param['filter_size']
        filter_pad=conv_param['pad']
        filter_stride=conv_param['stride']
        input_size=input_dim[1]#?? 정사각형이라고 가정해서 높이, 너비 같은 값 사용하는듯.
        conv_output_size=(input_size-filter_size+2*filter_pad)/filter_stride+1#합성곱 출력 크기 계산
        pool_output_size=int(filter_num*(conv_output_size/2)*(conv_output_size/2))#풀링 출력 크기 계산

        #가중치 매개변수 초기화
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(filter_num, input_dim[0], filter_size, filter_size)#CNN(Kernel)
        self.params['b1']=np.zeros(filter_num)
        self.params['W2']=weight_init_std*np.random.randn(pool_output_size, hidden_size)#hidden
        self.params['b2']=np.zeros(hidden_size)
        self.params['W3']=weight_init_std*np.random.randn(hidden_size, output_size)#output
        self.params['b3']=np.zeros(output_size)

        #CNN계층 생성
        self.layers=OrderedDict()
        self.layers['Conv1']=Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1']=Relu()
        self.layers['Pool1']=Pooling(pool_h=2, pool_w=2, stride=2)#정사각형모양으로 pooling 그래서 SimpleCNNnet이구나.
        self.layers['Affine1']=Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2']=Relu()
        self.layers['Affine2']=Affine(self.params['W3'], self.params['b3'])
        self.last_layer=SoftmaxWithLoss()#softmax확률값.

    def predict(self, x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x

    def loss(self, x, t):
        y=self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):#Backpropagation
        self.loss(x,t)

        dout=1#backward를 역순으로 실행한 다음에
        dout=self.last_layer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)

        grads={}#각 레이어의 W, b미분결과들을 dictionary에 담아 리턴.
        grads['W1']=self.layers['Conv1'].dW
        grads['b1']=self.layers['Conv1'].db
        grads['W2']=self.layers['Affine1'].dW
        grads['b2']=self.layers['Affine1'].db
        grads['W3']=self.layers['Affine2'].dW
        grads['b3']=self.layers['Affine2'].db

        return grads
