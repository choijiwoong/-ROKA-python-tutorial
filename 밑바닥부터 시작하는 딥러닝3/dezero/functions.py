import numpy as np
from dezero.core import Function, as_variable, Variable, as_array
from dezero import utils
from dezero import cuda

class Sin(Function):
    def forward(self, x):
        y=np.sin(x)
        return y

    def backward(self, gy):
        x,=self.inputs#모든 변수가 Variable이다.
        gx=gy*cos(x)
        return gx
def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y=np.cos(x)
        return y

    def backward(self, gy):
        x,=self.inputs
        gx=gy*-sin(x)
        return gx
def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y=np.tanh(x)
        return y

    def backward(self, gy):
        y=self.outputs[0]()
        gx=gy*(1-y*y)#미분은 1-y^2
        return gx
def tanh(x):
    return Tanh()(x)

class Exp(Function):
    def forward(self, x):
        xp=cuda.get_array_module(x)
        y=xp.exp(x)
        return y

    def backward(self, gy):
        y=self.outputs[0]()
        gx=gx*y
        return gx
def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        xp=cuda.get_array_module(x)
        y=xp.log(x)
        return y

    def backward(self, gy):
        x,=self.inputs
        gx=gx/x
        return gx
def log(x):
    return Log()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape=shape

    def forward(self, x):
        self.x_shape=x.shape#backward를 위해 원래 형상 백업
        y=x.reshape(self.shape)#init시 입력된 모양으로 reshaping(np의 reshape호출)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)#백업된 형상으로 다시 복원
def reshape(x, shape):
    if x.shape==shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):#np.transpose내부적 사용
    def __init__(self, axes=None):#transpose적용할 축.(다차원의 경우)
        self.axes=axes#backward시 다시 원상복구해야되서 저장
    
    def forward(self, x):
        y=x.transpose(self.axes)#np.transpose호출시 init하며 저장한 축 반영
        return y

    def backward(self, gy):
        if self.axes is None:#별도로 지정한 축이 없을 경우 
            return transpose(gy)

        axes_len=len(self.axes)
        inv_axes=tuple(np.argsort([ax%axes_len for ax in self.axes]))#index정렬
        return transpose(gy, inv_axes)#값을 기준으로 정렬된 축의 index들을 np.transpose에 전달
def transpose(x, axes=None):
    return Transpose(axes)(x)
        
def transpose(x):
    return Transpose()(x)

"""
class Sum(Function):#벡터에서의 미분 시 원소별 복사(브로드캐스팅)역시 Function으로 만들어야하기에
    def forward(self, x):
        self.x_shape=x.shape
        y=x.sum()
        return y

    def backward(self, gy):
        gx=broadcast_to(gy, self.x_shape)#이거
        return gx
def sum(x):
    return Sum()(x)
"""
class Sum(Function):
    def __init__(self, axis, keepdims):#미세한 조정을 위한 인자 저장
        self.axis=axis
        self.keepdims=keepdims

    def forward(self, x):
        self.x_shape=x.shape
        y=x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy=utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)#axis와 keepdims로 인한 형상변화 복구(broadcast_to 적용 전)
        gx=broadcast_to(gy, self.x_shape)
        return gx
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroascastTo(Function):
    def __init__(self, shape):
        self.shape=shape

    def forward(self, x):
        self.x_shape=x.shape
        y=np.broadcast_to(x, self.shape)#크기명시적 브로드캐스팅
        return y

    def backward(self, gy):
        gx=sum_to(gy, self.x_shape)#broadcast_to의 미분은 sum_to
        return gx
def broascast_to(x, shape):
    if x.shape==shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape=shape

    def forward(self, x):
        self.x_shape=x.shape#기존형상 백업
        y=utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx=broadcast_to(gy, self.x_shape)
        return gx
def sum_to(x, shape):
    if x.shape==shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y=x.dot(W)
        return y

    def backward(self, gy):
        x, W=self.inputs
        gx=matmul(gy, W.T)
        gW=matmul(x.T, gy)
        return gx, gW
def matmul(x, W):
    return MatMul()(x,W)

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff=x0-x1
        y=(diff**2).sum()/len(diff)
        return y

    def backward(self, gy):
        x0, x1=self.inputs
        diff=x0-x1
        gx0=gy*diff*(2./len(diff))
        gx1=-gx0
        return gx0, gx1
def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def linear_simple(x, W, b=None):
    t=matmul(x, W)
    if b is None:
        return t

    y=t+b
    t.data=None#중간데이터 삭제. 불필요한 인스턴스는 삭제하는것이 좋으며 Aggressive Buffer Release구조등을 통해 자동화 가능하다.
    return y

class Linear(Function):
    def forward(self, x, W, b):
        y=x.dot(W)
        if b is not None:
            y+=b
        return y

    def backward(self, gy):
        x, W, b=self.inputs
        gb=None if b.data is None else sum_to(gy, b.shape)#브로드캐스팅 미분 sum_to
        gx=matmul(gy, W.T)
        gW=matmul(x.T, gy)
        return gx, gW, gb
def linear(x, W, b=None):
    return Linear()(x, W, b)


def sigmoid_simple(x):
    x=as_variable(x)
    y=1/(1+exp(-x))
    return y

class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y=xp.tanh(x*0.5)*0.5+0.5#위 연산과 결과가 동일하다
        return y

    def backward(self, gy):
        y=self.outputs[0]()
        gx=gy*y*(1-y)
        return gx
def sigmoid(x):
    return Sigmoid()(x)

def softmax_simple(x, axis=1):
    x=as_variable(x)
    y=exp(x)
    sum_y=sum(y, axis=axis, keepdims=True)
    return y/sum_y

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis=axis

    def forward(self, x):
        xp=cuda.get_array_module(x)
        y=x-x.max(axis=self.axis, keepdims=True)#정규화
        y=xp.exp(y)
        y/=y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y=self.outputs[0]()
        gx=y*gy
        sumdx=gx.sum(axis=self.axis, keepdims=True)
        gx-=y*sumdx
        return gx
def softmax(x, axis=1):
    return Softmax(axis)(x)


def softmax_cross_entropy_simple(x, t):
    x, t=as_variable(x), as_variable(t)
    N=x.shape[0]

    p=softmax_simple(x)
    p=clip(p, 13-15, 1.0)#gradient_cliping
    log_p=log(p)
    tlog_p=log_p[np.arange(N), t.data]#t에 해당하는 요소 log_p에서 추출
    y=-1*sum(tlog_p)/N#평균적 손실
    return y

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N=x.shape[0]
        log_z=utils.logsumexp(x, axis=1)#
        log_p=x-log_z#정규화
        log_p=log_p[np.arange(N), t.ravel()]#t에 해당하는거만 남기기(손실?)
        y=-log_p.sum()/np.float32(N)#평균
        return y

    def backward(self, gy):
        x, t=self.inputs
        N, CLS_NUM=x.shape

        gy*=1/N
        y=softmax(x)
        xp=cuda.get_array_module(t.data)
        t_onehot=xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y=(y-t_onehot)*gy
        return y
def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

class ReLU(Function):
    def forward(self, x):
        y=np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x,=self.inputs
        mask=x.data>0
        gx=gy*mask
        return gx
def relu(x):
    return ReLU()(x)

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min=x_min
        self.x_max=x_max

    def forward(self, x):
        xp=cuda.get_array_module(x)
        y=xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x,=self.inputs
        mask=(x.data>=self.x_min)*(x.data<=self.x_max)
        gx=gy*mask
        return gx
def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

#슬라이싱
class GetItemGrad(Function):#backward에 사용
    def __init__(self, slices, in_shape):
        self.slices=slices#마찬가지 2중 미분 대비 backward구현
        self.in_shape=in_shape

    def forward(self, gy):
        gx=np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)#gx에 슬라이싱했던 부분에 gy를 더한다.
        return gx#슬라이싱 부분에만 gy가 누산된 ndarray를 반환

    def backward(self, ggx):
        return get_item(ggx, self.slices)

class GetItem(Function):
    def __init__(self, slices):
        self.slices=slices#for backward

    def forward(self, x):
        y=x[self.slices]
        return y

    def backward(self, gy):
        x,=self.inputs
        f=GetItemGrad(self.slices, x.shape)#slice를 참고하여 backward결과 만들어주는 함수 리턴
        return f(gy)

def get_item(x, slices):
    f=GetItem(slices)
    return f(x)

def accuracy(y, t):
    y, t=as_variable(y), as_variable(t)

    pred=y.data.argmax(axis=1).reshape(t.shape)#원핫화
    result=(pred==t.data)#비교
    acc=result.mean()#평균
    return Variable(as_array(acc))#정확도 짜자잔
