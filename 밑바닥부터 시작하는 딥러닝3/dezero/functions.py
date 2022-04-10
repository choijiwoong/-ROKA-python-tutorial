import numpy as np
from dezero.core import Function, as_variable
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
