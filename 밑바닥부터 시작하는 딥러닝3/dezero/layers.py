import numpy as np
from dezero.core import Parameter
import dezero.functions as F
import weakref
import os

class Layer:
    def __init__(self):
        self._params=set()

    def __setattr__(self, name, value):#인스턴스 변수를 설정시 호출
        if isinstance(value, (Parameter, Layer)):#add전 체크(step45. Layer도 품게 확장)
            self._params.add(name)
        super().__setattr__(name ,value)

    def __call__(self, *inputs):#forward후 weakref로 inputs, outputs저장
        outputs=self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs=(outputs,)
        self.inputs=[weakref.ref(x) for x in inputs]
        self.outputs=[weakref.ref(y) for y in outputs]
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:#결론적으로 계속 호출하면 self._params의 name을 다 반환받을 수 있다.
            obj=self.__dict__[name]#현재의 오브젝트를 가져와서

            if isinstance(obj, Layer):#step.45
                yield from obj.params()#그 Layer의 params를 yield(재귀)
            else:
                yield obj#yield는 return처럼 사용이 가능하며, 작업을 마친 후가 아닌 일시정지하여 반환한다.

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):#매개변수(Variable)들마다 to_cpu실행
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=''):#for save recursively for layer in layer
        for name in self._params:
            obj=self.__dict__[name]
            key=parent_key+'/'+name if parane_key else name

            if isinstance(obj, Layer):#레이어면
                obj._flatten_params(params_dict, key)#재귀로 전달.(params_dict에 같이 기록)
            else:#일반적인 parameter면 바로 params_dict에 저장
                params_dict[key]=obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict={}
        self._flatten_params(params_dict)
        array_dict={key: param.data for key, param in params_dict.items() if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)#atomic
            raise#설마 rethrow??!! WOW Python빨리 제대로 배우고 싶다.

    def load_weights(self, path):
        npz=np.load(path)
        params_dict={}
        self._flatten_params(params_dict)#self.__dict__를 통해 params_dict에 name로드
        for key, param in params_dict.items():#그 뒤 내용집어넣기(초기엔 None)
            param.data=npz[key]

"""class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()

        I, O=in_size, out_size
        W_data=np.random.randn(I,O).astype(dtype)*np.sqrt(1/I)
        self.W=Parameter(W_data, name='W')
        if nobias:
            self.b=None
        else:
            self.b=Parameter(np.zeros(0, dtype=dtype), name='b')

    def forward(self, x):
        y=F.linear(x, self.W, self.b)
        return y"""
class Linear(Layer):#개선된 버전(_init_W를 forward시점으로 연기)
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size=in_size
        self.out_size=out_size
        self.dtype=dtype

        self.W=Parameter(None, name='W')
        if self.in_size is not None:#in_size가 있다면 init시점에 미리 초기화
            self._init_W()

        if nobias:
            self.b=None
        else:
            self.b=Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I,O=self.in_size, self.out_size
        W_data=np.random.randn(I,O).astype(self.dtype)*np.sqrt(1/I)
        self.W.data=W_data

    def forward(self, x):
        if self.W.data is None:#forward실행 시점까지 W가 초기화되지 않았다면 x값으로 초기화
            self.in_size=x.shape[1]
            self._init_W()

        y=F.linear(x, self.W, self.b)
        return y

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.pad=pad
        self.dtype=dtype

        self.W=Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b=None
        else:
            self.b=Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC=self.in_channels, self.out_channels
        KH, KW=pair(self.kernel_size)
        scale=np.sqrt(1/(C*KH*KW))
        W_data=xp.random.randn(OC,C,KH,KW).astype(self.dtype)*scale#Xaiver
        self.W.data=W_data

    def forward(self, x):
        if self.W.data is None:#init시 in_channel이 None일 경우 초기화를 지연했기에 지금 실행
            self.in_channels=x.shape[1]
            xp=cuda.get_array_module(x)
            self._init_W(xp)

        y=F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)
        return y

class RNN(Layer):#tanh(hW+xW+b)_편향 하나임
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h=Linear(hidden_size, in_size=in_size)#hidden_state크기, input data크기
        self.h2h=Linear(hidden_size, in_size=in_size, nobias=True)
        self.h=None#hidden_state

    def reset_state(self):
        self.h=None

    def forward(self, x):
        if self.h is None:
            h_new=F.tanh(self.x2h(x))
        else:
            h_new=F.tanh(self.x2h(x)+self.h2h(self.h))
        self.h=h_new
        return h_new#hidden_state가 곧 출력
    
class LSTM(Layer):#forget, input, output, how many(u, tanh)
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H,I=hidden_size, in_size
        self.x2f=Linear(H, in_size=I)
        self.x2i=Linear(H, in_size=I)
        self.x2o=Linear(H, in_size=I)
        self.x2u=Linear(H, in_size=I)
        self.h2f=Linear(H, in_size=H, nobias=True)#hW+xW+b
        self.h2i=Linear(H, in_size=H, nobias=True)
        self.h2o=Linear(H, in_size=H, nobias=True)
        self.h2u=Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h=None
        self.c=None

    def forward(self, x):
        if self.h is None:
            f=F.sigmoid(self.x2f(x))
            i=F.sigmoid(self.x2i(x))
            o=F.sigmoid(self.x2o(x))
            u=F.tanh(self.x2u(x))#how many remember
        else:
            f=F.sigmoid(self.x2f(x)+self.h2f(self.h))
            i=F.sigmoid(self.x2i(x)+self.h2i(self.h))
            o=F.sigmoid(self.x2o(x)+self.h2o(self.h))
            u=F.tanh(self.x2u(x)+self.h2u(self.h))

        if self.c is None:#첫 timestep
            c_new=(i*u)#전 cell_state가 없으니 input만 생각
        else:
            c_new=(f*self.c)+(i*u)#forget*cell_state+input*how many
        h_new=o*F.tanh(c_new)#cell_state에 tanh끼고 output이랑 곱하면 hidden_state

        self.h, self.c=h_new, c_new
        return h_new#c는 내부처리목적
