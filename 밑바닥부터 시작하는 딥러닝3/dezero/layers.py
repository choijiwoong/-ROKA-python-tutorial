import numpy as np
from dezero.core import Parameter
import dezero.functions as F
import weakref

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
