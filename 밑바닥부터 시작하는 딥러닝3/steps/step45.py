if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#레이어 자체가 Parameter뿐이 아닌 레이어 자체도 품을 수 있게 확장.

# 기존의 신경망을 확장된 Layer로 구현
import dezero.layers as L
import dezero.functions as F

model=L.Layer()
model.l1=L.Linear(5)
model.l2=L.Linear(3)

def predict(model, x):
    y=model.l1(x)
    y=F.sifmoid(y)
    y=model.l2(y)
    return y

for p in model.params():
    print(p)

model.cleargrads()

# 아예 하나의 클래스로 정의하여 구현
import dezero.models as M
from dezero.core import Variable
import numpy as np

#class TwoLayerNet(L.Layer):
class TwoLayerNet(M.Model): # plot가능한 Model을 상속받아 구현
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1=L.Linear(hidden_size)
        self.l2=L.Linear(out_size)

    def forward(self, x):
        y=F.sigmoid(self.l1(x))
        y=self.l2(y)
        return y
x=Variable(np.random.randn(5,10), name='x')
model=TwoLayerNet(100,10)
model.plot(x)
