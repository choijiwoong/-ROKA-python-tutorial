if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

#사인 함수의 고차 미분(feat. 가시화)
x=Variable(np.linspace(-7,7,200))#-7~7까지 균등하게 200등분
y=F.sin(x)
y.backward(create_graph=True)

logs=[y.data]

for i in range(3):
    logs.append(x.grad.data)#(grad는 Variable을 가리킴)
    gx=x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    #print(x.grad)
labels=['y=sin(x)', "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()
