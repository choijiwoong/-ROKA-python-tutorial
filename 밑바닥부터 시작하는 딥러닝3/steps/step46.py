if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#SGD를 사용한 문제 해결
import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

np.random.seed(0)
x=np.random.rand(100,1)
y=np.sin(2*np.pi*x)+np.random.rand(100,1)

lr=0.2
max_iter=10000
hidden_size=10

model=MLP((hidden_size, 1))
optimizer=optimizers.SGD(lr)
optimizer.setup(model)
#or optimizer=optimizers.SGD(lr).setup(model)
for i in range(max_iter):
    y_pred=model(x)
    loss=F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()#정해진 update로 갱신(hooks & update_one. SGD같은 경우 -=lr*pa.gr.da)
    if i%1000==0:
        print(loss)
