if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.layers as L

rnn=L.RNN(10)
x=np.random.rand(1,1)
h=rnn(x)
print(h.shape)#1,10_10번의 timestep에서의 hidden_state(출력)값

#RNN계층으로 신경망 모델 구현
from dezero import Model
import dezero.functions as F

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):#in_size없애서 알아서 x입력으로 판단. out_size는 출력 값 by fc
        super().__init__()
        self.rnn=L.RNN(hidden_size)
        self.fc=L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h=self.rnn(x)
        y=self.fc(h)
        return y

seq_data=[np.random.randn(1,1) for _ in range(1000)]#더미 시계열 데이터
xs=seq_data[0:-1]
ts=seq_data[1:]#한단계 앞선 데이터

model=SimpleRNN(10,1)

loss, cnt=0,0
for x, t in zip(xs, ts):
    y=model(x)
    loss+=F.mean_squared_error(y, t)

    cnt+=1
    if cnt==2:#2회 째에서 loss계산을 의미(timestep==2 not 10 fully.)
        model.cleargrads()
        loss.backward()
        break
#계산그래프가 너무 깊어지지 않도록 Truncated Backpropagation Through Time을 수행한다. Variable

import matplotlib.pyplot as plt
import dezero
"""노이즈가 낀 사인파 출력 예시
train_set=dezero.datasets.SinCurve(train=True)
print(len(train_set))
print(train_set[0])
print(train_set[1])
print(train_set[2])

xs=[example[0] for example in train_set]
ts=[example[1] for example in train_set]
plt.plot(np.arange(len(xs)), xs, label='xs')
plt.plot(np.arange(len(ts)), ts, label='ts')
plt.show()"""

max_epoch=100
hidden_size=100
bptt_length=30#몇개단위로 끊을건지
train_set=dezero.datasets.SinCurve(train=True)
seqlen=len(train_set)

model=SimpleRNN(hidden_size, 1)
optimizer=dezero.optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count=0,0

    for x,t in train_set:
        x=x.reshape(1,1)
        y=model(x)
        loss+=F.mean_squared_error(y,t)
        count+=1

        if count%bptt_length==0 or count==seqlen:#BPTT의 타이밍 조정
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss=float(loss.data)/count
    print('| epoch %d | loss %f'%(epoch+1, avg_loss))

xs=np.cos(np.linspace(0, 4*np.pi, 1000))#테스트데이터
model.reset_state()
pred_list=[]

with dezero.no_grad():
    for x in xs:
        x=np.array(x).reshape(1,1)
        y=model(x)
        pred_list.append(float(y.data))
plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
