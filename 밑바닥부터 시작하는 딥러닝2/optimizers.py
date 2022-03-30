import numpy as np

#[1. Optimizer]
class SGD:#(기울기만큼 내리기)
    def __init__(self, lr=0.01):
        self.lr=lr#learning rate저장

    def update(self, params, grads):
        for i in range(len(params)):#모든 매개변수에 대하여
            params[i]-=self.lr*grads[i]#기울기 lr만큼 하강시킴.

class Momentum:#(관성 도입)
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr
        self.momentum=momentum
        self.v=None

    def update(self, params, grads):
        if self.v is None:#필요시 초기화
            self.v=[]
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i]=self.momentum*self.v[i]-self.lr*grads[i]#일반적인 SGD를 진행하되, 이전의 v값(파라미터의 값이랄까.. 기존의 값?)을 momentum만큼 더해준다.
            params[i]+=self.v[i]#parameter에 반영.

class Nesterov:#(가속 경사)NAG*Nesterow's Accelerated Gradient
    def __init__(self, lr=0.01, momentum=0.9):#Momentum과 동일
        self.lr=lr
        self.momentum=momentum
        self.v=None

    def update(self, params, grads):
        if self.v is None:#필요시 포기화
            self.v=[]
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i]*=self.momentum
            self.v[i]-=self.lr*grads[i]#까지 Momentum과 동일
            params[i]+=self.momentum*self.momentum*self.v[i]#모멘텀을 제곱한 값을 더하여 params에 더한다.
            params[i]-=(1+self.momentum)*self.lr*grads[i]#momentum배수된 경사하강을 실시하여 params에서 뺀다.
            #가속 경사라는게 이전의 값에 momentum..을 제곱으로 곱하면 더 작아질거고, 기울기의 값을 (1+momentum)을 곱해버려 빼면... momentum의 구조는 유지하되, 줄어드는 경사가 커지겠구나
            #반영되는 관성값을 줄이고, 하강되는 경사값을 모멘텀배수하여 확 내리는 가속 경사 하강법

class AdaGrad:#(이전값 반영) 학습률을 element마다 다르게 적용한다.
    def __init__(self, lr=0.01):
        self.lr=lr
        self.h=None

    def update(self, params, grads):
        if self.h is None:#필요시 초기화
            self.h=[]
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i]+=grads[i]*grads[i]#해당 element의 grad를 제곱한값을
            params[i]-=self.lr*grads[i]/(np.sqrt(self.h[i])+1e-7)#sqrt하여 lr을 나눈다.
            #이전에 많이 움직였다면 grad값이 클것이고 이를 제곱하여 경사하강시 나눠버리기에 느려지고, 적게 움직였다면 보다 빠르게 움직이게 한다.
            #다만 계속 학습시킬 시 h가 누적되어 사실상 학습이 이루어지지 않는다.
            
class RMSprop:#(이전값을 서서히 잊고, 최신 기울기를 많이 반영한다.)
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr=lr
        self.decay_rate=decay_rate
        self.h=None

    def update(self, params, grads):
        if self.h is None:#필요시 초기화
            self.h=[]
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i]*=self.decay_rate#2) 기존의 상쇄 값(h)을 약간 감소시킨 뒤
            self.h[i]+=(1-self.decay_rate)*grads[i]*grads[i]#3) 현재의 기울기를 많이 반영하게 한다. 이 h값은 분모로 들어가기에 작을수록 실제 반영값이 커짐.
            params[i]-=self.lr*grads[i]/(np.sqrt(self.h[i])+1e-7)#1) AdaGrad와 같지만

class Adam:#(AdaGrad+Momentum) 잘 이해가 가지 않는다!
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.iter=0#
        self.m=None#
        self.v=None#

    def update(self, params, grads):
        if self.m is None:#필요시 m, v 초기화
            self.m, self.v=[], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter+=1
        lr_t=self.lr*np.sqrt(1.0-self.beta2**self.iter)/(1.0-self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i]+=(1-self.beta1)*(grads[i]-self.m[i])
            self.v[i]+=(1-self.beta2)*(grads[i]**2-self.v[i])

            params[i]-=lr_t*self.m[i]/(np.sqrt(self.v[i])+1e-7)
