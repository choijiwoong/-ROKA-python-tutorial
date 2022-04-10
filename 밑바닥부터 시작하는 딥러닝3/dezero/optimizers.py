import numpy as np

class Optimizer:
    def __init__(self):
        self.target=None
        self.hooks=[]

    def setup(self, target):
        self.target=target
        return self

    def update(self):
        params=[p for p in self.target.params() if p.grad is not None]
        for f in self.hooks:#전처리(옵션)
            f(params)
        for param in params:#매개변수 갱신
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementError()

    def add_hook(self, f):
        self.hooks.append(f)
        
class SGD(Optimizer):#Stochastic Gradient Descent
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr=lr

    def update_one(self, param):
        param.data-=self.lr*param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr=lr
        self.momentum=momentum
        self.vs={}

    def update_one(self, param):
        v_key=id(param)
        if v_key not in self.vs:
            self.vs[v_key]=np.zeros_like(param.data)

        v=self.vs[v_key]#이전의 값에 momentum을 곱해 더해주는 느낌
        v*=self.momentum
        v-=self.lr*param.grad.data
        param.data+=v
