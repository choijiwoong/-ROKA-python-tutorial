import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Sigmoid:
    def __init__(self):
        self.params=[]

    def forward(self, x):
        return 1/(1+np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params=[W, b]

    def forward(self, x):
        W, b=self.params
        out=np.matmul(x,W)+b
        return out

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O=input_size, hidden_size, output_size

        W1=np.random.randn(I, H)
        b1=np.random.randn(H)
        W2=np.random.randn(H, O)
        b2=np.random.randn(O)

        self.layers=[
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params=[]
        for layer in self.layers:
            self.params+=layer.params

    def predict(self, x):
        for layer in self.layers:
            x=layer.forward(x)
        return x

    def backward(self, dout=1):
        dout=self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
        return dout
#책에서 제공하는 Trainer 클래스, Optimizers 구경해보장 https://github.com/ExcelsiorCJH/DLFromScratch2

#[2. Trainer]는 다음 이시간에!
