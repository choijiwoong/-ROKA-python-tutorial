import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    if x.ndim==2:
        x=x-x.max(axis=1, keepdims=True)
        x=np.exp(x)
        x/=x.sum(axis=1, keepdims=True)
        
    elif x.ndim==1:
        x=x-np.max(x)#최대값을 빼주어 약간의 정규화
        x=np.exp(x)/np.sum(np.exp(x))#확률

    return x

def cross_entropy_error(y, t):
    if y.ndim==1:#손실 계산을 위한 reshaping.
        t=t.reshape(1, t.size)
        y=t.reshape(1, y.size)

    if t.size==y.size:#t가 one-hot일 경우 고려, integerize
        t=t.argmax(axis=1)

    batch_size=y.shape[0]

    cross_entropy=np.log(t[np.arange(batch_size), t]+1e-7)#cross_entropy구하기
    loss=-np.sum(cross_entropy)/batch_size

    return loss
