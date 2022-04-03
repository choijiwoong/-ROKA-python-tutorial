import numpy as np

class Variable:
    def __init__(self, data):
        self.data=data
"""
data=np.array(1.0)
x=Variable(data)
print(x.data)

x=np.array(1)
x.ndim#0

x=np.array([1,2,3])
x.ndim#1

x=np.array([[1,2,3],
            [4,5,6]])
x.ndim#2"""
