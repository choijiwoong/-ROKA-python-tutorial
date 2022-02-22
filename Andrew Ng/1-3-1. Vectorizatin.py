import numpy as np

a=np.array([1,2,3,4])
print(a)

import time
a=np.random.rand(1000000)
b=np.random.rand(1000000)

tic=time.time()
c=np.dot(a,b)
toc=time.time()

print(c)
print('Vectorized version: ', str(1000*(toc-tic)))


c=0
tic=time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc=time.time()
print(c)
print("For loop: ", str(1000*(toc-tic)))

#CPU와 GPU에서 SIMD(Single Instruction Multiple Data)명령어(dot같은거)는 병렬화의 장점을 이용하여 벡터화 연산을 보다 빠르게 가능하게 한다.


#np.exp(), np.log(), np.abs(), np.maximum(v,0), v**2, 1/v Numpy내장 함수를 사용하여 병렬화의 장점을 최대한 사룔하자.
