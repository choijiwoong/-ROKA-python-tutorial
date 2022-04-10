if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#출력 크기계산함수
from dezero.utils import get_conv_outsize, pair

H,W=4,4
KH,KW=3,3
SH,SW=1,1
PH,PW=1,1

OH=get_conv_outsize(H, KH, SH, PH)
OW=get_conv_outsize(W, KW, SW, PW)
print(OH, OW)

#im2col(x, kernel_size, stride=1, pad=0, to_matrix=True)
import numpy as np
import dezero.functions as F
from dezero.core import Variable

x1=np.random.rand(1,3,7,7)
col1=F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)#9,75

x2=np.random.rand(10,3,7,7)
kernel_size=(5,5)
stride=(1,1)
pad=(0,0)
col2=F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)#90,75(배치가 10개라)

#pair(x): 원소2개 튜플로 변환 짝맞추기
print(pair(1))#1,1
print(pair((1,2)))#1,2

#conv2d_simple 테스트
N,C,H,W=1,5,15,15
OC, (KH, KW)=8, (3,3)

x=Variable(np.random.randn(N, C, H, W))
W=np.random.randn(OC, C, KH, KW)
y=F.conv2d_simple(x, W, b=None, stride=1, pad=1)#내부에서 전개된 im2col은 사용 후 즉시 삭제됨(conv2d로 묶어)
y.backward()

print(y.shape)
print(x.grad.shape)
