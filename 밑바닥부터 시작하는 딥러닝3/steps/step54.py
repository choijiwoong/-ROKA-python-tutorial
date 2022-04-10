if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#드롭아웃
import numpy as np
"""
dropout_ratio=0.6
x=np.ones(10)

mask=np.random.rand(10)>dropout_ratio#이만큼을 거를거기에 나머지 영역을 마스킹
y=x*mask

#학습시와 테스트시의 차이
mask=np.random.rand(*x.shape)>dropout_ratio
y=x*mask

scale=1-dropout_ratio#학습시에 살아남은 뉴런의 비율(테스트비율과 동일하게 설정한다) 1에서 빼는 이유는 학습시에 mask는 거를거, 지금은 사용할거기에 같게하려고 1에서 뺀다
y=x*scale

#역드롭아웃: 스케일 조정을 학습시 미리 수행해서 테스트에 그냥 적용하게 하게끔(테스트 속도가 존나 약간 상승한다, 학습시 동적변경가능)
scaled=1-dropout_ratio
mask=np.random.rand(*x.shape).dropout_ratio
y=y*mask/scale#학습시

y=x#테스트시
"""
#test_mode를 core에 추가 및 functions에 구현
from dezero import test_mode
import dezero.functions as F

x=np.ones(5)
print(x)

y=F.dropout(x)#학습시
print(y)

with test_mode():#테스트시
    y=F.dropout(x)
    print(y)
