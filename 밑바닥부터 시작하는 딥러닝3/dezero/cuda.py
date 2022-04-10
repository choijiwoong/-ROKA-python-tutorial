import numpy as np
gpu_enable=True
try:
    import cupy as cp
    cupy=cp
except ImportError:
    gpu_enable=False
import dezero

def get_array_module(x):
    if isinstance(x, dezero.core.Variable):
        x=x.data#데이터만 쏙

    if not gpu_enable:#gpu사용 불가면 numpy로
        return np
    xp=cp.get_array_module(x)#쿠파이의 적합합 모듈 반환 이용
    return xp

def as_numpy(x):
    if isinstance(x, dezero.core.Variable):
        x=x.data

    if npisscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)#cupy->numpy

def as_cupy(x):
    if isinstance(x, dezero.core.Variable):
        x=x.data

    if not gpu_enable:
        raise Exception('쿠파이(CuPy)를 로드할 수 없습니다. 쿠파이를 설치해주세요!')
    return cp.asarray(x)#numpy->cupy
