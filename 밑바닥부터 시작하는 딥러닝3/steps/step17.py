"""Python의 컴파일러 즉 파이썬 인터프리터의 메모리 관리는 크게 refernce count, generation기반 GC가 있다.
현재의 DeZero구조는 Function.outputs와 Variable.creator가 순환참조를 이루는 구조이기에 개선이 필요하다.
C++과 마찬가지로 weakref를 이용하여 이를 해결할 수 있다."""
import weakref
import numpy as np

a=np.array([1,2,3])
b=weakref.ref(a)

print(b)#정보
print(b())#값접근
a=None
print(b)#dead!

#의사코드
class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        self.generation=max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs
        self.outputs=[weakref.ref(output) for output in outputs]#Function의 output을 가리키는 변수를 weakref로서 설정
        return outputs if len(outputs)>1 else output[0]
    ...

class Variable:
    ...
    def backward(self):
        ...
        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]#weakref는 그 자체로는 weak_ptr이고 ()을 통해 값에 접근한다
        
#weakref로 Function과 Variable간의 순환참조문제를 해결해보았다. 동작 확인은 아래와 같다.
for i in range(10):
    x=Variable(np.random.randn(10000))#많은 데이터이기에 기존의 구조에서는 메모리 사용량이 증가한다. 
    y=square(square(square(x)))#하지만 현재 for루프가 시작될 때 마다 참조 카운트가 0이 되므로 메모리가 바로 삭제된다.
#메모리 사용량 측정을 위해선 외부 라이브러리 memory profiler을 사용할 수 있다.
    
