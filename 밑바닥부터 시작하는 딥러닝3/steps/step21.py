#ndarray와의 연산자 사용, 수치 데이터와의 연산자 사용->as_variable로 모든 입력 variable로 변환
#수치데이터와 ndarray의 피연산위치가 앞쪽이라 앞쪽 기준 메서드가 실행될 경우 선자는 rmul처럼 순서바뀐 메서드를 정의하여,
#후자는 ndarray의 연산자 우선순위보다 높은 우선순위를 Variable에 할당하여 해결한다.

#의사코드
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs):
        inputs=[as_variable(x) for x in inputs]#들어온 모든 입력의 타입이 Variable이 아닐 경우 변경. for 수치데이터, ndarray...etc

        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        ...

x=Variable(np.array(2.0))
y=x+np.array(3.0)
print(y)#well done


def add(x0, x1):#함수에 관해서도 피연산자 입력값을 array로. 즉 Variable인스턴스를 float과같은 타입의 조합도 계산이 가능해졌다.
    x1=as_array(x1)
    return Add()(x0, x1)

x=Variable(np.array(2.0))
y=x+3.0
print(y)

#좌항이 float이나 int와 같아 그들의 연산메서드가 실행될 시에는 r(reverse)메서드를 정의하자
Variable.__add__=add
Variable.__radd__=add
Variable.__mul__=mul
Variable.__rmul__=mul

x=Variable(np.array(2.0))
y=3.0*x+1.0
print(y)#well done!

#좌항이 ndarray인 경우는 아래와 같다.
x=Variable(np.array([1.0]))
y=np.array([2.0])+x#이 경우엔 ndarray의 연산자오버로딩된 mul메서드가 실행되기에 Variable의 우선순위를 ndarray보다 크게 설정하여 해결한다.

class Variable:
    __array_priority__=200#해결
