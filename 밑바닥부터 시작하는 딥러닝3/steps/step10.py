import unittest#표준 테스트 라이브러리
from step09 import *

def nemerical_diff(f, x, epos=1e-4):#수치미분, 중앙차분. backward확인용
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)

class SquareTest(unittest.TestCase):#TestCase를 상속하고, test로 시작하는 메서드를 만들면 된다.
    def test_forward(self):
        x=Variable(np.array(2.0))
        y=square(x)
        expected=np.array(4.0)
        self.assertEqual(y.data, expected)#unitest.TestCase에는 다양한 메서드들이 내장되어있다.

    def test_backward(self):
        x=Variable(np.array(3.0))
        y=square(x)
        y.backward()#y에 저장되어있는 Square()객체의 backward메서드 실행
        expected=np.array(6.0)#y=x^2 y'=2*x
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x=Variable(np.random.rand(1))
        y=square(x)
        y.backward()#backwarding
        num_grad=numerical_diff(square, x)#수치미분
        flg=np.allclose(x.grad, num_grad)#둘의 값이 얼마나 가까운지를 판점하는 함수로 인수 rtol과 rtol로 지정이 가능하다
        self.assertTrue(flg)#|a-b|<=(atol+rtol*|b|)
    
"""
실행은 python -m unittest steps/step10.py로 파이썬을 테스트모드로 실행이 가능하다.
장점은 코드를 수정할 때마다 즉시즉시 테스트를 실행하여 반복적인 상태확인을 통한 버그의 예방 및 발견이다
테스트코드는 하나의 디렉토리로 모아 관리하는 것이 일반적이며, python -m unittest discover tests명령으로 테스트파일들을 일괄실행할 수 있다.
이름이 test*.py인 형태를 모두 인식한다."""        
