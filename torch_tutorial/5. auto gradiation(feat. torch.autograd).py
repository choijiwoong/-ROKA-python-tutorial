"""
역전파 시 손실 함수의 변화도(gradient)에 따라 weight가 조정되기에 미분이 필요하다.
엔트로피는 불확실성의 척도로 어떤 데이터가 나올지 예측이 어렵다, 즉 손실함수의 리턴값이 높다라는 의미로 사용된다.
Cross-Entroy는 실제분포 q에 대하여 알지 못하는 상태에서, 모델링으로 구한 분포인 p를 통하여 q를 예측하는 것이다. 단순히 p랑 q가 모두 들어가 크로스 엔트로피라한다.
이는 머신러닝 모델은 p로 예측했는데, 실제는 q라는 사실을 알고 있을 때 손실함수로 사용한다.
 아래에 사용되는 binary_cross_entropy_with_logits는 function객체이며, forward & backward계산을 가지고 있으며, 역방향 전파함수의 참조자는 grad_fn속성에 위치한다.
Gradient계산 시 requires_grad=True인 노드들만 구할 수 있으며, 성능상의 이유로 backward사용은 한번만 가능하다. 만약 여러 backward가 필요하면 retrain_graph=True를 backward호출 인자로 전달하면 된다.
기본적으로 이러한 requires_grad=True텐서들은 연산기록을 추적하고, gradient기능을 지원하는데, 모델 학습 뒤 추가적인 학습(backwarding)없이 적용(forward)만 할 경우
연산 코드를 torch.no_grad()블록으로 감싸서 불필요한 지원과 추적을 멈출 수 있다. 다른 방법으로는 detach()메소드를 사용하여 이러한 지원, 추적을 멈출 수 있긴 하다.
 대표적으로 변화도 추적을 멈추는 경우는 사전학습된 신경망을 미세조정할 때 일부 매개변수를 frozen parameter로 표시하는 경우, 연산속도 향상의 경우가 있다.
"""
import torch

x=torch.ones(5)
y=torch.zeros(3)
w=torch.randn(5,3, requires_grad=True)#parameter optimization for calculation of gradient
b=torch.randn(3, requires_grad=True)#나중에 x.requires_grad_(True)로도 설정할 수 있다. 해당 torch자료형의 requires_grad값만 True로 설정하는 기능이다.
z=torch.matmul(x,w)+b
loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
#loss=CE(x*w+b, y)

print("Gradient function for z= ", z.grad_fn)#show z's functor for backwarding
print("Gradient function for loss= ", loss.grad_fn)

#calculate Gradient
loss.backward()#calculate!
print("w.grad: ", w.grad, "\nb.grad: ", b.grad)

#Stop track of Gradient (being useless after using only forwarding) by wrapping torch.no_grad() block!
z=torch.matmul(x,w)+b
print("\n\nCurrent z's requires_grad state: ",z.requires_grad)
with torch.no_grad():#way1
    z=torch.matmul(x,w)+b
print("Current z's requires_grad state by detach: ",z.requires_grad)

z_det=z.detach()#way2
print("Current z's requires_grad state by detach: ",z_det.requires_grad)


#(p.s) Gradient of tensor & Jacobian Product
"""
대부분 스칼라 손실 함수로 일부 매개변수 변화도를 계산하지만, 출력함수가 임의의 텐서인 경우 실제 변화도가 아닌 야코비안 곱을 계산한다.
x증분에 대한 y증분 즉 변화도는 야코비안 행렬로 주어지며, pytorch는 입력벡터에 대한 야코비안 곱을 계산한다.
 즉, 이 야코비안 곱은 gradient를 값이 아닌 x와y가 대응되는 각 야코비안 행렬을 반환해야하는 경우 사용된다
동일한 인자로 backward하면 backpropagation시 pytorch가 변화도를 누적해두기에 변화도 값이 달라진다.
고로 제대로 된 변화도를 계산하기 위해서는 grad를 먼저 0으로 만들어야 하는데, 실제 과정에선 optimizer가 이 과정을 수행한다.

야코비안 행렬은 비선형을 선형으로 근사시킨 행렬이다. 아래 결과 이유를 잘 모르겠네..
"""
inp=torch.eye(5, requires_grad=True)#대각 1
print("\ninitial value of inp: \n", inp)
out=(inp+1).pow(2)
print("\ninitial value of out: \n", out)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
