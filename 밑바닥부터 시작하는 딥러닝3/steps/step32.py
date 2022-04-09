#고차 미분을 위한 준비
"""대충 이 단계는 복습이라 요약하면 variable의 data와 grad는 ndarray를 저장하고
Function에서 generation기반으로 set_creator, inputs outputs로 연결함. 이때 output은 weakref임
generation을 잘 사용하기 위하여 funcs와 seen_set을 사용하며 funcs는 추가시마다 정렬됨
grad는 처음엔 대입, 나머진 +=로 처리해서 중복된 값도 덮어쓰기 안되게처리함"""

#고차미분을 위한 이론
#그냥 y미분한거처럼 미분해 나온 grad도 마찬가지로 backward계산 과정을 기록하고, 이를 역전파한다는 간단하면서 기발한 아이디어임.

#고차미분을 위한 구현_dezero/core.py에 구현. variable의 grad가 ndarray가 아닌ㄴ variable을 참조하게 한다.
#이를 좀 더 발전하기 위해 역전파의 역전파도 모드를 구현하여 불필요한 정보를 최소화한다.
#__init__에서 is_simple_core가 의미하는 것이 이것이다. 고차 미분을 구현한 버전과 아닌 버전이다.
