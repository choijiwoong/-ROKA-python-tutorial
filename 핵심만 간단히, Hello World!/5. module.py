#module importing like concept of namespace
import random
print(random)#normal importing
print()

import sys as system_library#importing with namespace
print(system_library.path)
print()

from os import getcwd#importing partly(or *)
print(getcwd())

#만약 모듈로서 사용되는 것을 금지하고, 최종 파일(main)으로서 실행되고 싶다면
def su(a,b):
    if __name__=="__main__":
        r=a*b
        return r
#처럼 __name__속성을 확인하여 사용할 수 있다. 이렇게 하면 타 파일에서 from ~ import su를 하더라도 if문을 통과하지 못하기에 해당 함수를 사용할 수 없다.
#__name__속성은 main값을 갖다가 다른 곳에 모듈로서 사용되면 파일명의 값으로 변경된다.
