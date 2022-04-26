#유닛테스트는 유닛(함수)단위로 진행하는 테스트이며, 리팩터링 과정에 유용하게 사용된다.
#파이썬 유닛 테스트에는 표준 라이브러리의 unittest와 외부의 pytest가 있다.
def average(a):#테스트 대상
    return sum(a)/len(a)

from mymath import *

def test_average():
    asser average([1,2,3])==2
#PyCharm기준, pytest입력 후 test_혹은 _test로 뜨는 파일을 찾아 실행하면 결과를 출력해준다. 자세히 보기 위해 -v옵션을 추가할 수 있다.

def test_average_fail():
    assert average([1,2,3])==1
#위와같이 테스트결과가 실패했을 경우 F가 출력되고 실패한 위치를 출력한다.
