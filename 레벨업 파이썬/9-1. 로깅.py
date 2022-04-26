#print 대신 로깅
def hap(a, b):
    ret=a+b
    print(a, b, ret)#함수가 제대로 입출력되었는지를 확인하기 위한 print구분 in 개발. 혹은 파일로 쓰기 위해 로깅 코드를 작성하거나 os의 redirection을 사용할 수 있다
    return ret#redirection예시 python run.py>log.txt      다만 파이썬의 logging모듈을 사용하여 이보다 편리하게 로깅이 가능하다.

result=hap(3,4)


#로깅 기초: logging모듈을 임포트하고 print대신에 logging.info()를 사용하는 것. 인자는 문자열이어야 한다.
import logging

#logging.basicConfig(level=logging.INFO)#로깅의 기본 레벨을 INFO로 상향한다.

def hap(a,b):
    ret=a+b
    logging.info(f"input: {a}, {b}, output={ret}")#로깅 모듈에는 로깅 레벨이 존재하는데, default레벨인 WARNING은 info함수의 레벨보다 낮아 출력되지 않는다.
    return ret#즉, 로깅 레벨을 변경해가면서 레벨에 따른 로깅(추가메시지)를 확인할 수 있다. 위의 logging.basicConfig

result=hap(3,4)#INFO:root:input: 3, 4, output=7


#파일에 로깅하기_basicConfig에 filename을 지정하면 된다
logging.basicConfig(filename="mylog.txt", level=logging.INFO)

#(p.s) 9.2 예외처리 try: except: else: finally:

#(p.s) 9.3 사용자 정의 에러_ 기본적으로 Exception클래스를상속받고, base exception을 상속받아 각 에러를 정의한다.
