#파이썬 기본 모듈로 thread과 threading이 있는데, 후자가 더 자주 사용된다.
#[threading 모듈 사용하기]
import threading
import time

class Worker(threading.Thread):
    def __init__(self, name):#name지정
        super().__init__()
        self.name=name

    def run(self):
        print("sub thread start", threading.currentThread().getName())#현재 실행중인 name출력
        time.sleep(3)
        print("sub thread end", threading.currentThread().getName())

print("main thread start")
for i in range(5):
    name="thread {}".format(i)
    t=Worker(name)#sub thread의 생성
    t.start()#do run()!
print("main thread end")#서브 스레드가 종료될 때 까지 기다렸다가 메인 스레드가 종료된다!
"""예측
main thread start
sub thread start thread0
sub thread start thread1
sub thread start thread2
sub thread start thread3
sub thread start thread4
main thread end
sub thread end thread0
sub thread end thread1
sub thread end thread2
sub thread end thread3
sub thread end thread4
"""

#[데몬 스레드 만들기_메인 스레드가 종료될 때 자신의 실행상태와 상관없이 종료되는 서브 스레드]
print("main thread start")
for i in range(5):
    name="thread {}".format(i)
    t=Worker(name)
    t.daemon=True#Thread객체의 daemon옵션을 True로 활성화한다.
    t.start()
print("main thread end")#메인 스레드 종료되면 무조건 sub thread들도 종료한다. 
"""아래처럼은 안나오네..결과가
main thread start
sub thread start  thread 0
sub thread start  thread 1
sub thread start  thread 2
sub thread start  thread 3
sub thread start  thread 4
main thread end
"""

#[Fork와 join_메인 스레드가 서브 스레드를 생성하는 것과, 기다리는 것을 의미. 값들을 모아서 순차적으로 처리할 경우 유용]
print("main thread start")

t1=Worker("1")
t1.start()#run

t2=Worker("2")
t2.start()#run

t1.join()#기다린다.
t2.join()

print("main thread post job")
print("main thread end")


#[리스트를 활용한 join()의 호출]
print("main thread start")

threads=[]
for i in range(3):
    thread=Worker(i)
    thread.start()#run하고
    threads.append(thread)#리스트에 추가

for thread in threads:#리스트 속 thread들에 대하여 일괄적으로 join호출
    thread.join()

print("main thread post job")
print("main thread end")


#[루틴(routine)과 서브루틴(subroutine)]
def hap(a, b):#서브루틴
    return a+b

ret=hap(3,4)#호출하는 쪽을 루틴이라고 부르고, 서브루틴이 리턴할때까지 대기한다.
print(ret)

#[코루틴(Coroutine)_기존의 종속적인 루틴과 서브루틴의 관계에서 벗어나 협동을 하는 관계를 의미하며, 와리가리 존나한다. 그리고 앞으론 코딩도장 책봐야겠다. 설명하다가 끊는건 정말 ㅇㅁ없네]
